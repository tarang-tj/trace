"""Frozen-checkpoint evaluation with scene bootstrap and untouched test images."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from .data import PCDDataset
from .inference import Detector
from .interventions import build_lattice
from .metrics import calibrate, score
from .simulator import make_scene


def simulated(start, count, scenario="mixed", severity=1):
    for index in range(count):
        scene = make_scene(start + index, scenario=scenario, severity=severity)
        observation, valid = scene.observe()
        yield scene.reference, observation, scene.mask, valid


def pixel_logits(a, b):
    vertices, _ = build_lattice(a, b)
    return [logit(np.clip(np.abs(a - query).mean(-1) * 4, .0001, .9999))
            for query in (b, vertices[3])]


def grouped_counts(rows, group_size=1):
    counts = np.array([[row[k] for k in ("tp", "fp", "fn")] for row in rows])
    if len(counts) % group_size:
        raise ValueError("Bootstrap groups must contain complete independent units")
    return counts.reshape(-1, group_size, 3).sum(1)


def f1_interval(rows, group_size=1):
    counts = grouped_counts(rows, group_size)
    rng = np.random.default_rng(102)
    sums = counts[rng.integers(len(counts), size=(1000, len(counts)))].sum(1)
    values = 2 * sums[:, 0] / np.maximum(2 * sums[:, 0] + sums[:, 1] + sums[:, 2], 1)
    return np.quantile(values, [.025, .975]).tolist()


def paired_f1_interval(a, b, group_size=1):
    ca, cb = grouped_counts(a, group_size), grouped_counts(b, group_size)
    index = np.random.default_rng(102).integers(len(ca), size=(1000, len(ca)))
    def values(rows):
        c = rows[index].sum(1)
        return 2 * c[:, 0] / np.maximum(2 * c[:, 0] + c[:, 1] + c[:, 2], 1)
    return np.quantile(values(ca) - values(cb), [.025, .975]).tolist()


def selective_risk(p, y, error, threshold):
    wrong = (p >= threshold) != (y >= .5)
    order = np.argsort(error, kind="stable")
    entropy = -(p * np.log(p + 1e-8) + (1 - p) * np.log(1 - p + 1e-8))
    ent_order = np.argsort(entropy, kind="stable")
    return [{"coverage": coverage,
             "learned_error_risk": float(wrong[order[:max(1, int(len(p) * coverage))]].mean()),
             "entropy_risk": float(wrong[ent_order[:max(1, int(len(p) * coverage))]].mean()),
             "random_expected_risk": float(wrong.mean())} for coverage in [1., .75, .5, .25]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["artifacts/trace", "artifacts/raw", "artifacts/combined"])
    parser.add_argument("--real-data", default="data/pcd")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--output", default="artifacts/evaluation")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    detectors = {Path(path).name: Detector(path) for path in args.models}
    validation = list(simulated(1_000_000, 96))
    if args.real_data:
        real_val = PCDDataset(args.real_data, "validation")
        validation += [real_val.arrays(i) for i in range(len(real_val))]
    pl, targets = [[], []], []
    for a, b, y, valid in validation:
        keep = valid.astype(bool)
        for index, values in enumerate(pixel_logits(a, b)):
            pl[index].append(values[keep])
        targets.append(y[keep])
    pixel_calibration = {name: calibrate(np.concatenate(values), np.concatenate(targets))
                         for name, values in zip(["pixel_raw", "pixel_combined"], pl)}
    suites = {"simulation_mixed": list(simulated(2_000_000, args.count)),
              "simulation_ood": list(simulated(3_000_000, args.count // 2, severity=2.))}
    for scenario, start in [("lighting", 2_100_000), ("camera", 2_200_000), ("blur", 2_300_000)]:
        suites[f"nuisance_{scenario}"] = list(simulated(start, 32, scenario))
    if args.real_data:
        dataset = PCDDataset(args.real_data, "test")
        suites["pcd_custom_test"] = [dataset.arrays(i) for i in range(len(dataset))]
    report = {"protocol": "Custom frozen panorama split; no test selection; 1000 bootstrap replicates over simulated scenes or real panoramas (four crops grouped).",
              "caveat": "Custom PCD split and resized crops are not comparable to published benchmark scores.",
              "checkpoints": {name: model.sha256 for name, model in detectors.items()},
              "pixel_validation_calibration": pixel_calibration, "suites": {}}
    predictions = {}
    for suite, scenes in suites.items():
        group_size = 4 if suite == "pcd_custom_test" else 1
        buffers = {name: {"p": [], "y": [], "error": [], "rows": [], "latency": []}
                   for name in [*detectors, *pixel_calibration]}
        for a, b, y, valid in scenes:
            keep = valid.astype(bool)
            for name, detector in detectors.items():
                result = detector.predict(a, b)
                p = result["probability"][keep]
                target = y[keep]
                buffer = buffers[name]
                buffer["p"].append(p)
                buffer["y"].append(target)
                buffer["error"].append(result["error"][keep])
                buffer["latency"].append(result["elapsed_ms"])
                buffer["rows"].append(score(p, target, threshold=detector.calibration["threshold"]))
            for (name, calibration), values in zip(pixel_calibration.items(), pixel_logits(a, b)):
                p = expit(values[keep] / calibration["temperature"])
                buffer = buffers[name]
                buffer["p"].append(p)
                buffer["y"].append(y[keep])
                buffer["rows"].append(score(p, y[keep], threshold=calibration["threshold"]))
        suite_report = {}
        for name, buffer in buffers.items():
            p, y = np.concatenate(buffer["p"]), np.concatenate(buffer["y"])
            threshold = detectors[name].calibration["threshold"] if name in detectors else pixel_calibration[name]["threshold"]
            metrics = score(p, y, threshold=threshold)
            metrics.update(scenes=len(scenes), independent_units=len(scenes) // group_size,
                           bootstrap_unit="panorama" if group_size == 4 else "scene",
                           bootstrap_f1_ci95=f1_interval(buffer["rows"], group_size))
            if buffer["error"]:
                error = np.concatenate(buffer["error"])
                wrong = ((p >= threshold) != (y >= .5)).astype(float)
                error_scores = score(error, wrong)
                metrics["error_indicator_brier"] = error_scores["brier"]
                metrics["error_indicator_ece"] = error_scores["ece"]
                metrics["observed_pixel_error_rate"] = float(wrong.mean())
                metrics["selective_risk"] = selective_risk(p, y, error, threshold)
                metrics["latency_ms_p50_p95"] = np.quantile(buffer["latency"], [.5, .95]).tolist()
            suite_report[name] = metrics
            predictions[f"{suite}_{name}_scene_counts"] = np.array(
                [[r[k] for k in ("tp", "fp", "fn", "tn")] for r in buffer["rows"]])
        if "trace" in buffers:
            suite_report["paired_trace_minus_baseline_f1_ci95"] = {
                name: paired_f1_interval(buffers["trace"]["rows"], buffers[name]["rows"], group_size)
                for name in buffers if name != "trace"}
        report["suites"][suite] = suite_report
        print(suite, json.dumps({name: {k: m[k] for k in ["f1", "brier", "false_positive_rate"]}
                                for name, m in suite_report.items() if "f1" in m}), flush=True)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "scene-confusions.npz", **predictions)
    (destination / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
