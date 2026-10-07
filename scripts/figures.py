"""Render research figures from measured JSON/NPZ artifacts; never synthesize results."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracecv.simulator import ACTIONS

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"]
NAMES = {"trace": "TRACE", "raw": "Raw Siamese", "combined": "Corrected Siamese",
         "pixel_raw": "Pixel distance", "pixel_combined": "Corrected pixel distance"}


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-helper", type=Path, help="Optional directory containing scientific visualization visual_qa.py")
    args = parser.parse_args()
    audit = None
    if args.qa_helper:
        sys.path.insert(0, str(args.qa_helper))
        from visual_qa import audit_layout
        audit = audit_layout
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True})
    output = ROOT / "docs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    evidence = {"figure_contracts": {}, "data_profile": {}, "layout": {}}

    def save(fig, name, contract):
        issues = audit(fig) if audit else []
        evidence["layout"][name] = issues
        if issues:
            raise RuntimeError(f"Repair layout before export: {name}: {issues}")
        evidence["figure_contracts"][name] = contract
        fig.savefig(output / f"{name}.svg")
        fig.savefig(output / f"{name}.png", dpi=300)
        plt.close(fig)

    histories = {name: json.loads((ROOT / f"artifacts/{name}/history.json").read_text())
                 for name in ["trace", "raw", "combined"]}
    learning_rows = [{"model": name, "epoch": row["epoch"], "validation_f1": row["validation"]["f1"]}
                     for name, history in histories.items() for row in history]
    if not all(np.isfinite(row["validation_f1"]) for row in learning_rows):
        raise ValueError("Nonfinite training-history measurements")
    write_csv(output / "learning-source.csv", learning_rows)
    evidence["data_profile"]["learning"] = {name: {"epochs": len(history), "missing": 0,
        "f1_range": [min(r["validation"]["f1"] for r in history), max(r["validation"]["f1"] for r in history)]}
        for name, history in histories.items()}
    fig, ax = plt.subplots(figsize=(7.2, 3.2), layout="constrained")
    for (name, history), color, marker in zip(histories.items(), COLORS, ["o", "s", "^"]):
        ax.plot([r["epoch"] for r in history], [r["validation"]["f1"] for r in history],
                label=NAMES[name], color=color, marker=marker, markersize=4, linewidth=1.5)
    ax.set(xlabel="Training epoch", ylabel="Validation pixel F1", ylim=(0, 1), xticks=range(1, 11),
           title="Validation trajectory of three detector variants")
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "learning", {"claim": "Validation learning trajectories and checkpoint selection are visible.",
         "evidence": "Recorded epoch histories; one training seed, no interval.", "layout": "single quantitative axis",
         "backend": "Matplotlib", "export": "7.2 x 3.2 inches; editable SVG and 300 DPI PNG",
         "choice": "Line plot because epochs are ordered; alternative is epoch points."})

    report = json.loads((ROOT / "artifacts/evaluation/report.json").read_text())
    suites = [("simulation_mixed", "Mixed simulation"), ("simulation_ood", "Stronger nuisance"),
              ("pcd_custom_test", "Real PCD custom split")]
    evidence["data_profile"]["detectors"] = {suite: {"models": 5,
        "independent_units": report["suites"][suite]["trace"]["independent_units"],
        "resampling_unit": report["suites"][suite]["trace"]["bootstrap_unit"],
        "f1_range": [min(report["suites"][suite][name]["f1"] for name in NAMES),
                     max(report["suites"][suite][name]["f1"] for name in NAMES)],
        "missing": 0} for suite, _ in suites}
    if not all(np.isfinite(report["suites"][suite][name]["f1"]) and
               np.isfinite(report["suites"][suite][name]["bootstrap_f1_ci95"]).all()
               for suite, _ in suites for name in NAMES):
        raise ValueError("Nonfinite detector measurements")
    detector_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True, layout="constrained")
    for ax, (suite, title) in zip(axes, suites):
        metrics = report["suites"][suite]
        for i, (name, color, marker) in enumerate(zip(NAMES, COLORS, ["o", "s", "^", "D", "v"])):
            row = metrics[name]
            lower, upper = row["bootstrap_f1_ci95"]
            ax.hlines(i, lower, upper, color=color, linewidth=1.2)
            ax.plot([lower, upper], [i, i], marker="|", color=color, linestyle="none", markersize=6)
            ax.plot(row["f1"], i, marker=marker, color=color, linestyle="none", markersize=5)
            detector_rows.append({"suite": suite, "model": name, "f1": row["f1"], "ci95_lower": lower,
                                  "ci95_upper": upper, "n_units": row["independent_units"], "unit": row["bootstrap_unit"]})
        ax.set(xlim=(0, 1), xlabel="Pooled pixel F1", xticks=[0, .5, 1],
               title=f"{title}\nn = {metrics['trace']['independent_units']} {metrics['trace']['bootstrap_unit']}s")
        ax.grid(axis="x", alpha=.2)
    axes[0].set(yticks=range(5), yticklabels=list(NAMES.values()))
    axes[0].invert_yaxis()
    write_csv(output / "detectors-source.csv", detector_rows)
    save(fig, "detectors", {"claim": "The benefit of pooling depends on the evaluation domain.",
         "evidence": "Frozen-checkpoint pooled F1 and 95% percentile intervals from 1000 scene/panorama bootstrap replicates.",
         "layout": "three quantitative facets", "backend": "Matplotlib", "export": "7.2 x 3.4 inches; editable SVG and 300 DPI PNG",
         "choice": "Point/interval plot exposes estimation uncertainty; alternatives: paired difference plot or numeric table."})

    policy_report = json.loads((ROOT / "artifacts/trace/policy-report.json").read_text())
    with np.load(ROOT / "artifacts/trace/policy-examples.npz") as data:
        actual, predicted, features = data["test_gains"], data["test_predicted"], data["test_features"]
    if not all(np.isfinite(a).all() for a in [actual, predicted, features]):
        raise ValueError("Nonfinite policy measurements")
    n = len(actual)
    choice = predicted.argmax(1)
    selected = actual[np.arange(n), choice]
    fixed = ACTIONS.index(policy_report["test"]["validation_selected_fixed_action"])
    heuristic = np.where(np.abs(features[:, 0]) + np.abs(features[:, 1]) >= 2, 1,
                         np.where(features[:, 3] - features[:, 2] > .02, 0, 2))
    gains = {"Learned": selected, "Abstain": np.where(predicted.max(1) > 0, selected, 0),
             "Uniform\nexpected": actual.mean(1), "Fixed\n(validation)": actual[:, fixed],
             "Quality\nheuristic": actual[np.arange(n), heuristic], "Passive": np.zeros(n), "Oracle": actual.max(1)}
    policy_rows = [{"test_scene": 6_000_000 + i, "strategy": name.replace("\n", " "), "brier_reduction": float(value)}
                   for name, values in gains.items() for i, value in enumerate(values)]
    write_csv(output / "policy-source.csv", policy_rows)
    evidence["data_profile"]["policy"] = {"scenes": n, "actions_per_scene": 3, "missing": 0,
        "measured_gain_range": [float(actual.min()), float(actual.max())],
        "negative_action_fraction": float((actual < 0).mean()),
        "strategy_means": {name.replace("\n", " "): float(values.mean()) for name, values in gains.items()}}
    fig, ax = plt.subplots(figsize=(7.2, 3.7), layout="constrained")
    ax.boxplot(list(gains.values()), tick_labels=list(gains), showfliers=False,
               medianprops={"color": "black"}, widths=.5)
    rng = np.random.default_rng(731)
    for i, values in enumerate(gains.values(), 1):
        ax.scatter(i + rng.uniform(-.16, .16, n), values, s=7, alpha=.45,
                   color=COLORS[(i-1) % len(COLORS)], linewidths=0)
        ax.scatter(i, values.mean(), marker="D", s=25, color="black", zorder=5)
    ax.axhline(0, color="black", linewidth=.8, linestyle="--")
    ax.set(ylabel="Measured Brier reduction after follow-up", title=f"A second look can help or harm: {n} held-out scenes")
    ax.grid(axis="y", alpha=.2)
    save(fig, "policy", {"claim": "Average policy value can conceal harmful individual follow-ups.",
         "evidence": "Scene-level gains from three actual simulator acquisitions; points retain every observation, diamond means, box IQR, whiskers 1.5 IQR.",
         "layout": "single quantitative axis", "backend": "Matplotlib", "export": "7.2 x 3.7 inches; editable SVG and 300 DPI PNG",
         "choice": "Box/strip plot preserves dispersion; alternatives: gain histogram or empirical CDF."})
    (output / "figure-audit.json").write_text(json.dumps(evidence, indent=2))
    (output / "captions.md").write_text(
        "# Research figure captions\n\n"
        "**Learning.** Validation pixel F1 before post-training temperature/threshold calibration, over ten epochs. "
        "Each variant uses 752 training examples and 176 validation examples; one seed (42), no multi-run interval. "
        "Checkpoint selection uses validation only.\n\n"
        "**Detectors.** Pooled pixel F1 for five frozen detectors. Bars are 95% percentile confidence intervals from "
        "1000 nonparametric bootstrap replicates. Simulation uses 128 mixed and 64 stronger-nuisance independent scenes. "
        "Real data uses 20 held-out panoramas, grouping four crops per panorama. Intervals quantify test-sample "
        "variation, not training randomness; geographic independence is unverified.\n\n"
        "**Policy.** Measured Brier-error reduction over 128 independent simulator test scenes. Each strategy is "
        "evaluated on the same scenes. Points show all scene gains; diamonds are means; boxes show medians and IQR, "
        "whiskers extend to 1.5 IQR. Uniform is its exact per-scene expected gain over three actions. The fixed action "
        "uses validation-only selection; the quality heuristic is predeclared. Passive has zero gain. Oracle uses "
        "ground truth and is unattainable at inference. Negative values mean the chosen acquisition made error worse. "
        "Abstention chooses passive when all predicted gains are nonpositive. No significance symbols are used.\n")
    print(json.dumps(evidence["data_profile"], indent=2))


if __name__ == "__main__":
    main()
