"""Learn follow-up values from actual interventions, on detector-disjoint scenes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .inference import FEATURES, Detector
from .model import ActionValueNet
from .simulator import ACTIONS, make_scene
from .train import seed_everything


def examples(detector, start, count):
    features, gains = [], []
    for index in range(count):
        scene = make_scene(start + index)
        observation, valid = scene.observe()
        before = detector.predict(scene.reference, observation)
        later = [scene.observe(action) for action in ACTIONS]
        common = valid.astype(bool) & before["valid"]
        for _, action_valid in later:
            common &= action_valid.astype(bool)
        initial = float(((before["probability"] - scene.mask) ** 2)[common].mean())
        values = []
        for after, _ in later:
            p = detector.predict(scene.reference, after)["probability"]
            values.append(initial - float(((p - scene.mask) ** 2)[common].mean()))
        features.append(before["features"])
        gains.append(values)
        if (index + 1) % 32 == 0:
            print(f"Measured {index + 1}/{count} intervention scenes", flush=True)
    return np.stack(features), np.array(gains, np.float32)


def evaluate_values(predicted, actual, seed=42, fixed_action=None, observable_features=None):
    choice = predicted.argmax(1)
    selected = actual[np.arange(len(actual)), choice]
    random = actual.mean(1)  # Exact expected value of uniform random, no sampling noise.
    oracle = actual.max(1)
    rng = np.random.default_rng(seed)
    improvements = np.array([(selected[idx] - random[idx]).mean()
                             for idx in rng.integers(0, len(actual), (1000, len(actual)))])
    report = {"scenes": len(actual), "learned_mean_brier_reduction": float(selected.mean()),
            "uniform_random_mean_brier_reduction": float(random.mean()),
            "passive_mean_brier_reduction": 0., "oracle_mean_brier_reduction": float(oracle.mean()),
            "oracle_regret": float((oracle - selected).mean()),
            "top_action_accuracy": float((choice == actual.argmax(1)).mean()),
            "harmful_action_fraction": float((selected < 0).mean()),
            "paired_gain_over_random_ci95": np.quantile(improvements, [.025, .975]).tolist(),
            "action_counts": {name: int((choice == i).sum()) for i, name in enumerate(ACTIONS)},
            "per_action_mean_brier_reduction": dict(zip(ACTIONS, actual.mean(0).tolist())),
            "value_rmse": float(np.sqrt(((predicted - actual) ** 2).mean()))}
    capture = predicted.max(1) > 0
    report.update(abstaining_policy_mean_brier_reduction=float(np.where(capture, selected, 0).mean()),
                  abstention_fraction=float((~capture).mean()))
    if fixed_action is not None:
        report["validation_selected_fixed_action"] = ACTIONS[fixed_action]
        report["fixed_action_mean_brier_reduction"] = float(actual[:, fixed_action].mean())
    if observable_features is not None:
        # Predeclared image-quality heuristic, not tuned against test labels.
        f = observable_features
        heuristic = np.where(np.abs(f[:, 0]) + np.abs(f[:, 1]) >= 2, 1,
                             np.where(f[:, 3] - f[:, 2] > .02, 0, 2))
        report["quality_heuristic_mean_brier_reduction"] = float(actual[np.arange(len(actual)), heuristic].mean())
        report["quality_heuristic_rule"] = "L1 estimated shift >=2: register; raw-minus-relit residual >.02: relight; otherwise focus."
    strategies = {"uniform_random": random, "passive": np.zeros(len(actual))}
    if fixed_action is not None:
        strategies["fixed_action"] = actual[:, fixed_action]
    if observable_features is not None:
        strategies["quality_heuristic"] = actual[np.arange(len(actual)), heuristic]
    index = np.random.default_rng(seed).integers(0, len(actual), (1000, len(actual)))
    report["paired_learned_minus_baseline_ci95"] = {
        name: np.quantile((selected[index] - values[index]).mean(1), [.025, .975]).tolist()
        for name, values in strategies.items()}
    report["oracle_with_passive_mean_brier_reduction"] = float(np.maximum(oracle, 0).mean())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/trace")
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()
    seed_everything(42)
    detector = Detector(args.model)
    groups = [examples(detector, start, count) for start, count in
              [(4_000_000, args.train), (5_000_000, args.validation), (6_000_000, args.test)]]
    (x, y), (vx, vy), (tx, ty) = groups
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-4)
    test_features = tx.copy()
    x, vx, tx = [torch.tensor((data - mean) / std) for data in [x, vx, tx]]
    target, vt = torch.tensor(y * 100), torch.tensor(vy * 100)
    model = ActionValueNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    best, best_state = float("inf"), None
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(x))
        for batch in order.split(32):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x[batch]), target[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            error = float(torch.nn.functional.mse_loss(model(vx), vt))
        history.append(error)
        if error < best:
            best, best_state = error, {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    # Convert network outputs back to Brier units, preserving deployment architecture.
    with torch.no_grad():
        model.net[-1].weight.div_(100)
        model.net[-1].bias.div_(100)
        predicted = model(tx).numpy()
    directory = Path(args.model)
    np.savez_compressed(directory / "policy-examples.npz", train_features=groups[0][0],
                        train_gains=y, validation_features=groups[1][0], validation_gains=vy,
                        test_features=test_features, test_gains=ty, test_predicted=predicted)
    torch.save({"state_dict": model.state_dict(), "feature_mean": mean.tolist(), "feature_std": std.tolist(),
                "detector_sha256": detector.sha256, "features": list(FEATURES)}, directory / "policy.pt")
    report = {"protocol": "Detector-disjoint seeds; labels only construct training targets and score test actions.",
              "seed_ranges": {"train": [4_000_000, args.train], "validation": [5_000_000, args.validation],
                              "test": [6_000_000, args.test]},
              "features": list(FEATURES), "validation_best_scaled_mse": best,
              "test": evaluate_values(predicted, ty, fixed_action=int(vy.mean(0).argmax()),
                                      observable_features=test_features), "epochs": args.epochs,
              "detector_sha256": detector.sha256, "validation_history": history}
    report["test_examples_file"] = "policy-examples.npz"
    (directory / "policy-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
