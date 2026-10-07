"""Re-score a frozen policy with fixed-action, quality-heuristic and abstention baselines.

This measures existing validation/test scenes without training or selecting new policy weights.
The fixed action is selected on validation gains only. Preserve all examples for auditing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tracecv.inference import Detector
from tracecv.policy import evaluate_values, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/trace"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    detector = Detector(args.model, device=args.device)
    if detector.policy is None:
        parser.error("Train the policy first")
    report = json.loads((args.model / "policy-report.json").read_text())
    cache = args.model / "policy-examples.npz"
    preserved = {}
    if cache.exists():
        with np.load(cache) as stored:
            preserved = {name: stored[name] for name in stored.files}
            vx, vy = preserved["validation_features"], preserved["validation_gains"]
            tx, ty = preserved["test_features"], preserved["test_gains"]
    else:
        vx, vy = examples(detector, *report["seed_ranges"]["validation"])
        tx, ty = examples(detector, *report["seed_ranges"]["test"])
    with torch.inference_mode():
        predicted = detector.policy(torch.tensor((tx - detector.feature_mean) / detector.feature_std)).numpy()
    preserved.update(validation_features=vx, validation_gains=vy,
                     test_features=tx, test_gains=ty, test_predicted=predicted)
    np.savez_compressed(cache, **preserved)
    report["test"] = evaluate_values(predicted, ty, fixed_action=int(vy.mean(0).argmax()), observable_features=tx)
    report["test_examples_file"] = cache.name
    report["test_examples_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    report["audit_device"] = args.device
    report["policy_sha256"] = hashlib.sha256((args.model / "policy.pt").read_bytes()).hexdigest()
    (args.model / "policy-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
