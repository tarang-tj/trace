"""Verify frozen checkpoint identities, split boundaries and exported research evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tracecv.inference import FEATURES
from tracecv.simulator import ACTIONS

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    report = read("artifacts/evaluation/report.json")
    checks = {}
    for name in ("trace", "raw", "combined"):
        provenance = read(f"artifacts/{name}/provenance.json")
        config = read(f"artifacts/{name}/config.json")
        history = read(f"artifacts/{name}/history.json")
        sha = digest(f"artifacts/{name}/model.pt")
        assert sha == provenance["checkpoint_sha256"] == report["checkpoints"][name]
        assert len(history) == config["epochs"] == 10
        assert [row["epoch"] for row in history] == list(range(1, 11))
        assert provenance["test_used_for_selection"] is False
        splits = read(f"artifacts/{name}/pcd-splits.json")
        assert [len(splits[key]) for key in ("train", "validation", "test")] == [60, 20, 20]
        assert len(set().union(*map(set, splits.values()))) == 100
        assert splits == read("artifacts/trace/pcd-splits.json")
        checks[name] = {"sha256": sha, "epochs": len(history), "panorama_split": [60, 20, 20]}
    policy_report = read("artifacts/trace/policy-report.json")
    policy = torch.load(ROOT / "artifacts/trace/policy.pt", map_location="cpu", weights_only=True)
    assert policy["detector_sha256"] == policy_report["detector_sha256"] == checks["trace"]["sha256"]
    assert policy["features"] == list(FEATURES)
    assert digest("artifacts/trace/policy.pt") == policy_report["policy_sha256"]
    assert digest("artifacts/trace/policy-examples.npz") == policy_report["test_examples_sha256"]
    with np.load(ROOT / "artifacts/trace/policy-examples.npz") as data:
        actual, predicted = data["test_gains"], data["test_predicted"]
        assert actual.shape == predicted.shape == (128, len(ACTIONS))
        assert all(np.isfinite(data[key]).all() for key in data.files)
        selected = actual[np.arange(len(actual)), predicted.argmax(1)]
        assert np.isclose(selected.mean(), policy_report["test"]["learned_mean_brier_reduction"])
    assert report["suites"]["pcd_custom_test"]["trace"]["independent_units"] == 20
    assert report["suites"]["pcd_custom_test"]["trace"]["bootstrap_unit"] == "panorama"
    browser = read("artifacts/studio/qa.json")
    assert browser["exportedCheckpoint"] == checks["trace"]["sha256"] and not browser["errors"]
    assert browser["reobservation"] == "acquired a simulated follow-up and updated inference"
    assert browser["research"] == "six frozen detector suites and seven action strategies verified"
    assert browser["aboveFoldCopyDiff"] == {
        "passed": True,
        "added": [],
        "removed": [],
        "renamed": [],
        "reordered": False,
        "closedSelectOptions": browser["aboveFoldCopyDiff"]["closedSelectOptions"],
    }
    figures = read("docs/figures/figure-audit.json")
    assert all(not issues for issues in figures["layout"].values())
    checks["policy"] = {"sha256": policy_report["policy_sha256"], "test_scenes": 128}
    checks["browser"] = {
        "desktop": browser["viewport"], "mobile": browser["mobile"],
        "errors": browser["errors"], "copy_diff_passed": True, "research": browser["research"],
    }
    checks["figures"] = {"layout_issues": figures["layout"]}
    (ROOT / "artifacts/verification.json").write_text(json.dumps(checks, indent=2))
    print("Checkpoint/report identity, panorama splits, policy cache and final browser evidence verified.")


if __name__ == "__main__":
    main()
