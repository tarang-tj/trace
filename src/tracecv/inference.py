"""Label-free inference and auditable observable diagnostics."""
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import find_objects, label
from scipy.special import expit

from .interventions import VERTICES, build_lattice
from .model import TRACE, ActionValueNet

FEATURES = ("translation_dy", "translation_dx", "photometric_residual", "raw_residual",
            "registered_residual", "sharpness", "probability_mean", "probability_std",
            "probability_q90", "error_mean", "branch_disagreement", "trust_entropy")


def png_uri(array):
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def heatmap(image, probability):
    # Blue through cyan/yellow to red; the scale always represents [0, 1].
    p = np.clip(probability, 0, 1)
    color = np.stack([np.clip(1.5 - np.abs(4 * p - 3), 0, 1),
                      np.clip(1.5 - np.abs(4 * p - 2), 0, 1),
                      np.clip(1.5 - np.abs(4 * p - 1), 0, 1)], -1)
    alpha = (.08 + .68 * p)[..., None]
    return np.clip(image * (1 - alpha) + color * alpha, 0, 1)


def read_image(data, size=96):
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 16_000_000:
            raise ValueError("Images must contain at most 16 million pixels")
        image = ImageOps.exif_transpose(image).convert("RGB")
        dimensions = image.size
        if min(dimensions) < 32:
            raise ValueError("Images must be at least 32 pixels in each dimension")
        array = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR)).astype(np.float32) / 255
    return array, dimensions


class Detector:
    def __init__(self, directory="artifacts/trace", device="cpu"):
        self.directory = Path(directory)
        checkpoint_path = self.directory / "model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.variant = checkpoint["variant"]
        self.device = torch.device(device)
        self.model = TRACE(checkpoint["width"], self.variant == "trace").to(self.device).eval()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.calibration = json.loads((self.directory / "calibration.json").read_text())
        self.sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        self.policy = None
        self.policy_sha256 = None
        if (self.directory / "policy.pt").exists():
            self.policy_sha256 = hashlib.sha256((self.directory / "policy.pt").read_bytes()).hexdigest()
            policy = torch.load(self.directory / "policy.pt", map_location="cpu", weights_only=True)
            if policy["detector_sha256"] != self.sha256:
                raise ValueError("Policy belongs to a different detector checkpoint")
            if policy["features"] != list(FEATURES):
                raise ValueError("Policy feature schema does not match this runtime")
            self.policy = ActionValueNet().eval()
            self.policy.load_state_dict(policy["state_dict"])
            self.feature_mean = np.array(policy["feature_mean"], np.float32)
            self.feature_std = np.array(policy["feature_std"], np.float32)

    @torch.inference_mode()
    def predict(self, reference, observation):
        start = time.perf_counter()
        vertices, stats = build_lattice(reference, observation)
        a = torch.from_numpy(reference.transpose(2, 0, 1).copy())[None].to(self.device)
        v = torch.from_numpy(vertices.transpose(0, 3, 1, 2).copy())[None].to(self.device)
        if self.variant == "combined":
            v = v[:, 3:4]
        output = self.model(a, v)
        logits = output["logits"][0, 0].cpu().numpy()
        probability = expit(logits / self.calibration["temperature"])
        error = expit(output["error_logits"][0, 0].cpu().numpy() / self.calibration["error_temperature"])
        branches = expit(output["branches"][0].cpu().numpy() / self.calibration["temperature"])
        weights = output["weights"][0].cpu().numpy()
        stats.update(probability_mean=float(probability.mean()), probability_std=float(probability.std()),
                     probability_q90=float(np.quantile(probability, .9)), error_mean=float(error.mean()),
                     branch_disagreement=float(branches.std(0).mean()),
                     trust_entropy=float(-(weights * np.log(weights + 1e-8)).sum(0).mean()))
        features = np.array([stats[name] for name in FEATURES], np.float32)
        mask = probability >= self.calibration["threshold"]
        # Training excludes unobserved borders. Exclude five pixels in reported area too.
        valid = np.ones(mask.shape, bool)
        valid[:5] = valid[-5:] = False
        valid[:, :5] = valid[:, -5:] = False
        mask &= valid
        components, _ = label(mask)
        regions = []
        for i, box in enumerate(find_objects(components), 1):
            if box is None:
                continue
            pixels = components == i
            count = int(pixels.sum())
            if count >= 4:
                regions.append({"box_xyxy": [box[1].start, box[0].start, box[1].stop, box[0].stop],
                                "pixels": count, "mean_probability": float(probability[pixels].mean())})
        actions = []
        if self.policy is not None:
            values = self.policy(torch.from_numpy((features - self.feature_mean) / self.feature_std)[None])[0].numpy()
            from .simulator import ACTIONS
            actions = sorted([{"action": name, "predicted_brier_reduction": float(value)}
                              for name, value in zip(ACTIONS, values)],
                             key=lambda item: item["predicted_brier_reduction"], reverse=True)
        return {"probability": probability, "error": error, "branches": branches, "weights": weights,
                "features": features, "diagnostics": stats, "mask": mask, "valid": valid,
                "regions": regions, "actions": actions,
                "changed_fraction": float(mask.sum() / valid.sum()),
                "elapsed_ms": (time.perf_counter() - start) * 1000}

    def evidence(self, reference, observation, result=None):
        result = self.predict(reference, observation) if result is None else result
        return {"schema_version": 1, "model_sha256": self.sha256, "policy_sha256": self.policy_sha256,
                "variant": self.variant,
                "reference_array_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                "observation_array_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
                "processing_size": list(reference.shape[:2]), "threshold": self.calibration["threshold"],
                "temperature": self.calibration["temperature"], "changed_fraction": result["changed_fraction"],
                "mean_error_indicator": float(result["error"][result["valid"]].mean()),
                "diagnostics": result["diagnostics"], "regions": result["regions"], "actions": result["actions"],
                "elapsed_ms": result["elapsed_ms"],
                "vertices": list(VERTICES) if self.variant == "trace" else [self.variant],
                "branch_mean_probabilities": [float(x[result["valid"]].mean()) for x in result["branches"]],
                "probability_image": png_uri(result["probability"]),
                "error_image": png_uri(result["error"]), "mask_image": png_uri(result["mask"].astype(float)),
                "probability_overlay": png_uri(heatmap(reference, result["probability"])),
                "error_overlay": png_uri(heatmap(reference, result["error"])),
                "limitations": ["Computational probes do not establish causality.",
                                 "Error indicator is learned on validation-domain data; domain shifts can invalidate it.",
                                 "Action values are trained and evaluated only in the simulator."]}
