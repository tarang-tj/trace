"""Four vertices of an acquisition-intervention lattice, built without labels.

None, photometric matching, translation registration, both. These are computational
probes, not causal proof and not newly captured sensor observations.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, shift

VERTICES = ("observed", "relit", "registered", "combined")


def photometric_match(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    # Robust central quantiles limit influence of small physical changes.
    result = np.empty_like(query)
    for channel in range(3):
        a, b = reference[..., channel], query[..., channel]
        aq, bq = np.quantile(a, [0.15, 0.5, 0.85]), np.quantile(b, [0.15, 0.5, 0.85])
        gain = np.clip((aq[2] - aq[0]) / max(bq[2] - bq[0], 0.025), 0.4, 2.5)
        result[..., channel] = np.clip((b - bq[1]) * gain + aq[1], 0, 1)
    return result


def estimate_translation(reference: np.ndarray, query: np.ndarray, radius: int = 4) -> tuple[int, int]:
    # Gradients reject global exposure shifts. Trim high residuals (possible changes).
    a = gaussian_filter(reference.mean(-1), 0.6)
    b = gaussian_filter(query.mean(-1), 0.6)
    ag = np.stack(np.gradient(a), -1)
    bg = np.stack(np.gradient(b), -1)
    border = radius + 2
    inner = (slice(border, -border), slice(border, -border))
    best = (float("inf"), 0, 0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            rolled = np.roll(bg, (dy, dx), axis=(0, 1))
            residual = np.abs(ag[inner] - rolled[inner]).mean(-1).ravel()
            cutoff = np.quantile(residual, 0.8)
            score = float(np.minimum(residual, cutoff).mean()) + 1e-6 * (abs(dx) + abs(dy))
            if score < best[0]:
                best = (score, dy, dx)
    return best[1], best[2]


def build_lattice(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, dict]:
    if reference.shape != query.shape or reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError("Reference and query must be equally sized H×W×3 arrays")
    dy, dx = estimate_translation(reference, query)
    registered = shift(query, (dy, dx, 0), order=1, mode="nearest").astype(np.float32)
    relit = photometric_match(reference, query)
    combined = photometric_match(reference, registered)
    vertices = np.stack([query, relit, registered, combined]).astype(np.float32)
    # These observed quality statistics are available to the planner, not hidden sensor values.
    stats = {"translation_dy": dy, "translation_dx": dx,
             "photometric_residual": float(np.abs(reference - relit).mean()),
             "raw_residual": float(np.abs(reference - query).mean()),
             "registered_residual": float(np.abs(reference - combined).mean()),
             "sharpness": float(np.abs(np.diff(query.mean(-1), axis=0)).mean())}
    return vertices, stats
