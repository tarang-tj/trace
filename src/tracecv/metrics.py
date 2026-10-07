import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


def score(probability, target, valid=None, threshold=0.5):
    p, y = np.asarray(probability), np.asarray(target)
    if valid is not None:
        keep = np.asarray(valid) > 0.5
        p, y = p[keep], y[keep]
    p, y = p.ravel(), y.ravel() > 0.5
    pred = p >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(float(p[bins == i].mean()) - float(y[bins == i].mean())) *
              float((bins == i).mean()) for i in range(10) if (bins == i).any())
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "brier": float(np.mean((p - y) ** 2)), "ece": ece,
            "false_positive_rate": fp / max(fp + tn, 1),
            "pixels": len(p), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def calibrate(logits, targets):
    logits, targets = np.asarray(logits).ravel(), np.asarray(targets).ravel()
    result = minimize_scalar(lambda log_t: float(np.mean(np.logaddexp(0, logits / np.exp(log_t)) -
                                       targets * logits / np.exp(log_t))), bounds=(-2, 2), method="bounded")
    temperature = float(np.exp(result.x))
    probability = expit(logits / temperature)
    thresholds = np.linspace(0.05, 0.95, 37)
    threshold = float(max(thresholds, key=lambda t: score(probability, targets, threshold=t)["f1"]))
    return {"temperature": temperature, "threshold": threshold,
            "validation": score(probability, targets, threshold=threshold),
            "uncalibrated": score(expit(logits), targets, threshold=0.5)}
