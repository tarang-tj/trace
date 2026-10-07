"""Deterministic paired-scene renderer with independently intervenable sensor nuisance.

Labels live in the reference coordinate frame. Re-observation changes acquisition,
never scene geometry or labels. Seeds define entire scenes, not individual patches.
"""
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, shift

ACTIONS = ("relight", "reregister", "focus")
SCENARIOS = ("mixed", "moved", "removed", "added", "lighting", "camera", "blur", "clean")
SPLIT_START = {"train": 0, "validation": 1_000_000, "test": 2_000_000, "ood": 3_000_000}


@dataclass(frozen=True)
class Sensor:
    gain: float = 1.0
    bias: float = 0.0
    shadow: float = 0.0
    shadow_x: float = 0.5
    dx: int = 0
    dy: int = 0
    blur: float = 0.0
    noise: float = 0.0


@dataclass
class Scene:
    reference: np.ndarray
    latent: np.ndarray
    mask: np.ndarray
    sensor: Sensor
    seed: int
    kind: str

    def observe(self, action: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        sensor = self.sensor
        if action == "relight":
            sensor = replace(sensor, gain=1.0, bias=0.0, shadow=0.0)
        elif action == "reregister":
            sensor = replace(sensor, dx=0, dy=0)
        elif action == "focus":
            sensor = replace(sensor, blur=0.0, noise=0.0)
        elif action is not None:
            raise ValueError(f"Unknown action: {action}")
        return acquire(self.latent, sensor, self.seed + 47)


def acquire(image: np.ndarray, sensor: Sensor, seed: int) -> tuple[np.ndarray, np.ndarray]:
    h, w, _ = image.shape
    x = np.arange(w, dtype=np.float32)[None, :] / w
    shade = 1 - sensor.shadow * np.exp(-((x - sensor.shadow_x) / 0.22) ** 2)
    out = np.clip(image * sensor.gain * shade[..., None] + sensor.bias, 0, 1)
    if sensor.blur:
        out = gaussian_filter(out, (sensor.blur, sensor.blur, 0))
    out = shift(out, (sensor.dy, sensor.dx, 0), order=1, mode="nearest")
    if sensor.noise:
        out += np.random.default_rng(seed).normal(0, sensor.noise, out.shape)
    # Since the model performs global registration, exclude any unknown border.
    valid = np.ones((h, w), np.float32)
    border = max(abs(sensor.dx), abs(sensor.dy)) + 5
    valid[:border] = valid[-border:] = 0
    valid[:, :border] = valid[:, -border:] = 0
    return np.clip(out, 0, 1).astype(np.float32), valid


def _draw_part(canvas, box, color, kind):
    draw = ImageDraw.Draw(canvas)
    x, y, w, h = box
    bounds = (x, y, x + w, y + h)
    if isinstance(color, int):
        if kind == 0:
            draw.rounded_rectangle(bounds, radius=3, fill=color)
        elif kind == 1:
            draw.ellipse(bounds, fill=color)
        else:
            draw.polygon([(x, y + h), (x + w // 2, y), (x + w, y + h)], fill=color)
        return
    darker = tuple(max(0, c - 28) for c in color)
    if kind == 0:
        draw.rounded_rectangle(bounds, radius=3, fill=color, outline=darker, width=2)
        draw.line((x + 3, y + 3, x + w - 3, y + 3), fill=tuple(min(255, c + 35) for c in color))
    elif kind == 1:
        draw.ellipse(bounds, fill=color, outline=darker, width=2)
        draw.ellipse((x + w // 3, y + h // 3, x + 2 * w // 3, y + 2 * h // 3), fill=(65, 70, 68))
    else:
        draw.polygon([(x, y + h), (x + w // 2, y), (x + w, y + h)], fill=color)


def make_scene(seed: int, size: int = 96, scenario: str = "mixed", severity: float = 1.0) -> Scene:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if size < 32:
        raise ValueError("Scene size must be at least 32")
    rng = np.random.default_rng(seed)
    # Continuous surfaces and colors make held-out seeds different physical scenes.
    background = rng.uniform(100, 180, 3)
    grain = gaussian_filter(rng.normal(0, 7, (size, size)), 0.7)
    y, x = np.mgrid[:size, :size]
    texture = np.sin(x * rng.uniform(0.1, 0.35)) * rng.uniform(0, 3) + grain
    bg = np.clip(background[None, None] + texture[..., None], 0, 255).astype(np.uint8)
    reference = Image.fromarray(bg)
    draw = ImageDraw.Draw(reference)
    for line in range(size // 8, size, size // 4):
        draw.line((0, line, size, line), fill=tuple((background * 0.93).astype(int)))
    parts = []
    for row in range(2):
        for col in range(3):
            w, h = rng.integers(size // 9, size // 5, 2)
            px = int(size * (0.13 + col * 0.27) + rng.integers(-3, 4))
            py = int(size * (0.2 + row * 0.43) + rng.integers(-3, 4))
            parts.append(((px, py, int(w), int(h)), tuple(rng.integers(25, 230, 3).tolist()), int(rng.integers(3))))
    for part in parts:
        _draw_part(reference, *part)
    latent = reference.copy()
    mask_image = Image.new("L", (size, size))
    change = scenario in ("moved", "removed", "added") or (scenario == "mixed" and rng.random() < 0.75)
    kind = scenario if scenario != "mixed" else str(rng.choice(["moved", "removed", "added"]))
    if change:
        index = int(rng.integers(len(parts)))
        box, color, shape = parts[index]
        px, py, w, h = box
        if kind in ("moved", "removed"):
            latent.paste(Image.fromarray(bg[py:py + h + 1, px:px + w + 1]), (px, py))
            # Re-render the background grid consistently inside erased objects.
            erase_draw = ImageDraw.Draw(latent)
            for line in range(size // 8, size, size // 4):
                if py <= line <= py + h:
                    erase_draw.line((px, line, px + w, line), fill=tuple((background * 0.93).astype(int)))
            _draw_part(mask_image, box, 255, shape)
        if kind in ("moved", "added"):
            nx = int(rng.integers(size // 10, size - w - 4))
            ny = int(rng.integers(size // 10, size - h - 4))
            new_box = (nx, ny, w, h)
            _draw_part(latent, new_box, color, shape)
            _draw_part(mask_image, new_box, 255, shape)
    else:
        kind = "unchanged"
    gain, bias, shadow, dx, dy, blur, noise = 1., 0., 0., 0, 0, 0., 0.
    if scenario in ("mixed", "lighting", "moved", "removed", "added"):
        gain = float(rng.uniform(0.68, 1.28))
        bias = float(rng.uniform(-0.06, 0.06))
        shadow = float(rng.uniform(0., 0.5))
    if scenario in ("mixed", "camera", "moved", "removed", "added"):
        dx, dy = rng.integers(-3, 4, 2).tolist()
    if scenario in ("mixed", "blur"):
        blur = float(rng.uniform(0.2, 1.5))
        noise = float(rng.uniform(0, 0.025))
    sensor = Sensor(gain=1 + (gain - 1) * severity, bias=bias * severity,
                    shadow=min(0.85, shadow * severity), shadow_x=float(rng.uniform(0.25, 0.75)),
                    dx=int(round(dx * severity)), dy=int(round(dy * severity)),
                    blur=blur * severity, noise=noise * severity)
    return Scene(np.asarray(reference).astype(np.float32) / 255,
                 np.asarray(latent).astype(np.float32) / 255,
                 (np.asarray(mask_image) > 0).astype(np.float32), sensor, seed, kind)
