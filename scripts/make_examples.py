"""Export original procedural example pairs, separate from all experimental splits."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tracecv.simulator import make_scene

root = Path(__file__).resolve().parents[1] / "examples"
root.mkdir(exist_ok=True)
rows, labels = [], []
for i, scenario in enumerate(["moved", "lighting", "removed"]):
    seed = 8_000_000 + i
    scene = make_scene(seed, scenario=scenario)
    observation, _ = scene.observe()
    for role, array in [("reference", scene.reference), ("observation", observation), ("label", scene.mask)]:
        Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8)).save(root / f"{scenario}-{role}.png")
    rows.append({"id": scenario, "reference": f"{scenario}-reference.png", "observation": f"{scenario}-observation.png"})
    labels.append({"id": scenario, "seed": seed, "label": f"{scenario}-label.png"})
with (root / "pairs.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["id", "reference", "observation"])
    writer.writeheader()
    writer.writerows(rows)
(root / "provenance.json").write_text(json.dumps({"source": "Original procedural renderer", "examples": labels,
    "usage": "Demonstration only; these seeds are outside training, validation and test experiments."}, indent=2))
print("Wrote three example pairs and labels to", root)
