from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .interventions import build_lattice
from .simulator import SPLIT_START, make_scene


def tensors(reference, observation, mask, valid):
    vertices, _ = build_lattice(reference, observation)
    return (torch.from_numpy(reference.transpose(2, 0, 1).copy()),
            torch.from_numpy(vertices.transpose(0, 3, 1, 2).copy()),
            torch.from_numpy(mask[None].copy()), torch.from_numpy(valid[None].copy()))


class SimulationDataset(Dataset):
    def __init__(self, count=512, size=96, split="train", scenario="mixed", severity=1.):
        self.count, self.size, self.start = count, size, SPLIT_START[split]
        self.scenario, self.severity = scenario, severity

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        scene = make_scene(self.start + index, self.size, self.scenario, self.severity)
        observation, valid = scene.observe()
        return tensors(scene.reference, observation, scene.mask, valid)


def pcd_root(root):
    root = Path(root)
    candidates = [root] + list(root.glob("*"))
    for directory in candidates:
        if (directory / "t0").is_dir() and (directory / "mask").is_dir():
            return directory
    raise FileNotFoundError("PCD TSUNAMI missing. Run python scripts/fetch_pcd.py")


def pcd_ids(root, split):
    # Image-pair split BEFORE extracting crops: prevents panorama leakage.
    ids = sorted(p.stem for p in (pcd_root(root) / "t0").glob("*.jpg"))
    order = np.random.default_rng(731).permutation(len(ids))
    ranges = {"train": (0, 60), "validation": (60, 80), "test": (80, 100)}
    lo, hi = ranges[split]
    return [ids[i] for i in order[lo:hi]]


class PCDDataset(Dataset):
    def __init__(self, root, split="train", size=96):
        self.root, self.size = pcd_root(root), size
        self.ids = pcd_ids(root, split)
        self.split = split
        self.manifest = [{"id": id_, "crop": j} for id_ in self.ids for j in range(4)]

    def __len__(self):
        return len(self.manifest)

    def arrays(self, index):
        item = self.manifest[index]
        id_, crop = item["id"], item["crop"]
        images = [Image.open(self.root / folder / f"{id_}{suffix}")
                  for folder, suffix in [("t0", ".jpg"), ("t1", ".jpg"), ("mask", ".png")]]
        w, h = images[0].size
        bounds = (crop * w // 4, 0, (crop + 1) * w // 4, h)
        a, b = [np.asarray(img.convert("RGB").crop(bounds).resize((self.size, self.size),
                         Image.Resampling.BILINEAR)).astype(np.float32) / 255 for img in images[:2]]
        raw = np.asarray(images[2].convert("L").crop(bounds).resize((self.size, self.size),
                          Image.Resampling.NEAREST))
        # White means change. Some author-supplied masks have gray boundary pixels;
        # consistently binarize at 128 before all training and evaluation.
        mask = (raw >= 128).astype(np.float32)
        valid = np.ones(mask.shape, np.float32)
        valid[:5] = valid[-5:] = 0
        valid[:, :5] = valid[:, -5:] = 0
        return a, b, mask, valid

    def __getitem__(self, index):
        return tensors(*self.arrays(index))
