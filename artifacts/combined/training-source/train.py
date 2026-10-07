"""Reproducible detector training; selection and calibration only use validation scenes."""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from .data import PCDDataset, SimulationDataset, pcd_ids
from .metrics import calibrate, score
from .model import TRACE, segmentation_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def materialize(dataset):
    first = dataset[0]
    tensors = [value.new_empty((len(dataset), *value.shape)) for value in first]
    for index in range(len(dataset)):
        row = first if index == 0 else dataset[index]
        for tensor, value in zip(tensors, row, strict=True):
            tensor[index].copy_(value)
    return TensorDataset(*tensors)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def select_vertices(vertices, variant):
    return vertices[:, 3:4] if variant == "combined" else vertices


@torch.inference_mode()
def collect(model, loader, device, variant):
    model.eval()
    logits, targets, errors = [], [], []
    for a, v, y, valid in loader:
        output = model(a.to(device), select_vertices(v.to(device), variant))
        keep = valid.numpy().astype(bool)
        logits.append(output["logits"].cpu().numpy()[keep])
        targets.append(y.numpy()[keep])
        errors.append(output["error_logits"].cpu().numpy()[keep])
    return np.concatenate(logits), np.concatenate(targets), np.concatenate(errors)


def source_digest():
    root = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}


def run(args):
    seed_everything(args.seed)
    device = device_for(args.device)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    save_json(destination / "config.json", config)
    print("Preparing independent train/validation scenes on", device, flush=True)
    train_parts = [SimulationDataset(args.sim_train, args.size)]
    validation_parts = [SimulationDataset(args.sim_validation, args.size, "validation")]
    if args.real_data:
        train_parts.append(PCDDataset(args.real_data, "train", args.size))
        validation_parts.append(PCDDataset(args.real_data, "validation", args.size))
        splits = {split: pcd_ids(args.real_data, split) for split in ["train", "validation", "test"]}
        assert not (set(splits["train"]) & set(splits["validation"]) or
                    set(splits["test"]) & set(splits["train"]) or set(splits["test"]) & set(splits["validation"]))
        save_json(destination / "pcd-splits.json", splits)
    train = materialize(ConcatDataset(train_parts))
    validation = materialize(ConcatDataset(validation_parts))
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(validation, batch_size=args.batch)
    model = TRACE(args.width, lattice=args.variant == "trace").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history, best, start = [], -1., time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for a, vertices, y, valid in train_loader:
            a, vertices, y, valid = [x.to(device) for x in (a, vertices, y, valid)]
            optimizer.zero_grad(set_to_none=True)
            output = model(a, select_vertices(vertices, args.variant))
            loss = segmentation_loss(output, y, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        logits, targets, _ = collect(model, validation_loader, device, args.variant)
        metrics = score(1 / (1 + np.exp(-np.clip(logits, -30, 30))), targets)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation": metrics,
               "elapsed_seconds": round(time.monotonic() - start, 1)}
        history.append(row)
        save_json(destination / "history.json", history)
        print(json.dumps(row), flush=True)
        if metrics["f1"] > best:
            best = metrics["f1"]
            temporary = destination / "model.pt.tmp"
            torch.save({"state_dict": model.cpu().state_dict(), "width": args.width,
                        "variant": args.variant, "epoch": epoch + 1, "seed": args.seed}, temporary)
            temporary.replace(destination / "model.pt")
            model.to(device)
    checkpoint = torch.load(destination / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    logits, targets, errors = collect(model, validation_loader, device, args.variant)
    calibration = calibrate(logits, targets)
    # Calibrate the learned error indicator separately, at the selected decision threshold.
    p = 1 / (1 + np.exp(-np.clip(logits / calibration["temperature"], -30, 30)))
    actual_errors = ((p >= calibration["threshold"]) != (targets > .5)).astype(float)
    error_calibration = calibrate(errors, actual_errors)
    calibration["error_temperature"] = error_calibration["temperature"]
    calibration["error_validation"] = error_calibration["validation"]
    save_json(destination / "calibration.json", calibration)
    provenance = {"source_sha256": source_digest(), "python": platform.python_version(),
                  "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
                  "train_samples": len(train), "validation_samples": len(validation),
                  "test_used_for_selection": False, "elapsed_seconds": time.monotonic() - start,
                  "checkpoint_sha256": hashlib.sha256((destination / "model.pt").read_bytes()).hexdigest()}
    save_json(destination / "provenance.json", provenance)
    print("Saved trained model and validation calibration:", destination, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/trace")
    parser.add_argument("--variant", choices=["trace", "raw", "combined"], default="trace")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--sim-train", type=int, default=512)
    parser.add_argument("--sim-validation", type=int, default=96)
    parser.add_argument("--real-data", default="data/pcd")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
