import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="tracecv", description="TRACE visual change verification")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Start local API and built studio")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--model", default="artifacts/trace")
    infer = sub.add_parser("inspect", help="Inspect an equally sized image pair")
    infer.add_argument("reference", type=Path)
    infer.add_argument("observation", type=Path)
    infer.add_argument("--model", default="artifacts/trace")
    infer.add_argument("--output", type=Path, default=Path("evidence.json"))
    batch = sub.add_parser("batch", help="Inspect CSV pairs with id,reference,observation columns")
    batch.add_argument("manifest", type=Path)
    batch.add_argument("--model", default="artifacts/trace")
    batch.add_argument("--output", type=Path, default=Path("artifacts/batch"))
    args = parser.parse_args()
    if args.command in {"inspect", "batch"}:
        import torch
        torch.set_num_threads(4)
    if args.command == "serve":
        import os

        import uvicorn
        os.environ["TRACE_MODEL"] = args.model
        uvicorn.run("tracecv.api:app", host="127.0.0.1", port=args.port)
    elif args.command == "inspect":
        from .inference import Detector, read_image
        a, shape_a = read_image(args.reference.read_bytes())
        b, shape_b = read_image(args.observation.read_bytes())
        if shape_a != shape_b:
            parser.error("Image dimensions must match")
        model = Detector(args.model)
        prediction = model.predict(a, b)
        result = model.evidence(a, b, prediction)
        result["original_dimensions"] = shape_a
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np
        raw = args.output.with_name(args.output.stem + "-maps.npz")
        np.savez_compressed(raw, probability=prediction["probability"], error=prediction["error"],
                            mask=prediction["mask"], branches=prediction["branches"], weights=prediction["weights"])
        result["float_maps_file"] = raw.name
        args.output.write_text(json.dumps(result, indent=2))
        print(f"Saved evidence to {args.output}; changed area {result['changed_fraction']:.2%}")
    else:
        import re

        from .inference import Detector, read_image
        model = Detector(args.model)
        args.output.mkdir(parents=True, exist_ok=True)
        rows, used = [], set()
        with args.manifest.open(newline="") as file:
            reader = csv.DictReader(file)
            if not {"id", "reference", "observation"}.issubset(reader.fieldnames or []):
                parser.error("CSV requires id,reference,observation columns")
            for item in reader:
                identifier = item["id"]
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identifier) or identifier in used:
                    parser.error("Pair IDs must be unique and contain only letters, numbers, hyphens or underscores")
                used.add(identifier)
                a, sa = read_image((args.manifest.parent / item["reference"]).read_bytes())
                b, sb = read_image((args.manifest.parent / item["observation"]).read_bytes())
                if sa != sb:
                    parser.error(f"Dimension mismatch for pair {identifier}")
                result = model.evidence(a, b)
                result["original_dimensions"] = sa
                (args.output / f"{identifier}.json").write_text(json.dumps(result, indent=2))
                rows.append({"id": identifier, "changed_fraction": result["changed_fraction"],
                             "mean_error_indicator": result["mean_error_indicator"],
                             "elapsed_ms": result["elapsed_ms"]})
        with (args.output / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["id", "changed_fraction", "mean_error_indicator", "elapsed_ms"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Inspected {len(rows)} pairs; results in {args.output}")


if __name__ == "__main__":
    main()
