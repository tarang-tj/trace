"""One command rebuilds all learned components and the frozen evaluation."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--skip-trained", action="store_true", help="Reuse completed local artifacts explicitly")
    parser.add_argument("--synthetic-only", action="store_true", help="Avoid research-licensed real data")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def run(*command):
        subprocess.run(command, cwd=root, check=True)
    if not args.synthetic_only:
        run(sys.executable, "scripts/fetch_pcd.py")
    real = ["--real-data", ""] if args.synthetic_only else []
    for variant in ["trace", "raw", "combined"]:
        if not (args.skip_trained and (root / f"artifacts/{variant}/provenance.json").exists()):
            run(sys.executable, "-u", "-m", "tracecv.train", "--variant", variant, "--epochs", str(args.epochs),
                "--output", f"artifacts/{variant}", *real)
    run(sys.executable, "-u", "-m", "tracecv.policy")
    run(sys.executable, "-u", "-m", "tracecv.evaluate", *real)
    run(sys.executable, "-u", "scripts/audit_policy.py")
    if not args.synthetic_only:
        run(sys.executable, "scripts/figures.py")


if __name__ == "__main__":
    main()
