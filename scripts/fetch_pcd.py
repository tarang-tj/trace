"""Download the authors' compact TSUNAMI research dataset; never commit raw data."""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://drive.usercontent.google.com/download?id=1C3682c4oD4nuq8vhfkh0-6HrLBs5XC3H&export=download"
EXPECTED_SHA256 = "3589bd3f33209bc197c727d1541ec1a8a55650f6ffadcda5c39f76b28968f911"


def main():
    destination = ROOT / "data" / "pcd"
    if (destination / "provenance.json").exists():
        dataset = destination / "TSUNAMI"
        if all(len(list((dataset / folder).glob(extension))) == 100
               for folder, extension in [("t0", "*.jpg"), ("t1", "*.jpg"), ("mask", "*.png")]):
            print("Dataset already downloaded;", destination)
            return
    request = urllib.request.Request(URL, headers={"User-Agent": "TRACE-Research/0.1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = response.read(50_000_001)
    if len(payload) > 50_000_000 or not payload.startswith(b"PK"):
        raise RuntimeError("Expected a ZIP archive under 50 MB from the official public download")
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("Author archive changed; inspect it before updating the pinned checksum")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or member.file_size > 10_000_000:
                raise RuntimeError("Unsafe archive member")
        if sum(m.file_size for m in archive.infolist()) > 100_000_000:
            raise RuntimeError("Archive expands beyond the expected dataset size")
        for member in archive.infolist():
            # The PNG labels are sufficient; omit redundant uncompressed BMP copies.
            if "/bmp/" not in member.filename:
                archive.extract(member, destination)
    (destination / "provenance.json").write_text(json.dumps({
        "source": "https://sakuradaken.net/pcd_dataset.html", "download": URL,
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
        "usage": "Research purposes; cite Sakurada and Okatani, BMVC 2015. Raw data excluded from git.",
        "note": "Historical compact benchmark. Authors recommend newer PSCD for contemporary evaluation."
    }, indent=2))
    print("Downloaded", len(payload), "bytes to", destination, flush=True)


if __name__ == "__main__":
    main()
