"""Restore the original ARM wheels from verified XZ archive parts."""

import argparse
import hashlib
import json
import lzma
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def restore(archive_root, output, abi):
    manifest = json.loads((archive_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "xz-split-v1":
        raise ValueError("Unsupported wheel archive format")
    selected = [entry for entry in manifest["wheels"] if abi == "all" or entry["abi"] == abi]
    if not selected:
        raise ValueError("No wheels match the selected Python ABI")
    output.mkdir(parents=True, exist_ok=True)
    for entry in selected:
        filename = entry["file"]
        if Path(filename).name != filename:
            raise ValueError("Invalid wheel filename")
        destination = output / filename
        if destination.exists():
            if sha256(destination.read_bytes()) != entry["wheel_sha256"]:
                raise ValueError("Existing wheel has a different SHA256: " + filename)
            print("Verified existing:", filename)
            continue
        parts = []
        for part in entry["parts"]:
            path = (archive_root / part["file"]).resolve()
            if archive_root.resolve() not in path.parents:
                raise ValueError("Invalid archive part path")
            data = path.read_bytes()
            if len(data) != part["bytes"] or sha256(data) != part["sha256"]:
                raise ValueError("Archive part SHA256 mismatch: " + part["file"])
            parts.append(data)
        packed = b"".join(parts)
        if len(packed) != entry["xz_bytes"] or sha256(packed) != entry["xz_sha256"]:
            raise ValueError("Compressed archive SHA256 mismatch: " + filename)
        wheel = lzma.decompress(packed)
        if len(wheel) != entry["wheel_bytes"] or sha256(wheel) != entry["wheel_sha256"]:
            raise ValueError("Restored wheel SHA256 mismatch: " + filename)
        destination.write_bytes(wheel)
        print("Restored and verified:", filename)


def main():
    archive_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abi", choices=("all", "cp311", "cp312"), default="all")
    parser.add_argument("--output", type=Path, default=archive_root.parent / "wheels")
    args = parser.parse_args()
    restore(archive_root, args.output, args.abi)


if __name__ == "__main__":
    main()
