"""Create deterministic small, line-rich PNG fixtures from hash-verified frozen pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib


def png(path, bgr, width, height):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rgb = bytearray()
    for y in range(height):
        rgb.append(0)
        for x in range(width):
            b, g, r = bgr[(y * width + x) * 3: (y * width + x + 1) * 3]
            rgb.extend((r, g, b))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(bytes(rgb), 9)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compact", action="store_true", help="Two real text lines for small all-node CI goldens")
    args = parser.parse_args()
    manifest = json.loads((args.reference / "manifest.json").read_bytes())
    case = next(c for c in manifest["cases"] if c["id"] == "sample")
    def artifact(name):
        descriptor = next(a for a in case["artifacts"] if a["id"] == name)
        raw = (args.reference / descriptor["path"]).read_bytes()
        if len(raw) != descriptor["bytes"] or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
            raise ValueError("Frozen pixel identity differs")
        return raw
    pixels = artifact("input.pixels")
    meta = json.loads(artifact("input.metadata"))
    x, y, width, height = (20, 82, 160, 58) if args.compact else (20, 30, 350, 140)
    if meta["width"] < x + width or meta["height"] < y + height:
        raise ValueError("Frozen sample is smaller than the defined crop")
    crop = b"".join(pixels[(y + row) * meta["stride"] + x * 3:(y + row) * meta["stride"] + (x + width) * 3] for row in range(height))
    args.out.mkdir(parents=True, exist_ok=False)
    png(args.out / "00-page.png", crop, width, height)
    rotated = b"".join(crop[i:i + 3] for i in range(len(crop) - 3, -1, -3))
    png(args.out / "01-page-180.png", rotated, width, height)
    png(args.out / "02-blank.png", b"\xff" * (32 * 32 * 3), 32, 32)
    png(args.out / "03-single.png", b"\x17\x65\xd3", 1, 1)


if __name__ == "__main__":
    main()
