"""Restore committed or explicitly requested local goldens with hash-pinned models."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile


def checked_file(path, expected):
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        return False
    hash_value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            hash_value.update(block)
    return hash_value.hexdigest() == expected["sha256"]


def obtain(model, args, url_root):
    cached = args.cache / model["sha256"] / model["source_name"]
    if checked_file(cached, model):
        return cached
    cached.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as output:
        temporary = Path(output.name)
    try:
        if args.models is not None:
            local = args.models / model["source_name"]
            download = model.get("download")
            if not checked_file(local, model) and not (download is not None and checked_file(local, download)):
                raise ValueError(f"Local model is absent or differs from the pinned identity: {local}")
            shutil.copyfile(local, temporary)
        else:
            for attempt in range(3):
                try:
                    request = urllib.request.Request(url_root + model["source_name"], headers={"User-Agent": "SimdPaddleOCR-fixtures/1"})
                    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                        shutil.copyfileobj(response, output, 65536)
                    break
                except (OSError, urllib.error.URLError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        # A local checkout may contain the same pinned LF transport bytes as
        # GitHub. Preserve the source directory and canonicalize the staged copy.
        download = model.get("download")
        if not checked_file(temporary, model) and download is not None:
            if not checked_file(temporary, download):
                raise ValueError("Transport asset hash/size differs: " + model["source_name"])
            if download["transform"] != "lf_to_crlf":
                raise ValueError("Unknown pinned asset transformation")
            raw = temporary.read_bytes()
            if b"\r" in raw:
                raise ValueError("Pinned LF dictionary contains carriage returns")
            temporary.write_bytes(raw.replace(b"\n", b"\r\n"))
        if not checked_file(temporary, model):
            raise ValueError("Acquired model hash/size differs: " + model["source_name"])
        os.replace(temporary, cached)
    finally:
        temporary.unlink(missing_ok=True)
    return cached


def main():
    parser = argparse.ArgumentParser()
    repository = Path(__file__).resolve().parent.parent
    parser.add_argument("--goldens", type=Path, default=repository / "tests/fixtures/step12")
    parser.add_argument("--out", type=Path, default=repository / "target/step12-fixtures")
    parser.add_argument("--cache", type=Path, default=repository / "target/reference-models")
    parser.add_argument("--models", type=Path, help="Offline model/dictionary directory; canonical or pinned LF transport bytes")
    parser.add_argument("--include-local", action="store_true",
                        help="Also restore large local-only REC/OCR archives, which must be supplied in --goldens")
    args = parser.parse_args()
    index = json.loads((args.goldens / "index.json").read_bytes())
    if index["schema_version"] != 1:
        raise ValueError("Unknown golden index schema")
    for fixture in index["fixtures"]:
        if fixture.get("local_only", False) and not args.include_local:
            print(f"Skipped local-only golden: {fixture['archive']} (use --include-local to restore)")
            continue
        archive = args.goldens / fixture["archive"]
        if not checked_file(archive, fixture):
            raise ValueError("Golden archive hash/size differs: " + str(archive))
        destination = args.out.resolve() / fixture["directory"]
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                relative = PurePosixPath(entry.filename)
                if relative.is_absolute() or ".." in relative.parts or "\\" in entry.filename or entry.is_dir():
                    raise ValueError("Unsafe golden archive entry")
                path = destination.joinpath(*relative.parts)
                if not path.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Golden entry escapes destination")
                path.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(entry) as source, path.open("wb") as output:
                    shutil.copyfileobj(source, output, 65536)
        for model in fixture["models"]:
            source = obtain(model, args, index["model_url_root"])
            path = destination / model["path"]
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe model destination")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not checked_file(path, model):
                shutil.copyfile(source, path)
        print(f"Prepared immutable golden: {destination}")


if __name__ == "__main__":
    main()
