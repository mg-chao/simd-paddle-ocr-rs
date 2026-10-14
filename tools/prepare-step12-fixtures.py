"""Restore immutable local/hosted goldens and models, checking every acquired identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.request
import zipfile


def checked_file(path, expected):
    validate_identity(expected)
    if not path.is_file() or path.is_symlink() or path.stat().st_size != expected["bytes"]:
        return False
    hash_value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            hash_value.update(block)
    return hash_value.hexdigest() == expected["sha256"]


def validate_identity(expected):
    if (type(expected["bytes"]) is not int or expected["bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])):
        raise ValueError("Invalid fixture byte-count/SHA-256 identity")


def relative_file(value):
    """Use the same unambiguous relative file names on Windows and Linux."""
    path = PurePosixPath(value)
    reserved = {"con", "prn", "aux", "nul"} | {f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)}
    if (not value or path.is_absolute() or any(character in value for character in '\\:<>"|?*')
            or any(ord(character) < 32 for character in value)
            or any(part in ("", ".", "..") or part.rstrip(" .") != part
                   or part.split(".")[0].casefold() in reserved for part in value.split("/"))):
        raise ValueError("Unsafe fixture path: " + value)
    return path


def file_name(value):
    path = relative_file(value)
    if len(path.parts) != 1:
        raise ValueError("Expected a fixture file/directory name: " + value)
    return value


def download(url, destination):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SimdPaddleOCR-fixtures/1"})
            with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
                shutil.copyfileobj(response, output, 65536)
            return
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def supplied_model(model, args):
    if args.models is None:
        return None
    local = args.models / file_name(model["source_name"])
    transport = model.get("download")
    if not checked_file(local, model) and not (transport is not None and checked_file(local, transport)):
        raise ValueError(f"Local model is absent or differs from the pinned identity: {local}")
    return local


def obtain(model, args, url_root):
    validate_identity(model)
    source_name = file_name(model["source_name"])
    local = supplied_model(model, args)
    cached = args.cache / model["sha256"] / source_name
    if checked_file(cached, model):
        return cached
    cached.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as output:
        temporary = Path(output.name)
    try:
        if local is not None:
            shutil.copyfile(local, temporary)
        elif getattr(args, "offline", False):
            raise ValueError("Offline model cache is absent or differs: " + str(cached))
        else:
            download(url_root + source_name, temporary)
        transport = model.get("download")
        if not checked_file(temporary, model) and transport is not None:
            if not checked_file(temporary, transport):
                raise ValueError("Transport asset hash/size differs: " + source_name)
            if transport["transform"] != "lf_to_crlf":
                raise ValueError("Unknown pinned asset transformation")
            raw = temporary.read_bytes()
            if b"\r" in raw:
                raise ValueError("Pinned LF dictionary contains carriage returns")
            temporary.write_bytes(raw.replace(b"\n", b"\r\n"))
        if not checked_file(temporary, model):
            raise ValueError("Acquired model hash/size differs: " + source_name)
        os.replace(temporary, cached)
    finally:
        temporary.unlink(missing_ok=True)
    return cached


def obtain_archive(fixture, args):
    validate_identity(fixture)
    archive_name = file_name(fixture["archive"])
    supplied = args.goldens / archive_name
    # Supplied inputs take precedence, including errors; a good cache must not hide corruption.
    if supplied.exists():
        if not checked_file(supplied, fixture):
            raise ValueError("Golden archive hash/size differs: " + str(supplied))
        return supplied
    cached = args.cache / "archives" / fixture["sha256"] / archive_name
    if checked_file(cached, fixture):
        return cached
    if fixture.get("local_only", False) or not fixture.get("url"):
        raise ValueError("Golden archive hash/size differs: " + str(supplied))
    if args.offline:
        raise ValueError("Offline golden archive cache is absent or differs: " + str(cached))
    cached.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as output:
        temporary = Path(output.name)
    try:
        download(fixture["url"], temporary)
        if not checked_file(temporary, fixture):
            raise ValueError("Acquired golden archive hash/size differs: " + archive_name)
        os.replace(temporary, cached)
    finally:
        temporary.unlink(missing_ok=True)
    return cached


def descriptors(manifest, models):
    declared = list(manifest.get("models", [])) + list(manifest.get("dictionaries", []))
    declared += manifest.get("auxiliary_files", [])
    declared += [artifact for case in manifest.get("cases", []) for artifact in case["artifacts"]]
    declared += [contract[stage] for contract in manifest.get("contracts", []) for stage in ("raw", "normalized")]
    declared += models
    result = {}
    for item in declared:
        path = str(relative_file(item["path"]))
        if path in result and any(result[path][field] != item[field] for field in ("bytes", "sha256")):
            raise ValueError("Conflicting fixture descriptor: " + path)
        result[path] = item
    return result


def verify_prepared(root, manifest_bytes, payloads):
    if not root.is_dir() or root.is_symlink():
        return False
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink() or manifest_path.read_bytes() != manifest_bytes:
        return False
    expected = set(payloads) | {"manifest.json"}
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            return False
        if path.is_file():
            observed.add(path.relative_to(root).as_posix())
    return observed == expected and all(checked_file(root / path, item) for path, item in payloads.items())


def restore(fixture, args, url_root):
    archive = obtain_archive(fixture, args)
    out = args.out.resolve()
    directory = file_name(fixture["directory"])
    destination = out / directory
    if destination.is_symlink() or not destination.resolve().is_relative_to(out):
        raise ValueError("Unsafe fixture destination: " + str(destination))
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        seen = set()
        for entry in entries:
            relative = relative_file(entry.filename)
            key = str(relative).casefold()
            if key in seen or entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Duplicate or unsafe golden archive entry: " + entry.filename)
            seen.add(key)
        manifest_bytes = zipped.read("manifest.json")
        manifest = json.loads(manifest_bytes)
        payloads = descriptors(manifest, fixture["models"])
        archive_paths = {entry.filename for entry in entries}
        if not archive_paths.issubset(set(payloads) | {"manifest.json"}):
            raise ValueError("Golden archive contains undeclared payloads")
        # A reused destination must not conceal invalid declared archive contents.
        for entry in entries:
            if entry.filename == "manifest.json":
                continue
            expected = payloads[entry.filename]
            value = hashlib.sha256()
            with zipped.open(entry) as source:
                for block in iter(lambda: source.read(65536), b""):
                    value.update(block)
            if entry.file_size != expected["bytes"] or value.hexdigest() != expected["sha256"]:
                raise ValueError("Golden archive payload hash/size differs: " + entry.filename)
        # Explicit supplied model inputs must not be hidden by valid output/cache.
        for model in fixture["models"]:
            supplied_model(model, args)
        if verify_prepared(destination, manifest_bytes, payloads):
            print(f"Reused verified golden: {destination} (archive SHA-256 {fixture['sha256']})")
            return
        # Obtain all external inputs before touching an earlier prepared bundle.
        models = [(model, obtain(model, args, url_root)) for model in fixture["models"]]
        out.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="." + directory + "-", dir=out))
        backup = None
        try:
            for entry in entries:
                path = staging.joinpath(*relative_file(entry.filename).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(entry) as source, path.open("wb") as output:
                    shutil.copyfileobj(source, output, 65536)
            for model, source in models:
                path = staging.joinpath(*relative_file(model["path"]).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                if not checked_file(path, model):
                    shutil.copyfile(source, path)
            if not verify_prepared(staging, manifest_bytes, payloads):
                raise ValueError("Restored golden payload hash/size differs: " + directory)
            if destination.exists():
                backup = Path(tempfile.mkdtemp(prefix="." + directory + "-old-", dir=out))
                backup.rmdir()
                destination.rename(backup)
            try:
                staging.rename(destination)
            except BaseException:
                if backup is not None:
                    backup.rename(destination)
                    backup = None
                raise
        finally:
            # Both paths were created under the resolved output root in this operation.
            for temporary in (staging, backup):
                if temporary is not None and temporary.exists():
                    if temporary.parent.resolve() != out or temporary.is_symlink():
                        raise ValueError("Unsafe fixture staging cleanup")
                    if temporary.is_dir():
                        shutil.rmtree(temporary)
                    else:
                        temporary.unlink()
    print(f"Prepared immutable golden: {destination} (archive SHA-256 {fixture['sha256']})")


def main():
    parser = argparse.ArgumentParser()
    repository = Path(__file__).resolve().parent.parent
    parser.add_argument("--goldens", type=Path, default=repository / "tests/fixtures/step12")
    parser.add_argument("--out", type=Path, default=repository / "target/step12-fixtures")
    parser.add_argument("--cache", type=Path, default=repository / "target/reference-models")
    parser.add_argument("--models", type=Path, help="Offline model/dictionary directory; canonical or pinned LF transport bytes")
    parser.add_argument("--offline", action="store_true", help="Use only supplied files or verified cache entries; never use the network")
    parser.add_argument("--fixture", action="append", help="Restore this fixture directory identity (repeat for multiple selections)")
    parser.add_argument("--include-local", action="store_true", help="Also restore local-only archives, which must be supplied in --goldens")
    args = parser.parse_args()
    index = json.loads((args.goldens / "index.json").read_bytes())
    if index["schema_version"] != 1:
        raise ValueError("Unknown golden index schema")
    fixtures = index["fixtures"]
    names = [file_name(fixture["directory"]) for fixture in fixtures]
    if len(set(name.casefold() for name in names)) != len(names):
        raise ValueError("Duplicate fixture directory identity")
    if args.fixture and not set(args.fixture).issubset(names):
        raise ValueError("Unknown fixture selection: " + ", ".join(sorted(set(args.fixture) - set(names))))
    for fixture in fixtures:
        if args.fixture and fixture["directory"] not in args.fixture:
            continue
        if fixture.get("local_only", False) and not (args.include_local or args.fixture):
            print(f"Skipped local-only golden: {fixture['archive']} (use --include-local to restore)")
            continue
        restore(fixture, args, index["model_url_root"])


if __name__ == "__main__":
    main()
