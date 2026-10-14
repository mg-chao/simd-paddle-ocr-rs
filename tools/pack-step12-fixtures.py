"""Select immutable goldens for compact CI or larger local-only REC/OCR checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

PIN = "33ff8a81672452041d110e8ba52dcd2639e54228"
MAX_ASSET_BYTES = 2 * 1024 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            value.update(block)
    return value.hexdigest()


def source_asset(asset, content, detector_name):
    variant = "small" if detector_name == "small_det.onnx" else "medium" if detector_name == "medium_det.onnx" else "tiny"
    names = {"det": detector_name, "cls": "cls.onnx",
             "rec": "rec.onnx" if variant == "tiny" else variant + "_rec.onnx",
             "dictionary": "ppocr_keys.txt" if variant == "tiny" else "rec_keys.txt"}
    if asset["id"] not in names:
        raise ValueError("Unknown source asset id: " + asset["id"])
    record = {"path": asset["path"], "bytes": asset["bytes"], "sha256": asset["sha256"], "source_name": names[asset["id"]]}
    if asset["id"] == "dictionary" and b"\r\n" in content:
        raw = content.replace(b"\r\n", b"\n")
        if b"\r" in raw or raw.replace(b"\n", b"\r\n") != content:
            raise ValueError("Dictionary cannot be represented by pinned LF transport bytes")
        record["download"] = {"bytes": len(raw), "sha256": digest(raw), "transform": "lf_to_crlf"}
    return record


def package(source, destination, select, model_name, model_ids=None):
    original_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(original_bytes)
    manifest["cases"] = select(manifest["cases"])
    if model_ids is not None:
        manifest["models"] = [asset for asset in manifest["models"] if asset["id"] in model_ids]
    manifest["fixture_selection"] = {"source_manifest_sha256": digest(original_bytes),
                                     "purpose": "immutable canonical regression oracle; selected cases retain every declared stage; model payloads acquired by pinned hash"}
    assets = manifest["models"]
    for asset in assets:
        content = (source / asset["path"]).read_bytes()
        if len(content) != asset["bytes"] or digest(content) != asset["sha256"]:
            raise ValueError("source model integrity differs")
    payloads = {a["path"]: a for case in manifest["cases"] for a in case["artifacts"]}
    payloads.update({a["path"]: a for a in manifest.get("auxiliary_files", [])})
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, descriptor in sorted(payloads.items()):
            path = source / relative
            if path.stat().st_size != descriptor["bytes"] or file_digest(path) != descriptor["sha256"]:
                raise ValueError("source artifact integrity differs: " + relative)
            entry = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as input_file, archive.open(entry, "w", force_zip64=True) as output:
                shutil.copyfileobj(input_file, output, 65536)
        entry = zipfile.ZipInfo("manifest.json", (1980, 1, 1, 0, 0, 0))
        archive.writestr(entry, json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n", compress_type=zipfile.ZIP_DEFLATED)
    if destination.stat().st_size >= MAX_ASSET_BYTES:
        raise ValueError("Fixture asset must be below 2 GiB; package explicit case shards")
    return {"archive": destination.name, "bytes": destination.stat().st_size, "sha256": file_digest(destination),
            "models": [source_asset(a, (source / a["path"]).read_bytes(), model_name) for a in assets]}


def package_imports(source, destination):
    original_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(original_bytes)
    if manifest["kind"] != "model_import_contracts":
        raise ValueError("Expected the original complete import archive")
    manifest["kind"] = "model_import_hash_contracts"
    manifest["fixture_selection"] = {"source_manifest_sha256": digest(original_bytes),
                                     "purpose": "independent C# semantic graphs and normalized-weight SHA256 contracts; weight bytes are not archived here"}
    manifest["auxiliary_files"] = [item for item in manifest["auxiliary_files"] if not item["path"].startswith("weights/")]
    payloads = {}
    for contract in manifest["contracts"]:
        for stage in ("raw", "normalized"):
            descriptor = contract[stage]
            data = (source / descriptor["path"]).read_bytes()
            if len(data) != descriptor["bytes"] or digest(data) != descriptor["sha256"]:
                raise ValueError("Original C# contract integrity differs")
            if stage == "normalized":
                snapshot = json.loads(data)
                for tensor in snapshot["tensors"]:
                    tensor["data_path"] = None
                data = json.dumps(snapshot, indent=2, sort_keys=True).encode() + b"\n"
                descriptor["bytes"], descriptor["sha256"] = len(data), digest(data)
            payloads[descriptor["path"]] = data
    for descriptor in manifest["auxiliary_files"]:
        data = (source / descriptor["path"]).read_bytes()
        if len(data) != descriptor["bytes"] or digest(data) != descriptor["sha256"]:
            raise ValueError("Original import provenance integrity differs")
        payloads[descriptor["path"]] = data
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, data in sorted(payloads.items()):
            archive.writestr(zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0)), data,
                             compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        archive.writestr(zipfile.ZipInfo("manifest.json", (1980, 1, 1, 0, 0, 0)),
                         json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n",
                         compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    assets = []
    for item in manifest["models"] + manifest["dictionaries"]:
        content = (source / item["path"]).read_bytes()
        if len(content) != item["bytes"] or digest(content) != item["sha256"]:
            raise ValueError("Original import source identity differs")
        asset = {"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"],
                 "source_name": Path(item["path"]).name}
        if item in manifest["dictionaries"] and b"\r\n" in content:
            raw = content.replace(b"\r\n", b"\n")
            if b"\r" in raw or raw.replace(b"\n", b"\r\n") != content:
                raise ValueError("Dictionary line endings cannot be represented by the pinned LF download")
            asset["download"] = {"bytes": len(raw), "sha256": digest(raw), "transform": "lf_to_crlf"}
        assets.append(asset)
    data = destination.read_bytes()
    return {"archive": destination.name, "directory": "imports", "bytes": len(data), "sha256": digest(data),
            "models": assets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imports-only", action="store_true", help="Append the immutable semantic importer fixture to an existing index")
    parser.add_argument("--ocr-source", type=Path, help="Append immutable tiny REC and complete OCR goldens from this fresh semantic reference")
    parser.add_argument("--bundle-source", type=Path, help="Package a complete immutable OCR bundle, with all cases unless --case selects shards")
    parser.add_argument("--fixture-name", help="Versioned archive/directory identity for --bundle-source")
    parser.add_argument("--variant", choices=("tiny", "small", "medium"), default="tiny")
    parser.add_argument("--case", action="append", help="Case id to retain (repeat to create a full-reference shard)")
    parser.add_argument("--url", help="Pinned versioned release-asset URL for the new archive")
    parser.add_argument("--index", type=Path, help="Append the new immutable entry to this existing schema-1 index")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.bundle_source:
        name = args.fixture_name
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in name):
            parser.error("--bundle-source requires a lowercase versioned --fixture-name")
        if args.ocr_source or args.imports_only:
            parser.error("--bundle-source cannot be combined with legacy packaging modes")
        index = json.loads(args.index.read_bytes()) if args.index else {
            "schema_version": 1, "model_commit": PIN,
            "model_url_root": f"https://raw.githubusercontent.com/mg-chao/SimdPaddleOCR/{PIN}/models/", "fixtures": []}
        if index["schema_version"] != 1 or any(entry["directory"] == name for entry in index["fixtures"]):
            raise ValueError("Fixture identity already exists or index schema differs")
        source_manifest = json.loads((args.bundle_source / "manifest.json").read_bytes())
        if source_manifest["contract"]["model"] != args.variant:
            raise ValueError("OCR source model variant differs")
        def selected_cases(cases):
            available = {case["id"] for case in cases}
            if args.case and not set(args.case).issubset(available):
                raise ValueError("Unknown source case selection")
            result = [case for case in cases if not args.case or case["id"] in args.case]
            if not result:
                raise ValueError("Empty fixture selection")
            return result
        record = package(args.bundle_source, args.out / (name + ".zip"), selected_cases,
                         "det.onnx" if args.variant == "tiny" else args.variant + "_det.onnx")
        record["directory"] = name
        if args.url:
            record["url"] = args.url
        index["fixtures"].append(record)
        index_path = args.index if args.index else args.out / "index.json"
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))
        return
    if args.ocr_source:
        index_path = args.out / "index.json"
        index = json.loads(index_path.read_bytes())
        if any(entry["directory"] in ("tiny-rec", "tiny-ocr") for entry in index["fixtures"]):
            raise ValueError("Tiny OCR fixture identity already exists")
        def recognition_cases(cases):
            selected = []
            widths = set()
            for case in cases:
                grouped = {}
                for artifact in case["artifacts"]:
                    if artifact["id"].startswith("line.") and ".rec." in artifact["id"]:
                        prefix = artifact["id"].split(".rec.")[0]
                        grouped.setdefault(prefix, []).append(artifact)
                chosen = []
                for prefix, artifacts in grouped.items():
                    input_artifact = next(a for a in artifacts if a["id"].endswith(".rec.input"))
                    width = input_artifact["shape"][3]
                    if width not in widths and len(widths) < 4:
                        widths.add(width)
                        chosen.extend(artifacts)
                if chosen:
                    selected.append({**case, "artifacts": chosen})
            if len(widths) < 2:
                raise ValueError("Recognition oracle must contain at least two distinct widths")
            return selected
        # The rotated page remains in the expanded oracle. Keep two real lines,
        # blank/tiny cases and complete per-node coverage in the local gate.
        def ocr_cases(cases):
            return [case for case in cases if case["id"] != "image-001"]
        for directory, select, ids in (("tiny-rec", recognition_cases, {"rec", "dictionary"}), ("tiny-ocr", ocr_cases, None)):
            record = package(args.ocr_source, args.out / (directory + ".zip"), select, "det.onnx", ids)
            record["directory"] = directory
            record["local_only"] = True
            index["fixtures"].append(record)
            print(f"Packed {directory}: {record['bytes']:,} bytes")
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return
    if args.imports_only:
        if args.references is None:
            parser.error("--imports-only requires --references")
        index_path = args.out / "index.json"
        index = json.loads(index_path.read_bytes())
        if any(entry["directory"] == "imports" for entry in index["fixtures"]):
            raise ValueError("Importer fixture identity already exists")
        record = package_imports(args.references / "import-reference-v1", args.out / "imports.zip")
        index["fixtures"].append(record)
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        print(f"Packed importer semantic/hash contracts: {record['bytes']:,} bytes")
        return
    entries = []
    if args.references is None:
        parser.error("Legacy packaging requires --references")
    for variant in ("tiny", "small", "medium"):
        source = args.references / f"{variant}-step12-compact-v1"
        record = package(source, args.out / f"{variant}.zip",
                         lambda cases: [c for c in cases if c["id"] in ("landscape", "portrait")],
                         "det.onnx" if variant == "tiny" else f"{variant}_det.onnx")
        record["directory"] = variant
        entries.append(record)
    source = args.references / "tiny-canonical-allnodes-v1"
    # The classifier bundle contains two independent existing frozen crops and all their nodes.
    def cls_cases(cases):
        selected = []
        for case in cases:
            if case["id"] in ("sample", "sample-180"):
                selected.append({**case, "artifacts": [a for a in case["artifacts"] if a["id"].startswith("line.000.cls.")]})
        return selected
    record = package(source, args.out / "cls.zip", cls_cases, "cls.onnx", {"cls"})
    record["directory"] = "cls"
    entries.append(record)
    entries.append(package_imports(args.references / "import-reference-v1", args.out / "imports.zip"))
    index = {"schema_version": 1, "model_commit": PIN,
             "model_url_root": f"https://raw.githubusercontent.com/mg-chao/SimdPaddleOCR/{PIN}/models/", "fixtures": entries}
    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"Packed {len(entries)} references, {sum(e['bytes'] for e in entries):,} compressed bytes")


if __name__ == "__main__":
    main()
