"""Select immutable golden outputs; CI consumes these archives without regenerating them."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

PIN = "33ff8a81672452041d110e8ba52dcd2639e54228"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package(source, destination, select, model_name, model_ids=None):
    original_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(original_bytes)
    manifest["cases"] = select(manifest["cases"])
    if model_ids is not None:
        manifest["models"] = [asset for asset in manifest["models"] if asset["id"] in model_ids]
    manifest["fixture_selection"] = {"source_manifest_sha256": digest(original_bytes),
                                     "purpose": "small canonical regression oracle; model payloads acquired by pinned hash"}
    assets = manifest["models"]
    for asset in assets:
        content = (source / asset["path"]).read_bytes()
        if len(content) != asset["bytes"] or digest(content) != asset["sha256"]:
            raise ValueError("source model integrity differs")
    payloads = {a["path"]: a for case in manifest["cases"] for a in case["artifacts"]}
    payloads.update({a["path"]: a for a in manifest.get("auxiliary_files", [])})
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, descriptor in sorted(payloads.items()):
            data = (source / relative).read_bytes()
            if len(data) != descriptor["bytes"] or digest(data) != descriptor["sha256"]:
                raise ValueError("source artifact integrity differs: " + relative)
            entry = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
            archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        entry = zipfile.ZipInfo("manifest.json", (1980, 1, 1, 0, 0, 0))
        archive.writestr(entry, json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n", compress_type=zipfile.ZIP_DEFLATED)
    data = destination.read_bytes()
    return {"archive": destination.name, "bytes": len(data), "sha256": digest(data),
            "models": [{"path": a["path"], "bytes": a["bytes"], "sha256": a["sha256"],
                        "source_name": model_name if a["id"] == "det" else "cls.onnx"} for a in assets]}


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
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imports-only", action="store_true", help="Append the immutable semantic importer fixture to an existing index")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.imports_only:
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
