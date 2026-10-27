"""Evaluate the agreed complete-corpus latency and CER gates; never retries OCR."""
import argparse
import json
from pathlib import Path

from metal_benchmark import save, summary_rows
from metal_accuracy import report_accuracy


def compare(csharp, rust, annotations):
    manifests = [json.loads((root / "manifest.json").read_text()) for root in (csharp, rust)]
    for manifest, implementation in zip(manifests, ("csharp", "rust")):
        if manifest["implementation"] != implementation or not manifest["complete"]:
            raise ValueError("both complete implementation manifests are required")
        if manifest["protocol"]["scope"] != "all" or set(manifest["protocol"]["presets"]) != {"tiny", "small", "medium"}:
            raise ValueError("acceptance requires sample and all 50 corpus pages for all three presets")
        if len(manifest["runs"]) != 162 or any(not r["summary"]["backend_verified"] for r in manifest["runs"]):
            raise ValueError("missing launches or unverified GPU execution")
    for key in ("hardware", "machine", "platform", "csharp_revision", "inputs_sha256", "models_sha256"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"comparison conditions differ: {key}")
    for key in manifests[0]["protocol"]:
        if key != "budget_seconds" and manifests[0]["protocol"][key] != manifests[1]["protocol"].get(key):
            raise ValueError(f"comparison protocol differs: {key}")
    rows = [{(r["preset"], r["metric"]): r for r in summary_rows(m)} for m in manifests]
    gates = []
    for preset in ("tiny", "small", "medium"):
        for metric in ("sample_cold", "sample_warm", "corpus_cold", "corpus_first_shape", "corpus_warm"):
            a, b = (values[(preset, metric)] for values in rows)
            expected = 3 if metric.startswith("sample") else 50
            if a["n"] != expected or b["n"] != expected:
                raise ValueError(f"wrong sample count: {preset} {metric}")
            for statistic in (("p50_ms", "p95_ms") if expected == 3 else ("mean_ms", "p50_ms", "p95_ms")):
                ratio = b[statistic] / a[statistic]
                limit = 1.0 if statistic == "p95_ms" else .95
                gates.append(dict(preset=preset, metric=metric, statistic=statistic,
                                  csharp=a[statistic], rust=b[statistic], ratio=ratio,
                                  required_ratio=limit, passed=ratio <= limit))
    accuracy = [report_accuracy(root, annotations) for root in (csharp, rust)]
    accuracy_rows = [{(r["preset"], r["phase"]): r for r in a["summary"]} for a in accuracy]
    for preset in ("tiny", "small", "medium"):
        for phase in ("cold", "first_shape", "warm"):
            a, b = (values[(preset, phase)] for values in accuracy_rows)
            difference = 100 * (b["cer"] - a["cer"])
            gates.append(dict(preset=preset, metric=f"cer_{phase}", statistic="percentage_points",
                              csharp=100*a["cer"], rust=100*b["cer"], difference=difference,
                              maximum_difference=.05, passed=difference <= .05))
    return dict(csharp=str(csharp.resolve()), rust=str(rust.resolve()),
                latency_and_corpus_accuracy_passed=all(g["passed"] for g in gates),
                note="Existing Rust fixture correctness gates must also pass separately. Three sample launches give limited tail precision; no samples are discarded.",
                gates=gates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csharp", required=True, type=Path)
    parser.add_argument("--rust", required=True, type=Path)
    parser.add_argument("--annotations", type=Path, default=Path(__file__).resolve().parents[2] / ".local/datasets/xfund-zh-val/zh.val.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.csharp, args.rust, args.annotations)
    save(args.output, result)
    for gate in result["gates"]:
        print(f"{'PASS' if gate['passed'] else 'FAIL'} {gate['preset']} {gate['metric']} {gate['statistic']}: C# {gate['csharp']:.4f}, Rust {gate['rust']:.4f}")
    raise SystemExit(0 if result["latency_and_corpus_accuracy_passed"] else 1)


if __name__ == "__main__":
    main()
