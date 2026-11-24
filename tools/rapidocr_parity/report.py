"""Summarize two complete Rust captures against the fixed per-preset Python output."""
import argparse
import json
from collect import FILES, OUT, save
from compare import main as compare


def main(before, after):
    summary = {}
    for preset in FILES:
        images = {}
        totals = {}
        for name in [before, after]:
            images[name] = compare(f"reference/{preset}", f"{name}/{preset}")
            path = OUT / f"reference-{preset}-vs-{name}-{preset}.json"
            totals[name] = json.loads(path.read_text("utf-8"))["summary"]
        regressions = []
        for path, baseline in images[before].items():
            current = images[after].get(path)
            if baseline["exact"] and (current is None or not current["exact"]):
                regressions.append({"path": path, "result": current or totals[after]["errors"][path]})
        summary[preset] = {"totals": totals, "exact_to_different": regressions}
    save(OUT / f"{before}-to-{after}.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before")
    p.add_argument("after")
    args = p.parse_args()
    main(args.before, args.after)
