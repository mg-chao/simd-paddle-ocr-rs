# Metal performance evidence

Performance qualification is local to the physical machine and its inputs.
Use release builds, one cold invocation and at most three measured warm samples.
Run correctness first; do not pool samples from different source versions or
rerun unchanged code to obtain more favorable numbers.

The current single-Metal source `0badef5b4d61e8be` passed qualification and was
measured once on Apple M4 / macOS 27. Each preset used one cold invocation and
three warm samples on the typical image: one request, one slot, recognition
batch 2, wave 16, classifier capacity 16, MixedF16 storage and F16 matrix operands.
The complete performance wrapper took 13.034018291 seconds; builds and the
separate correctness suite are outside that time.

| Model | Rust median (ms) | Reused C# Metal median (ms) |
|---|---:|---:|
| Tiny | 19.649750 | 20.1564 |
| Small | 43.714667 | 43.2270 |
| Medium | 159.113708 | 161.4153 |

| Model | Rust warm samples (ms) | Reused C# warm samples (ms) |
|---|---|---|
| Tiny | 22.425708, 18.853041, 19.649750 | 22.2229, 20.1564, 18.7985 |
| Small | 45.996250, 43.714667, 43.087084 | 44.2365, 43.2270, 42.8181 |
| Medium | 161.230333, 158.155292, 159.113708 | 161.8799, 161.4153, 161.4062 |

The C# control is unchanged data from an earlier run, not a fresh paired
measurement. Tiny and Small sample ranges overlap. Small's Rust median is 1.13%
slower, leaving uncertainty around parity; Tiny's lower median alone does not
establish a reliable lead. Medium's Rust median is 1.43% faster in this sample,
with the preexisting 0.6993007% reduction in recognition input elements caused by
an accepted crop-width difference. Tiny and Small recognition tensor dimensions
match. These results do not establish that Rust is uniformly faster or qualify
other devices. Individual stage gains are not additive.

Text and line counts match the retained C# output for all three presets. One
line per preset has a preexisting orientation difference: C# reports label 1 /
180°, while Rust reports label 0 / 0°, with the same resulting text. Matching
tensor dimensions therefore do not imply identical crop contents. Current Rust
independently passes its CPU orientation contracts.

All nine strict and nine explicit F16 corpus cases passed the existing CPU
accuracy contracts. The same source passed 475 regular and 40 focused hardware
tests; the F32 classifier tensor check `[2,3,80,160]` had maximum absolute error
`7.748603820800781e-7`. See the [current evidence map](../validation/metal/current/README.md)
for results, executable hashes and preserved-artifact commands. Raw Rust samples
are in [the performance manifest](../validation/metal/current/performance/manifest.json),
and reused C# samples are in [the retained control](../validation/metal/pre-cleanup-bd26694591978bc1/csharp-control/manifest.json).
Historical data-only records remain separately labeled; their old source and
binary paths are provenance, not retained implementation artifacts.

For a fresh current-source measurement, first run the correctness procedure in
[GPU validation](gpu-validation.md), then use a fresh report directory:

```sh
python tools/gpu_performance.py \
  --corpus validation/metal/current/fast-corpus/corpus --report target/metal-performance \
  --presets tiny small medium --cases typical --targets gpu \
  --configs 1:1:2 --matrix f16 --storage f16 \
  --wave 16 --classification-batch 16 --alignment --measurements 3 --budget-seconds 25
```

The harness records decoded input hashes, exact configuration, native execution
statistics and timing scope. It verifies CPU qualification before recording a
median and exits unsuccessfully on any failed qualification. `--profile` runs
separate instrumented calls through the same native schedule. It records kernel
metadata and host timing; native per-kernel device durations are currently
unavailable (`device_ns: null`), so these records cannot establish GPU kernel
costs. C# comparisons are optional developer tooling: build
`tools/reference/Reference.csproj` against the unchanged sibling C# repository
and add `--csharp gpu`. A reused C# reference must be labeled explicitly; a C#
own-cold-output check does not substitute for Rust's independent CPU contract.
