# Complete scalar OCR

The runtime executes all seven shipped networks: three detectors, three recognizers,
and the shared orientation classifier. `Ocr` accepts borrowed interleaved pixels and
returns ordered text, quadrilateral boxes, scores, rotations and optional CTC spans.
Image decoding remains caller-owned. Existing constructors and result ownership
contracts apply to every family; immutable models and dictionaries share through
`Arc`, while each mutable engine owns its reusable storage.

## Models and references

| Family | Source detector | Source recognizer | Source dictionary | CTC classes | Default box threshold |
|---|---|---|---|---:|---:|
| tiny | det.onnx | rec.onnx | ppocr_keys.txt | 6,906 | 0.4 |
| small | small_det.onnx | small_rec.onnx | rec_keys.txt | 18,710 | 0.45 |
| medium | medium_det.onnx | medium_rec.onnx | rec_keys.txt | 18,710 | 0.45 |

Reference bundles retain logical role IDs `det`, `cls`, `rec`, and `dictionary`.
Resolve files through their descriptors rather than guessing a dictionary filename.
Bundle schema remains version 1, and historical tiny references remain readable.

```powershell
./tools/export-reference.ps1 -Model small -InputBundle reference-bundles/tiny-ocr-ci-v2 -Nodes all -MaxTracedLines 0 -Out reference-bundles/small-new
cargo run --release --locked --offline -- run-rec reference-bundles/small-new --out target/small-rec-new
cargo run --release --locked --offline -- compare-rec reference-bundles/small-new target/small-rec-new --json
cargo run --release --locked --offline -- run-ocr reference-bundles/small-new --out target/small-ocr-new
cargo run --release --locked --offline -- compare-ocr reference-bundles/small-new target/small-ocr-new --json
```

Use `-Model medium` for the larger family. `-DetectorModel` retains its existing
Step12 detection/crop meaning. `-InputBundle` explicitly reuses verified source
pixels and case IDs while selecting new models. `-ReplayBundle` instead preserves
the original family and effective options; a conflicting family fails. Output
directories and published fixture identities are never overwritten.

## Shapes, arithmetic and memory

Small and medium recognition include unfused attention and LayerNorm. MatMul accepts
rank-two-or-higher F32 operands with checked leading-dimension broadcasting;
activation operands participate in lifetime planning. Slicing supports constant
integer parameters and positive unit steps. Unsupported ranks, steps and parameter
combinations return errors.

Dynamic metadata preserves scalar versus vector integers, original aliases and
operator-derived dimensions. Normalized import inspections and folded weight bytes
remain unchanged. Symbolic dimension names are opaque. The metadata program remains
bounded to 64 instructions and eight-element shape vectors.

Graph reductions and projection accumulation retain their source order. Pow supports
the shipped scalar exponent two through F32 multiplication; independent native
fixtures record its rounding differences from `MathF.Pow`. The complete operator,
LayerNorm and all-node gates retain `atol=1e-5`, `rtol=1e-4`. Shapes, text, token counts,
rotations and CTC boundaries compare exactly. Floating geometry and scores use the
established numerical contract. Public compact scores remain raw projection maxima,
separate from dense Softmax probabilities.

Recognition height stays 48. Adaptive width rounds upward to a multiple of 32 with
minimum 32 and no silent 320 cap. Fixed mode retains its configured width. Public
recognition runs a prefix ending before vocabulary projection and keeps only winners
and scores; dense diagnostics initialize separately when tracing is requested.

Attention storage grows quadratically with timesteps. Checked shapes and planned
activation bytes reject over-budget requests before allocation. The default limit
remains 1 GiB per session, excluding shared weights, retained result/workspace
capacity, owned copies and transient replacement buffers. Shape changes remain
transactional. Warm zero-allocation guarantees require warming both shape plans and
all relevant activation, resize, text and span capacities, including changed content.

## Production profiling

Enable `profiling` for `Recognizer::recognize_profiled` and `Ocr::run_profiled`.
Each returns the usual borrowed result plus a fixed-size `StageProfile` with
nanosecond durations for binding, preprocessing, DET, postprocessing/crops, CLS,
recognition backbone, compact projection, CTC and result-slab assembly. Ordinary
calls perform no timer reads. Profiling uses the production prefix and does not
initialize dense diagnostic storage. Explicit owned conversion is measured separately.

```powershell
cargo run --release --locked --features profiling --bin ocr-benchmark -- --reference target/step12-fixtures/medium-ocr-ci-v1 --mode public_stage_profile --warmup 3 --samples 10
```

Profile totals include instrumentation overhead and are excluded from official
latency and allocator results. The older node-profile mode observes the dense
diagnostic graph and has a distinct execution scope.

## Repeatable benchmark builds

Prepare one frozen build for a process series, then reuse its exact timing,
allocator and native exporter bytes across all model families and C# profiles:

```powershell
./tools/prepare-ocr-benchmark.ps1 -Out target/benchmark-build-v1 -References @('target/step12-fixtures/tiny-ocr-ci-v1', 'target/step12-fixtures/small-ocr-ci-v1', 'target/step12-fixtures/medium-ocr-ci-v1')
./tools/benchmark-ocr.ps1 -Reference target/step12-fixtures/small-ocr-ci-v1 -PreparedBuild target/benchmark-build-v1/build.json -Out target/small-canonical-v1 -Modes @('rec_public_owned', 'ocr_owned', 'owned_copies') -CSharpProfile canonical -Case image-000 -Rounds 6 -HeapRounds 0 -Warmup 30 -Repeat 100 -ProcessorAffinityMask 1
```

Use a fresh output directory for each family/profile group. `-CSharpProfile cpu`
measures optimized single-worker C# separately. Run on an idle host with a common
recorded affinity; preserve all six process records and their variation. Hosted
CI has no latency threshold. Preparation, allocator probes and profiling belong
to separate runs.

The prepared manifest binds source snapshots, compiler/runtime versions, build
flags, features, models, dictionaries and effective options. Changed identities
fail before inference. Runs copy and verify the frozen executables without
relinking. Standalone builds resolve their executable from Cargo's artifact
output, including custom target directories. `-StopFile` preserves completed
reports and lets the active process finish before stopping the series.

## Native validation

Each hosted complete fixture serves both REC and OCR tests. Required Windows/Linux
CI validates all three families, verifies corruption/omission rejection and asserts
all four instrumented warmed allocation/growth counters. Fixtures include upright
and rotated text, blank and one-pixel input. Expanded local references preserve the
original 32-line sample/rotation corpus, including wider native recognition shapes.

`tools/verify-all-models.py` runs CLI REC/OCR comparisons and retains source,
executable, reference and comparison identities in `proof.json`. Historical tiny
measurements and local validation reports retain their original scope; current
remote CI evidence is recorded separately.
