# Scalar DET and oriented-crop measurements

This milestone ends at ordered, oriented BGR crops. Recognition, image decoding,
manual SIMD, threading and GPU execution remain separate milestones.

## Evidence and method

[Parity evidence](step12-parity.json) records twelve expanded gates on Windows
and Linux. Each platform covers 33 variant-cases, 97 detected lines, 7,832 DET
node tensors, 10,282 CLS node tensors and 18,797 Step12 artifacts. Dimensions,
box/line ordering, labels, rotations and pixels are exact; floating tensors,
geometry and scores use `atol=1e-5`, `rtol=1e-4`. Both platforms also pass the
116 normal tests, formatting, Clippy, both seven-model import contracts, the
original 32-line full CLS gate and stateful public API checks.

Latency uses six alternating fresh-process C#/Rust pairs per mode and corpus, with
30 warmups and 100 measurements. Three allocator-instrumented Rust processes follow
each mode and corpus separately. Each process runs its case/stage rows in sequence;
`-Case` restricts a run to one selected case. Model loading, initial extractor preparation, additional session
binding and first execution have separate records. Warm intervals exclude file
I/O, hashes, traces and serialization. Graph calls include copying input in both
engines; C# explicitly uses its span overload. Production pixel calls fill Rust
session input directly.

Cold preparation has engine-specific boundaries. C# `prepare_ms` includes row
inputs, graph sessions and detector/classifier setup. Rust `prepare_ms` covers
the persistent extractor; additional graph and stage bindings are recorded in
`session_preparations`. Individual preparation fields therefore do not support
matched cold-start speedup claims. Warm row measurements use the matched scopes
described above.

Samples run in contiguous blocks per case/stage. Corpus samples sum matching
sample indexes; they represent aggregate latency rather than a manifest-order
wall sweep or throughput. Process medians and pooled samples are reported
separately. Process variation and scope prevent a universal speedup claim.

Compact corpora exercise rectangular, blank and tiny inputs. Tiny/small compact
inputs produce no lines; medium's portrait produces one. A separate rotated
500×500 page uses a 512×512 DET binding, produces 16 lines and applies
180-degree classification rotation to 15 of them.
Mixed-shape calls execute 14 bindings with 12 distinct 32–128 pixel shapes,
including large→small→large revisits on one session. Production page, wide,
tall and padded-stride shapes also pass the expanded parity gates.

## Measured latency

The [collection index](step12-inference-performance.json) links four independently
validated reports: [tiny compact](measurements/tiny-compact.json),
[small compact](measurements/small-compact.json),
[medium compact](measurements/medium-compact.json), and
[the rotated page](measurements/tiny-page-rotated.json).
These are Windows measurements. Each p50/p95 below is the median of six process
percentiles in milliseconds; it is not the percentile of pooled samples.
Compact values sum the four case rows; the page is one image.

| Pixel-to-oriented-crop corpus | Lines | C# p50 / p95 ms | Rust p50 / p95 ms |
| --- | ---: | ---: | ---: |
| Tiny compact | 0 | 7.329 / 8.236 | 6.700 / 7.323 |
| Small compact | 0 | 24.350 / 27.274 | 20.578 / 23.703 |
| Medium compact | 1 | 230.358 / 251.754 | 162.409 / 180.607 |
| Tiny rotated 500×500 page | 16 | 798.097 / 910.839 | 789.210 / 1059.903 |

Rust's compact process medians are lower. Page p50 is similar and Rust's p95 is
higher. Page process p50 ranges are 742.7–920.9 ms for C# and 602.7–895.8 ms for
Rust, so these measurements support no consistent page speedup claim.

The following frozen-stage values sum matching sample indexes within each
process before calculating its percentiles. They do not decompose the complete
pixel-pipeline wall time. Public CLS includes crop preprocessing and the
production backbone prefix/head; the separate dense `cls_graph` results remain
in each child report.

| Frozen stage scope | Calls per sample | C# p50 / p95 ms | Rust p50 / p95 ms |
| --- | ---: | ---: | ---: |
| Medium compact DB | 4 maps | 0.03560 / 0.04176 | 0.02125 / 0.02232 |
| Medium compact crop | 1 crop | 0.09870 / 0.11740 | 0.09013 / 0.09366 |
| Medium compact public CLS | 1 crop | 9.165 / 12.113 | 5.987 / 6.694 |
| Rotated page DB | 1 map | 1.095 / 1.545 | 0.928 / 1.381 |
| Rotated page crop | 16 crops | 6.738 / 7.199 | 5.685 / 6.073 |
| Rotated page public CLS | 16 crops | 214.680 / 225.183 | 146.477 / 158.886 |

Every fixed graph, pipeline and stage row passed the three-process zero-allocation
gate. Mixed-shape calls include rebinding: Rust allocates 266 times per 14-binding
sequence, requesting 2,142,224 bytes for tiny/small and 2,141,216 bytes for medium.
Those counts describe planning and capacity work, not warmed fixed-shape calls.

Each corpus preserves its original executable hashes and source inventory.
Fresh Rust links differ in PE timestamps/PDB GUIDs; the collection records a
read-only comparison masking only those fields, with identical code/data sections.
The page inventory also omits an unbuilt C# native-math capture file after an
external ignore-rule change. These differences remain explicit in the index;
the strict validator was run separately without weakening its checks or combining
cross-corpus raw samples.

## Memory interpretation

Fixed-shape graph and warmed borrowed pipeline/stage calls must allocate zero
times and zero bytes. Shape rebinding and capacity growth have separate allocation
records. Owned-result copying is measured independently and includes dropping the
new result. Arena counters distinguish planned lengths from retained capacities;
first-call/phase heap peaks include transient staging. These are requested heap
bytes rather than allocator overhead.

Activation counters include the persistent setup extractor and simultaneously
live additional benchmark sessions. Mixed-shape counters describe the final plan
and retained capacities. Workspace/slab/metadata counters describe the persistent
extractor; standalone stage scratch is included in process memory rather than
these workspace fields. Shared model weights and the preloaded corpus are outside
activation counters. Process resident memory includes validation and loading and
is sampled as a high-water value; a final observation can be missed. C# allocation
volume is current-thread managed allocation, with native/retained heap unavailable.

Pipeline retained activation and workspace counters from the three Rust heap
processes are below. Workspace totals include the crop slab and line storage
(metadata and transforms); those subset counters must not be added again.

| Corpus | Planned / retained activation MiB | Workspace total MiB | Crop capacity bytes |
| --- | ---: | ---: | ---: |
| Tiny compact | 0.470 / 0.559 | 0.011 | 0 |
| Small compact | 0.536 / 0.691 | 0.007 | 0 |
| Medium compact | 0.778 / 1.176 | 0.026 | 5,280 |
| Tiny rotated page | 23.193 / 23.193 | 1.034 | 249,462 |

The page pipeline's first call peaks at 24,911,430 requested heap bytes above its
phase baseline; this includes growth and staging and is separate from activation
capacity. Sampled whole-process high-water resident medians are 88.44 MiB for C#
and 48.29 MiB for Rust's uninstrumented page pipeline, including startup, models
and the preloaded corpus. These resident values are not comparable heap peaks.

Page `try_to_owned()` plus dropping the copy has Rust p50 0.004775 ms and p95
0.008305 ms. Each copy requests 251,254 bytes in two allocations, peaks at that
requested size above its baseline, and retains zero bytes after dropping.
Extraction is outside this separate owned-copy interval; there is no C# pair.

## Retained tuning

Node profiling identified dense convolution as dominant. Four independent output
channels share input/tap work while retaining bias-first, input-channel, kernel-row
and kernel-column accumulation order for each output. No scratch allocation was
added. Before/after profiles and three fresh-process paired experiments favored
this change for all three detector variants. Those diagnostic runs used three
warmups and ten measurements and are separate from the final six-pair protocol.

[Tuning evidence](step12-tuning.json) retains those earlier prototypes. The final
observer-instrumented node profile confirms the ranking: Conv accounts for
86.6% of the 594.463 ms mean interval sum, Erf 9.0%, and ConvTranspose 1.0%.
The largest observed node is `det.node.0234` Conv at p50 91.240 ms. Observer timing
is a diagnostic ranking and is excluded from production latency statistics.

The native exponential compatibility change is required for parity. Linux `expf`
first differed at an Erf node; a widened standard exponential corrected a subset
but failed broader Windows gates. The scalar AMD table/cubic/FMA reconstruction
matches 100,010 independent native exponential captures on both platforms and
all 78 isolated Erf tensors. Graph arguments, returned values and surrounding
arithmetic remain F32.

Historical [import measurements](step12-performance.md) and
[CLS measurements](cls-inference.md) preserve their original values and protocols.
The historical CLS C# array overload bound external input directly, while Rust
copied it; the new matched-copy measurements use C#'s span overload.

## Reproduction

```powershell
./tools/benchmark-step12.ps1 -Reference reference-bundles/tiny-step12-compact-v1 -Out target/measurements/new-tiny-run
./tools/benchmark-step12.ps1 -Reference reference-bundles/small-step12-compact-v1 -Out target/measurements/new-small-run
./tools/benchmark-step12.ps1 -Reference reference-bundles/medium-step12-compact-v1 -Out target/measurements/new-medium-run
./tools/benchmark-step12.ps1 -Reference reference-bundles/tiny-step12-expanded-v1 -Case sample-180 -Out target/measurements/new-page-run
python tools/summarize-step12-performance.py --measurements target/measurements/new-tiny-run --out target/measurements/new-tiny-report.json
python tools/summarize-step12-performance.py --measurements target/measurements/new-small-run --out target/measurements/new-small-report.json
python tools/summarize-step12-performance.py --measurements target/measurements/new-medium-run --out target/measurements/new-medium-report.json
python tools/summarize-step12-performance.py --measurements target/measurements/new-page-run --out target/measurements/new-page-report.json
```

The summarizer rejects missing rounds, changed models/corpora, mismatched input
rows, nonzero fixed-shape warm allocations and overwritten output identities.
Raw reports preserve samples, executable/source/model/manifest hashes, timings,
allocations and memory counters. Large references have immutable local archives.
The [measurement archive manifest](step12-measurement-archive.json) records the
verified raw evidence archive; the [baseline catalog](baselines.json) records the
three fresh all-node reference archives.
Keep one validated child report per corpus and link their exact SHA-256/size
identities in a collection index such as `step12-inference-performance.json`.
The published page child additionally validates `--owned` against its same
executable/source identity sidecar; independently rebuilt owned binaries require
their own matching measurement snapshot.
