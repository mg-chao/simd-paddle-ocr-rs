# Full GPU OCR: physical-device qualification

This extends the [2026-10-22 inference baseline](gpu-validation.md). The pinned
Rust toolchain remains `nightly-2026-09-08`; wgpu remains **30.0.1**. CPU remains
the default execution target. GPU-selected OCR defaults to `Full`;
`InferenceOnly` preserves the hybrid comparison path.

## Coverage contract

[The checked-in coverage summary](full-gpu-coverage.json) pins all seven model
hashes, input contracts and observed operator counts. Run
`python tools/gpu_validation.py inventory` to regenerate every original ONNX
node, attribute, tensor boundary, shape, dtype and initializer descriptor.
All bundled models include shape metadata for their original tensor boundaries.
No observed operator variant is intentionally unsupported. Arbitrary external
ONNX variants remain subject to `GpuCompiledModel::support_report` and strict
shape/capacity validation; an unrecognized variant is an error, never CPU work
inside strict GPU execution.

The nine deterministic fixtures are typical, tiny (1×27), sparse, dense (2×2
page), vertical, wide, perspective, border and holes. Wide fixtures exercise
recognition tensor widths **608–768**, beyond 320. Fixture manifests pin the
source image, generator, Pillow version, encoded input hashes and dimensions.
Synthetic tests cover exact crop bytes for four pixel formats and row padding,
foreground/background connectivity, dilation, candidate exhaustion, equal-key
reading order, vocabulary ties, blanks/repeats, Unicode lengths, empty outputs,
span/batch boundaries, budgets, context identity and resource lifetimes.

Neural tensors and scores use `1e-3 + 1e-3 × |reference|` in FP32 and
`5e-2 + 1e-2 × |reference|` with mixed storage. Text, orientation, count and
reading order must match exactly; corresponding coordinates must differ by
at most **0.5 source pixels**. These gates are unchanged. The C# CPU comparison
and the three **failing C# GPU recognition comparisons** remain in the earlier
report; they are not relabeled as successful Rust evidence.

## Reproduction on a physical device

Install Pillow 12.3.0 for corpus generation. Models remain local; the harness
does not download them. Use the same commands on AMD, NVIDIA, Intel and Apple,
selecting an available native backend (`vulkan`, `dx12`, or `metal`). Software
adapters, CI and remote runners are not qualification evidence.

```powershell
cargo build --release --locked --all-features --examples --tests
python tools/gpu_validation.py inventory --report reports/full-gpu/coverage
python tools/full_gpu_validation.py --backend vulkan --presets tiny small medium --all-cases --report reports/full-gpu/qualification-vulkan
python tools/full_gpu_validation.py --backend vulkan --precision mixed-f16 --modes cpu full --presets tiny small medium --all-cases --report reports/full-gpu/qualification-vulkan-mixed
python tools/gpu_validation.py compare --backend vulkan --report reports/full-gpu/tensors-vulkan
```

Set `SIMD_OCR_GPU_BACKEND` to the same backend before the hardware tests
(PowerShell: `$env:SIMD_OCR_GPU_BACKEND="vulkan"`; POSIX shells:
`export SIMD_OCR_GPU_BACKEND=vulkan`).

```powershell
cargo test --release --locked --all-features --lib gpu_ -- --ignored --nocapture --test-threads=1 --skip gpu_bounded_performance
cargo test --release --locked --all-features --lib
cargo check --locked --all-targets
cargo check --locked --no-default-features --features gpu --all-targets
```

The hardware suite checks portable kernels, mixed storage and supported optional
subgroups. Cooperative matrix selection is explicitly experimental: static Naga
validation is separate from physical execution. On hardware lacking the pinned
8×8 FP32 configuration, its test verifies portable selection instead. Injected
device-loss tests exercise policy; they do not force a physical device reset.

Qualification manifests include compiler, lock/toolchain, source, executable,
model and input identities; backend/adapter/driver; exact commands; numerical
failures; actual placement; transfers and selected kernels. The harness rejects
a release executable older than its build inputs. Traced transfers are reported
separately. Normal resident runs must have no CPU compute stages or diagnostic
transfers. Oriented-crop/tensor APIs deliberately read their requested output.

For a bounded performance comparison, first finish correctness, then run:

```powershell
python tools/full_gpu_benchmark.py --backend vulkan --corpus reports/full-gpu/qualification-vulkan/corpus --report reports/full-gpu/performance-vulkan
```

This freezes the executable and input identities and runs sparse/typical/dense
workloads, hybrid/full placement, and one/two simultaneous requests. Each
configuration has one cold run followed by exactly three warm measurements.
The process budget is 60 seconds, excluding compilation; no automatic sampling
expansion occurs. Cold construction and cold inference are separate from warmed
latency and throughput. Use a fresh report directory per source version.

## Physical qualification and measured results

Full OCR's precision policy protects discrete outputs. In a mixed context,
detector and recognizer weights/activations stay FP32; classification keeps
mixed storage. The detector safeguard prevents contour changes and coordinate
drift. Retaining recognition weights in FP32 also prevents one-column CTC span
changes that remained with FP32 attention residuals alone. These are GPU
precision choices, reported in stage statistics, with independently cached
weights on the same device. No CPU fallback is involved.

The preserved Vulkan mixed `InferenceOnly` comparison failed OCR semantics on
**12 of 27** fixtures/presets despite passing tensor error gates. Its behavior
is retained for compatibility and reported as failed qualification. The
earlier detector-only precision safeguard left three Full-mode span failures;
the final recognizer safeguard addresses them. Neither the coordinate nor
score tolerances were widened.

DX12 required explicit rounding barriers in compensated polygon-score sums and
the two-FP32 floor helper. Otherwise compiler reassociation erased compensation
or rounded `173.5 - 0.000001` upward, moving boxes by a source pixel. The
release `gpu_shader_check` example now checks eight positive/negative floor and
rounding boundaries against host double precision on the selected physical
backend. The large-polygon score regression retains its stricter `1e-5` check.

```powershell
# Set SIMD_OCR_GPU_BACKEND as above, then:
cargo run --release --locked --all-features --example gpu_shader_check
```

Qualification completed locally on **2026-10-23**, using the RX 6750 XT:

| Physical device / backend | Driver | Full FP32 | Full mixed context | Hardware tests |
| --- | --- | --- | --- | --- |
| AMD RX 6750 XT / Vulkan | 26.10.1 | 27/27 pass | 27/27 pass, precision guards above | 32/32 pass |
| AMD RX 6750 XT / DX12 | 32.0.21045.16010 | 27/27 pass | 27/27 pass, precision guards above | 32/32 pass |
| NVIDIA / Vulkan or DX12 | unmeasured | unverified | unverified | unverified |
| Intel / Vulkan or DX12 | unmeasured | unverified | unverified | unverified |
| Apple / Metal | unmeasured | unverified | unverified | unverified |

The final corpus comprises **216 CPU/Full runs**, including **108 Full GPU
runs**. Text, orientation, box count, reading order and CTC spans match exactly;
maximum corresponding coordinate error is **0.0 source pixels**. All score
comparisons pass their specified precision gates. Normal Full runs report no
CPU compute stages, no intermediate image/feature/vocabulary downloads, and
zero diagnostic readback bytes.

The 104 neural comparisons cover all seven models, batches 1/2, rectangular
detectors, and recognition widths 32/128/320 in FP32 and mixed storage on both
backends. All pass. Their neural kernels and storage bindings are unchanged by
the final geometry-only rounding correction. Earlier FP32 hybrid corpus
comparisons pass 27/27 on each backend. Every report pins its own source and
executable; historical and corrected runs are kept separately.

Release regression checks pass **353 tests** with default CPU features and
**365 tests** with all features. Default, GPU without default features, and
all-feature builds pass. Borrowed-output compile checks accept valid usage and
reject overlapping mutable session borrows with `E0499`. Both native backends
also pass the eight two-FP32 rounding checks. The experimental cooperative
matrix path is **not physically qualified**: this adapter lacks the requested
configuration, so tests verify portable selection. Other AMD devices are also
unverified.

[Machine-readable results](full-gpu-results.json) contain model/input/build
identities, commands, failures, placements, transfers, kernel selections and
raw timing samples. Complete local evidence and frozen binaries/source archives
are under ignored `reports/full-gpu/`. The retained C# GPU recognition failures
remain failures in the historical comparison.

### Bounded performance results

These measurements use the tiny preset, FP32 Vulkan, recognition batch size 2,
and one or two simultaneous requests; `max_in_flight` matches request count.
Each configuration has one cold invocation and three warm measurements. The
final 12-configuration experiment took **15.16 seconds** in total. The current
harness checks cold batched/concurrent text against the CPU corpus before any
warm measurement. No startup benchmarking selects production kernels.

Median milliseconds per request group (two-request rows finish two images):

| Fixture | Requests | Baseline hybrid | Initial correct Full | Final hybrid | Final Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| sparse | 1 | 11.02 | 11.31 | 15.83 | 9.31 |
| sparse | 2 | 13.67 | 18.93 | 12.41 | 15.64 |
| typical | 1 | 104.66 | 60.86 | 104.30 | 54.49 |
| typical | 2 | 82.90 | 93.19 | 74.45 | 84.62 |
| dense | 1 | 271.99 | 175.07 | 234.00 | 159.34 |
| dense | 2 | 271.78 | 287.57 | 250.40 | 260.84 |

Final Full improves median group times over the initial correct Full version
across these six configurations. Typical single-request samples are
54.49/54.63/53.83 ms; dense samples are 160.68/159.30/159.34 ms. Sparse timings
are noisy: Full samples are 19.30/9.31/9.09 ms, and hybrid samples are
27.09/15.83/11.35 ms. Do not infer a reliable sparse-workload speedup from their
medians alone.

Single-request Full is about **1.91× faster** than current hybrid on typical
pages and **1.47× faster** on dense pages. With two concurrent requests,
**hybrid retains higher throughput** in this experiment: typical Full/hybrid
are 23.63/26.86 images/s; dense Full/hybrid are 7.67/7.99 images/s. Full residency
does not imply universally lower latency or higher throughput. Other presets
have correctness coverage; these end-to-end performance numbers apply only
to the stated tiny-preset workloads and physical device.

Cold construction and first inference are separate from warm latency:

| Final Full, one request | Construction ms | First inference ms | Warm median ms |
| --- | ---: | ---: | ---: |
| sparse, first Full process | 149.30 | 3391.81 | 9.31 |
| typical | 153.21 | 116.58 | 54.49 |
| dense | 148.54 | 211.84 | 159.34 |

The first hybrid construction took 2985.09 ms. These are fresh processes and
request objects, not a controlled reset of the driver's shader cache. Pipeline
compilation is included in first-use inference; later cold rows must not be
advertised as first-installation startup latency.

Per-image payload transfers and final Full application device memory, one
request and batch size 2:

| Fixture | Hybrid upload bytes | Full upload bytes | Hybrid download bytes | Full output + control bytes | Full peak MiB | Full retained MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse | 1265664 | 150000 | 232448 | 2256 | 15.52 | 14.50 |
| typical | 10598400 | 750000 | 1477376 | 28144 | 42.42 | 37.41 |
| dense | 40427520 | 3000000 | 5370880 | 110032 | 122.03 | 104.25 |

Hybrid counters are cumulative over four calls and are divided by four here;
Full counters describe the last call. Uploads count image/tensor payloads,
excluding immutable model uploads and uniform packets. Peak memory excludes
opaque driver allocations. Dense Full with two requests peaks at 246.94 MiB.

The isolated projection comparison used one warmup and three GPU timestamp
samples per candidate. Median one-row/four-row tile times were 169.64/141.24 μs
for 16×80×6906, 606.76/567.28 μs for 80×80×6906, and 3612.80/1243.12 μs for
80×192×18710. The last four-row samples were variable, but all were faster than
the one-row samples. Four-row selection is restricted to FP32 features, at
least 16 rows, and vocabularies of at least 4096 classes.

The earlier scratch-cache experiment was inconclusive and its larger retained
memory was undesirable, so caching remains opt-in. SE-chain fusion is limited
to cases where the following gate multiplication can be absorbed; ordinary
convolution and attention kernels avoid unused specialization branches. Raw
rejected-candidate measurements are retained alongside the initial and final
results. No measurements from other GPU families are implied.
