# Metal backend: final five-round report

Validated locally on 2026-10-27 (Asia/Shanghai). The native Rust Metal backend is
implemented and passes the checks below. **The requested all-preset cold and
warm latency objective is not achieved: 33 of 48 acceptance gates pass.** All
15 cold-start and all nine accuracy gates pass; 15 of the 24 warm/first-shape
gates fail. Optimization stopped after the five additional rounds requested by
the user. Final qualification is v49, a fresh build of the retained v44 runtime
with the v42 build correction; it is not a sixth optimization round.

## Delivered behavior

- Native Metal device/context, checked shared buffers, uploads, quotas,
  submission, dependency barriers, completion-owned resources, device-loss
  handling and calibrated profiling counters.
- GPU neural operators, preprocessing, detection geometry, perspective crops,
  orientation classification, recognition projection, softmax and Unicode CTC.
  Strict GPU execution reports no CPU compute stages in the validated OCR paths.
- Offline MSL compilation, embedded portable IR and native M4 archive, per-context
  pipeline cache, dimension-independent shaders, and incremental relinking when
  the shader catalog changes. No Xcode installation is needed at deployment.
- Explicit FP32 and mixed storage policies. CPU execution and FP32 remain library
  defaults. Half detector storage is an explicit, workload-qualified opt-in;
  the benchmark uses FP32 detection for tiny and half detection for small/medium.
- Matrix bias/residual fusion preserves intermediate half rounding. Common
  convolution epilogues use fast paths with the existing arithmetic order.

Xcode 27.1 (27A9275) and its Metal Toolchain were installed and are operational.
The measured host is an Apple M4 Mac mini (10 CPU/10 GPU cores, 16 GiB), macOS
27.0.1. Rust is pinned to nightly-2026-09-08. The unchanged C# production reference
is commit `33ff8a81672452041d110e8ba52dcd2639e54228`, .NET SDK 10.0.303/runtime
10.0.11. See [GPU usage](gpu-usage.md#macos) for configuration and platform limits.

## Final measurements

The final campaign used 162 fresh launches and 468 outputs: three sample launches
per preset, 50 separate corpus-cold launches per preset, and one two-pass corpus
process per preset. It completed in 261.62 seconds within a declared 600-second
budget. The separately frozen C# baseline has the same counts and protocol.
No timing sample was discarded, retried or replaced with a prior candidate's
result. Earlier focused measurements do not qualify this final build.

Cold covers parent launch through the first owned result, including runtime,
raw input, model and GPU initialization. Warm covers decoded RGB to an owned
result. First-shape is the first corpus pass through one reused session. The
acceptance threshold is at least 5% lower sample medians and corpus means/medians,
with no p95 regression. Thus a small improvement can still fail the agreed gate.

| Preset | Sample cold median, C# → Rust ms | Sample warm median, C# → Rust ms | Corpus warm mean, C# → Rust ms |
|---|---:|---:|---:|
| Tiny | 337.40 → 112.14 | 30.85 → 29.68 | 60.94 → 72.66 |
| Small | 464.94 → 164.98 | 53.35 → 60.73 | 132.11 → 173.64 |
| Medium | 834.01 → 349.65 | 175.05 → 191.32 | 634.84 → 498.65 |

All cold gates pass. Tiny's sample warm median improves only 3.8%, below the
5% requirement; small and medium sample warm medians are slower by 13.8% and
9.3%. Corpus warm means are 19.2% slower for tiny, 31.4% slower for small, and
21.5% faster for medium. Medium passes every corpus latency gate. Tiny/small
first-shape means and medians and all their corpus-warm statistics fail.

Complete latency statistics are below; each cell is **mean / p50 / p95**, in ms.
Sample means are descriptive and are not acceptance gates.

| Preset | Metric | C# ms | Rust ms |
|---|---|---:|---:|
| Tiny | sample cold | 429.00 / 337.40 / 626.04 | 245.37 / 112.14 / 511.89 |
| Tiny | sample warm | 31.08 / 30.85 / 32.33 | 28.44 / 29.68 / 31.52 |
| Tiny | corpus cold | 428.55 / 411.36 / 491.70 | 168.85 / 166.49 / 213.06 |
| Tiny | corpus first shape | 73.07 / 61.90 / 160.08 | 80.56 / 74.37 / 123.55 |
| Tiny | corpus warm | 60.94 / 56.46 / 104.70 | 72.66 / 69.82 / 108.47 |
| Small | sample cold | 513.97 / 464.94 / 617.10 | 165.51 / 164.98 / 171.68 |
| Small | sample warm | 53.04 / 53.35 / 53.64 | 61.21 / 60.73 / 62.28 |
| Small | corpus cold | 616.61 / 594.27 / 847.02 | 298.38 / 291.23 / 424.25 |
| Small | corpus first shape | 145.84 / 133.60 / 333.39 | 180.91 / 168.48 / 306.13 |
| Small | corpus warm | 132.11 / 125.61 / 238.43 | 173.64 / 164.51 / 299.80 |
| Medium | sample cold | 835.91 / 834.01 / 856.98 | 363.17 / 349.65 / 390.23 |
| Medium | sample warm | 175.93 / 175.05 / 178.84 | 191.21 / 191.32 / 191.35 |
| Medium | corpus cold | 1209.42 / 1175.04 / 1658.39 | 695.30 / 676.35 / 1015.23 |
| Medium | corpus first shape | 597.55 / 587.34 / 1025.01 | 499.99 / 485.69 / 828.17 |
| Medium | corpus warm | 634.84 / 598.98 / 1116.51 | 498.65 / 486.57 / 828.15 |

## Correctness and accuracy

- 69 physical Metal checks passed on the final rebuilt source. These include
  operator/reference comparisons, half IEEE edges, fused rounding, batching and
  tails, ownership, quotas, concurrency, archive hits, profiling, geometry,
  crops, classifier, projection and CTC.
- All 54 OCR fixture comparisons passed with unchanged assertions: nine images
  across three presets in FP32, plus 27 under the benchmark precision policies.
- Default CPU library tests: 383 passed, three explicitly ignored.
- Benchmark policy tests: eight passed. The opt-in real Metal incremental-build
  regression passed separately at v42; the build implementation is unchanged.
- Formatting and whitespace checks pass. C# production sources remain clean.

All nine corpus CER gates pass. Cold, first-shape and warm phases have identical
CER for each preset. The permitted increase is 0.05 percentage points.

| Preset | C# CER | Rust CER | Difference, percentage points |
|---|---:|---:|---:|
| Tiny | 2.3501% | 2.3468% | -0.0032 |
| Small | 2.3275% | 2.3436% | +0.0161 |
| Medium | 2.4760% | 2.4598% | -0.0161 |

CER is the agreed spatial OCR metric over all 50 XFUND Chinese validation pages,
not the official semantic entity/relationship score. It includes missing and
unmatched text, applies NFKC/whitespace normalization, and preserves case and
punctuation. The existing Rust tall-crop rotation/metadata contract is preserved;
it differs from the C# rotation convention.

## The five final rounds

| Round | Candidate | Decision |
|---|---|---|
| 1, v43 | Matrix bias/full-residual Add fusion with intermediate half rounding | Retained; exact rounding and OCR checks pass. |
| 2, v44 | Fast paths for common convolution bias/GELU and bias/residual epilogues | Retained; unchanged arithmetic and OCR checks pass. |
| 3, v45 | Prefetch both matrix fragments in each direct K16 step | Removed; tiny sample regressed, no material gain elsewhere. |
| 4, v46 | C# three-term erf approximation in half-storage neural shaders | Rejected by the unchanged small vertical-text fixture; no timing run. |
| 5, v47 | Four-pixel depthwise vector stores with IEEE edge fallback | Removed; no meaningful overall improvement. |

The retained final implementation uses the original five-term GELU approximation,
original matrix fragment sequencing, and scalar depthwise stores. No accuracy
tolerance was relaxed to retain a candidate.

## Artifact integrity and limits

The first attempted final campaign, v48, is invalid and excluded. Restoring files
with their old modification times caused Cargo to reuse the removed v47
executable. Binary hashes exposed the mismatch. That campaign was aborted;
source input timestamps were refreshed, the backend genuinely rebuilt, and all
GPU/fixture checks repeated before v49. The invalid campaign and its explicit
marker remain available for audit; its measurements are never pooled with v49.

Before v49 measurement, all 243 runtime/build source hashes matched the retained
v44 source, the executable differed from v47, and its embedded portable shader
payload matched v44. After measurement, all 457 benchmark-tracked source hashes,
the frozen binary/source archive, and fixture executable hashes were verified
against the delivered tree. Benchmark executable SHA-256:
`50bee452ce9afbfa3da6ef92cbc5cf5f00f755e37025cb6b75b1938706abc5fd`.

Only this physical M4/macOS combination was qualified. Other supported Apple
Silicon devices, OS releases, and the Windows Vulkan path were not freshly
qualified by this campaign. Fresh processes do not purge OS filesystem or Metal
driver caches. macOS first-launch/security costs remain included, and three
sample launches give limited p95 precision. The C# baseline was captured earlier
in this session, so time-dependent machine/OS variation remains a limitation;
no causal explanation is asserted for the final tiny sample's variability.

Historical profiling identified recognition convolutions and many small neural
dispatches as substantial costs. The final end-to-end run does not isolate their
individual contributions. Meeting the remaining warm target would require
further work on those paths, potentially changing resident tensor layout and
scheduling, followed by fresh correctness and complete-corpus qualification.
That work is outside the requested five remaining rounds; it is not performed
or represented as a guaranteed improvement.

## Evidence and reproduction

Raw reports are local files under `target/metal-rebuild/` (ignored build output):

- [Final comparison, all 48 gates](../target/metal-rebuild/comparison-final-v49.json)
- [Final Rust manifest, outputs and hashes](../target/metal-rebuild/rust-final-v49/manifest.json)
- [Pinned C# manifest](../target/metal-rebuild/csharp-baseline/manifest.json)
- [Final artifact audit](../target/metal-rebuild/retained-final-v49-audit.json)
- [Pre-measurement artifact audit](../target/metal-rebuild/artifact-audit-final-v49.json)
- [GPU test log](../target/metal-rebuild/hardware-final-v49.log) and [CPU test log](../target/metal-rebuild/cpu-final-v49.log)
- [FP32 fixtures](../target/metal-rebuild/fixtures-final-v49-f32/manifest.json),
  [tiny benchmark fixtures](../target/metal-rebuild/fixtures-final-v49-tiny/manifest.json),
  [small/medium benchmark fixtures](../target/metal-rebuild/fixtures-final-v49-f16/manifest.json)
- [Five-round ledger](../target/metal-rebuild/final-five-rounds.json) and
  [complete experiment ledger](../target/metal-rebuild/optimization-ledger.json)
- [Invalid v48 marker](../target/metal-rebuild/rust-final-v48/INVALID-QUALIFICATION.json)

To inspect the existing evidence without rerunning OCR, from the Rust repository:

```sh
source ../env.sh
python3 tools/metal_comparison.py \
  --csharp target/metal-rebuild/csharp-baseline \
  --rust target/metal-rebuild/rust-final-v49 \
  --output target/metal-final-comparison.json
```

Exit status 1 is expected because the speed acceptance gates fail. For a fresh
measurement, first follow the [build and correctness commands](metal-validation.md#reproduction),
then use new report directories and the frozen input list:

```sh
python3 tools/metal_benchmark.py run --implementation csharp --scope all \
  --inputs target/metal-rebuild/inputs/inputs.json \
  --report target/csharp-metal-new --budget-seconds 600
python3 tools/metal_benchmark.py run --implementation rust --scope all \
  --inputs target/metal-rebuild/inputs/inputs.json \
  --report target/rust-metal-new --budget-seconds 600
python3 tools/metal_comparison.py \
  --csharp target/csharp-metal-new --rust target/rust-metal-new \
  --output target/metal-new-comparison.json
```

Each new executable needs its own qualification. Do not reuse source timestamps
from a rollback to establish Cargo freshness; rebuild changed inputs and verify
binary/source hashes. Keep all raw launches and report any failed gate.
