# Apple M4 narrow convolution optimization

Automatic NEON execution now keeps selected narrow convolutions in channels-last layout and uses the existing native dense kernel. On the physical Apple M4, the retained change reduced Tiny sparse OCR median latency by **7.7%** and Small sparse, dense and mixed-width latency by **22.0%, 3.8% and 9.4%**, respectively. These four before/after sample ranges were disjoint. The Small recognition backbone also benefits because its 48→24 2×2 convolution uses the new path.

The final Medium mixed-width median was **7.0% slower**, also with disjoint observed ranges. Medium contains no layers matching the added dispatch families, its outputs and retained buffers were identical, and its process CPU median changed by only **+0.1%**. The wall-time regression remains unexplained; those observations do not prove it is noise. This change establishes useful latency reductions on the measured Tiny sparse and Small inputs, rather than a universal or Medium speedup.

The subsequent [Medium mixed-width investigation](macos-arm64-medium-mixed-regression.md) keeps these historical measurements separate. Reversed-order controls did not reproduce a stable 7% gap. Owner profiling exposed a long-crop recognition bottleneck, and a bounded scheduling correction reduced Medium mixed latency by 5.6% against this retained build and 6.8% against the original in a fresh comparison, while preserving the measured Tiny/Small gains.

The [portable evidence](reports/macos-arm64-2026-10-17-narrow-convolution-optimization.json) preserves final raw wall/CPU samples and outputs, memory, version fingerprints, qualification results, focused diagnostics, intermediate summaries and reproduction scripts. The [preceding stage analysis](macos-current-pipeline-analysis.md) explains why sparse detector computation and narrow convolution paths were selected after an unsuccessful recognition pointwise experiment. Production changes are confined to the [NEON layout eligibility rule](../src/runtime/optimized.rs#L194), with independent [dense-kernel tests](../src/runtime/nhwc/dense/neon_tests.rs#L124) and [planner/session tests](../src/runtime/optimized/tests/selection_policy.rs#L190).

Previously, group-one NHWC convolutions required at least sixteen output channels and complete sixteen-channel panels. Narrow layers therefore used canonical NCHW execution, whose ARM specializations retain Portable4 arithmetic, and interrupted adjacent NHWC segments. Existing packed NEON dense execution already handles padded coefficient panels, partial output channels, bias, borders and final fused stores. The new rule admits only these additional families on `NeonFma`, with constant weights/bias, group one and dilation one:

- **2×2:** at least eight input channels; 8, 12 or 24 output channels; stride one; zero top/left padding; output height and width equal input height and width.
- **3×3:** at least 32 input channels; 24 output channels; symmetric stride one or two; top/left padding one; output dimensions equal the input dimensions divided by stride and rounded up.

Other ISA policies and explicit NCHW retain their previous selection. No new reduction kernel, unsafe code, model, dependency, recognition setting or accuracy tolerance was added. On the shipped graphs, the new families cover Tiny DET 16→8, Small DET 24→12 and 12→24 2×2 layers, Small DET 48→24 stride-two and 96→24 stride-one 3×3 layers, and Small REC 48→24 2×2. The classifier and both Medium graphs have no matching channel/kernel family.

These layers intentionally move from separate multiply/add to the existing ordered native FMA path. They preserve the numerical output contract rather than previous float bits. Across the final nine page comparisons, every non-score result field matched exactly, including text, order, geometry, rotation and recognition dimensions. The largest score differences were 1.073×10⁻⁶ for detection and 1.526×10⁻⁵ for recognition; classification scores were identical. Full reference qualification passed with unchanged tolerances.

Final measurements used native release/default-feature builds on an Apple M4 with four performance and six efficiency cores, 16 GiB RAM and macOS 27.0.1. Each fresh process used an eight-worker context, one request, automatic NEON/layout/fusion, detector limit 960, adaptive recognition width, batch size one, classification enabled and alignment disabled. Inputs were predecoded BGR24. One first inference, one warm-up and exactly three owned-result samples were recorded per case/version, with version order alternating by case. Lower latency is better.

| Model / input | Original ms | Retained ms | Latency reduction | Sample ranges overlap |
| --- | --- | --- | --- | --- |
| tiny / sparse | 52.182 | 48.163 | 7.70% | no |
| tiny / dense | 71.346 | 68.555 | 3.91% | yes |
| tiny / mixed | 92.033 | 87.089 | 5.37% | yes |
| small / sparse | 141.702 | 110.500 | 22.02% | no |
| small / dense | 206.146 | 198.308 | 3.80% | no |
| small / mixed | 293.019 | 265.369 | 9.44% | no |
| medium / sparse | 428.144 | 431.083 | -0.69% | yes |
| medium / dense | 737.909 | 755.791 | -2.42% | yes |
| medium / mixed | 1027.544 | 1098.974 | -6.95% | no |

Sparse, dense and mixed refer to `imgs_1` (720×1150), `imgs_00018069` (430×267) and `imgs_00006737` (896×528). Different bundles produce different detections, so these costs do not establish equal accuracy between model sizes. Tiny dense/mixed and Medium sparse/dense sample ranges overlap; their median changes alone do not establish improvements or regressions. Three samples and disjoint observed ranges are not a statistical significance test. No unchanged-code reruns were made for more favorable timings.

The focused detector diagnostic used the real sparse image at `[1,3,960,608]`, eight workers, one warm-up and one instrumented production pass per model/version. Grouped convolution time locates the effect:

| Fused production group | Original ms | Retained ms |
| --- | --- | --- |
| tiny DET Conv 2x2 | 7.399 | 3.685 |
| small DET Conv 2x2 | 13.526 | 4.985 |
| small DET Conv 3x3 | 31.623 | 11.187 |

These are separate diagnostic executions, including fused stores, rather than additive predictions for the ordinary OCR table. A synthetic segment test also verifies that an eligible narrow 2×2 convolution preserves NHWC continuity, reducing that test graph from three layout conversions to one. The existing packed weights are reused.

The change trades some Small activation capacity for latency. Tiny and Medium retained activation capacity did not change. Small results are:

| Small input | Original peak RSS MiB | Retained peak RSS MiB | Additional retained activations MiB |
| --- | --- | --- | --- |
| sparse | 139.55 | 146.00 | 6.68 |
| dense | 115.23 | 116.70 | 1.22 |
| mixed | 171.59 | 177.39 | 5.58 |

Small peak resident memory increased by 1.3–4.6%, and retained activations by 1.22–6.68 MiB. Shared packed weights and retained kernel scratch stayed unchanged. Existing memory limits and request-retention accounting remain active. Tiny sparse used 6.1% more process CPU time despite its lower wall latency; Small CPU medians decreased by about 5.2–5.5%. These results support a latency claim for the measured cases, not a general CPU-efficiency or throughput claim.

A distinct recognition experiment blocked the existing NEON pointwise reduction into K=256 or K=512 chunks across each 48-row group, using 3 KiB of partial sums. It preserved complete output bits but did not establish a useful gain. One warm-up and three single-call samples were measured per shape/policy in fixed rotated order:

| Rows / K / columns | Full K median ms | K256 median ms | K512 median ms |
| --- | --- | --- | --- |
| 96 / 768 / 384 | 0.494750 | 0.507708 | 0.508542 |
| 960 / 1024 / 512 | 8.457667 | 8.471459 | 8.419250 |
| 480 / 1536 / 768 | 9.214500 | 9.550750 | 9.395542 |

The blocked variants were slower on two shapes and overlapped the original on the other. The prototype was archived and removed from production/test sources. The earlier 2×2-only dispatch build was measured separately before adding the 3×3 path: Tiny sparse improved 12.0% and Small sparse 4.1% in that comparison. Those intermediate samples are not pooled with the retained build or used as its validation.

All validation ran locally on the physical machine. The retained source passed **395 default-feature checks** (345 library tests, 45 integration tests and five doctests), **334 serial library tests**, and **347 profiling-enabled library tests**. The full reference suite passed **213/213 requests**, including all 66 saved model/fixture expectations and serial/parallel, worker budgets, batching, alignment, pixel formats, classification toggles, dilation, reuse and recovery. Eight focused tests cover the new dispatch, independent FMA ordering, bias, padding, channel tails, fused operands and worker guards.

Native all-target/all-feature checking, x86_64 macOS cross-checking and Clippy passed. Changed files pass rustfmt, and whitespace checks pass. Whole-tree `cargo fmt --all -- --check` reports one pre-existing formatting difference in unchanged `src/runtime/nhwc/pointwise/large_channel_tests.rs`; its bytes match the frozen baseline. Ten library diagnostics and one local integration test remain ignored by the ordinary test suite. x86 compilation does not establish native x86 performance.

Retained performance execution consumed **17.784 seconds**, including startup/model loading and excluding builds/correctness checks. It contains nine first inferences, nine warm-ups and 27 ordinary samples, plus three detector warm-ups and three instrumented passes: 51 OCR/prefix executions. The 2×2-only trial consumed **17.141 seconds**, including its fixed kernel diagnostic. The unchanged original consumed **61.815 seconds** across the preceding diagnosis and both comparisons. Its aggregate bound was explicitly expanded from 60 to **70 seconds** before the final nine-cell comparison, to establish the effect of the added 3×3 path across all shipped bundles and representative inputs; a stage probe could not answer that question. Candidate bounds stayed at 60 seconds. Every page worker had an eight-second deadline and a 25-second per-version comparison cap. Diagnostic processes had ten-second deadlines.

The production source and executable fingerprints stayed fixed during the retained measurements. Raw local artifacts and scripts are under `target/arm64-kernel-adoption-20261017`; reference qualification is under `target/qualification/1792237557098-40648`. The evidence includes the original and retained source hashes, binaries, commands, model/input hashes, patch and rejected prototype. The original production source is commit `7ba96f9dc1bef61a7ad4a15c09f4f6278904f703`; the retained version is the documented worktree patch.

Relevant verification commands are:

```sh
cargo test --release --locked --lib neon_narrow
cargo test --release --locked
cargo test --release --locked --no-default-features --lib
cargo test --release --locked --features profiling --lib
cargo check --release --locked --all-targets --all-features
cargo check --release --locked --all-targets --all-features --target x86_64-apple-darwin
cargo clippy --release --locked --all-targets --all-features
cargo build --release --locked --example cpu_benchmark --example cpu_qualify
# Load the prepared local environment so dotnet is available.
target/release/examples/cpu_qualify
```

For another performance comparison, freeze both standard release executables, use a fresh output directory, keep fixed sample counts and explicit execution budgets, and keep builds/other inference idle. The stored driver uses the hash-checked jobs from the preceding analysis and records cumulative execution time. These results cover one physical M4 and single-request latency. Other ARM CPUs, native x86 timings, concurrent throughput, image decode, serialization and token-box output remain outside the performance scope.
