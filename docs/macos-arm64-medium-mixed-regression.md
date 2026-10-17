# Apple M4 Medium mixed-width recognition fix

Automatic NEON recognition now permits one uniquely dominant, single-crop owner to use two kernel workers within the existing fixed CPU pool. A fresh three-version comparison measured Medium mixed-width OCR at **925.462 ms**, versus **980.204 ms** for the retained narrow-convolution build and **992.592 ms** for the original build: **5.58% and 6.76% lower latency**, respectively. Both observed sample ranges were disjoint from the fix. Tiny and Small mixed-width latency also improved. Complete outputs and retained-memory diagnostics matched the retained build exactly across all nine measured cases.

The earlier **6.95% Medium gap did not reproduce consistently** in the controlled follow-up. The investigation establishes a recognition critical-path bottleneck and a measured improvement from correcting it; it does not establish that the narrow-convolution change caused the earlier slowdown. The historical measurements remain in the [narrow-convolution report](macos-arm64-narrow-convolution-optimization.md). The [portable evidence](reports/macos-arm64-2026-10-17-medium-mixed-regression.json) preserves controls, profiles, raw ordinary samples, complete outputs, validation, hashes, source and reproduction scripts.

## Cause and correction

Medium contains no convolution matching the new narrow dispatch families. Its original and retained outputs, activations, workspace and packed weights were identical. Frozen binaries also retained the same addresses and sizes for the hot NEON pointwise/dense kernels; differences examined in their instructions were late constant, call or panic references. This weakens an arithmetic or kernel-selection explanation, without proving that every possible binary-layout effect is absent.

The first diagnostic used the two unchanged frozen executables in **A B B A A B** process order on Medium mixed only. Each process performed one first inference, one warm-up and one measured inference: exactly three measured samples per version. This includes adjacent identical-executable controls and reverses the order in the middle pair.

| Frozen version | Wall samples ms | Wall median ms | Process CPU median ms |
| --- | --- | ---: | ---: |
| A: original | 1041.810, 968.468, 970.083 | 970.083 | 6537.149 |
| B: narrow convolution | 968.419, 1018.546, 997.090 | 997.090 | 6522.487 |

The gap fell to **2.78%**, with overlapping ranges and almost unchanged CPU work. The same executable varied by 50–73 ms across processes. In the later final comparison B's Medium mixed median was 1.25% lower than A's, again with overlapping ranges. These separate observations must not be pooled into one effect estimate. macOS reported AC power and no recorded thermal/performance warning; no frequency or physical-core residency measurements were available, so a particular P/E-core placement is not established.

Recognition ownership exposes the concrete bottleneck. Medium mixed has 31 crops totaling 9280 padded width units. Stable longest-first assignment gives owner 0 the single 1472-pixel crop. The other seven owners receive 1056–1152 width units each; the ideal average is 1160. All eight owners previously used one kernel worker. Moving shorter lines cannot split the largest crop or reduce its lower bound. Work stealing individual lines would also disturb the established ownership of shape caches and activation/resize buffers.

One instrumented production pass before and after the correction recorded:

| Medium mixed owner timing | Before ms | Fix ms |
| --- | ---: | ---: |
| All owner dispatch starts | 0.004–0.024 | 0.003–0.030 |
| Long-crop owner completion | 829.068 | 680.031 |
| Last other owner completion | 727.689 | 632.785 |
| Long-crop tail after all other owners finish | 101.379 | 47.246 |
| Recognition phase wall | 829.080 | 680.041 |

Before the fix, the long owner used **788.433 ms of calling-thread CPU** during its 829 ms interval. Prompt dispatch and substantial CPU work locate the problem in the indivisible crop's execution rather than launch order. The profile is one separate diagnostic pass per version, with timer overhead and host variation; its 18% phase reduction is not the official speedup. After splitting, owner CPU excludes helper work and must not be summed as process CPU.

The [scheduling rule](../src/ocr.rs) preserves stable assignment and existing error/result ordering. It selects at most one owner when all of these hold:

- Automatic NEON scheduling has at least eight active owners and a base kernel budget of one.
- The request has its full CPU share.
- A unique maximum-cost owner has exactly one job containing exactly one crop.
- Its width cost exceeds the average assigned owner cost.

That owner receives two kernel workers; all others retain their ordinary budgets. Each page resets the budgets, including after balanced pages, explicit settings and reduced request capacity. The rule uses geometry and configuration rather than model names, fixture IDs or elapsed timing. Ties and multi-crop batches are excluded. AVX and explicit positive line-worker policies retain their prior behavior.

The existing kernels supply the parallelism. For the long Medium crop, representative late pointwise layers have 2208 rows for 512→1024 channels and 1104 rows for 768↔1536 channels, about 1.16 and 1.30 billion MACs. These exceed the existing two-million-work sharding threshold. Two bands contain complete six-row tiles and preserve each row's reduction order. Nested Rayon work uses the same bounded pool and can use workers as their own recognition jobs finish. No new reduction kernel or additional OS thread pool is introduced.

The optional `profiling` feature now exposes fixed-capacity owner records with dispatch intervals, Rayon worker index, calling-thread CPU time on macOS, jobs/lines, width costs and actual kernel budgets. Default library builds contain no owner telemetry fields, clocks or added libc dependency. A Rayon worker index is a logical pool identity, not a physical CPU ID. `Ocr::cpu_budgets()` continues to report base budgets; owner records expose the individual two-worker hint.

## Final comparison

All ordinary measurements ran locally on the physical Apple M4, four performance and six efficiency cores, 16 GiB, macOS 27.0.1. Native release builds used default features, eight shared workers, one request, automatic NEON/layout/fusion, classification enabled, adaptive widths, batch one and owned results from predecoded BGR24. Models, dictionaries and pixels were hash checked. Builds and other inference were idle during measurement.

Each case/version used one fresh process, one first inference, one warm-up and exactly three measured samples. Process order rotated **ABC / BCA / CAB** by case. A is the original build; B retains the Tiny/Small narrow convolutions; C adds the recognition correction. Positive percentages mean lower C latency.

| Model / input | A original ms | B narrow ms | C fix ms | C vs A reduction | C vs B reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny mixed | 80.988 | 78.846 | 76.304 | 5.78% | 3.22% |
| Small mixed | 270.930 | 255.958 | 223.147 | 17.64% | 12.82% |
| Medium mixed | 992.592 | 980.204 | 925.462 | 6.76% | 5.58% |
| Tiny sparse | 48.356 | 43.931 | 42.603 | 11.90% | 3.02% |
| Tiny dense | 67.387 | 65.639 | 67.000 | 0.57% | -2.07% |
| Small sparse | 122.982 | 94.684 | 97.036 | 21.10% | -2.48% |
| Small dense | 195.324 | 190.619 | 190.239 | 2.60% | 0.20% |
| Medium sparse | 379.341 | 376.578 | 383.737 | -1.16% | -1.90% |
| Medium dense | 686.185 | 690.060 | 677.978 | 1.20% | 1.75% |

Sparse, dense and mixed are `imgs_1`, `imgs_00018069` and `imgs_00006737`. All three mixed C ranges are disjoint from A and B. Tiny sparse is also disjoint from both. Small sparse/dense retain disjoint improvements against A; their B/C ranges overlap. Tiny dense and Medium sparse/dense overlap both comparisons and remain inconclusive. In particular, the Tiny dense +0.57% median does not establish an improvement. The slight Tiny dense and Small sparse increases against B have overlapping ranges. These sparse/dense pages do not select the new boost; their timing changes cannot be attributed to it.

The Medium mixed raw B wall samples were **968.062, 980.204, 996.245 ms**; C was **929.833, 915.539, 925.462 ms**. Process CPU medians were **6518.618→6532.218 ms**, a +0.21% change while wall latency fell. CPU/wall utilization rose from about **6.65 to 7.06**. Small mixed CPU increased 1.23%; Tiny mixed decreased 0.17%. The result supports a latency improvement through better use of the bounded pool, not reduced arithmetic work or a concurrent-throughput claim.

All B/C owned outputs were bit-for-bit identical in every measured case. All nine B/C activation, workspace, kernel-scratch, shared-weight and request-count diagnostics were also identical. Medium mixed retained 184100864 activation bytes, 19924042 workspace bytes and 65536 kernel-scratch bytes. Its peak RSS changed from 475.156 to 475.188 MiB. The prior Small convolution activation tradeoff remains; this scheduling correction adds no measured retained-memory increase.

## Validation and bounds

The final source passed 397 default-feature checks (347 library, 45 integration, five doctests), 352 profiling library tests and 340 serial profiling library tests. Focused tests verify policy guards, ties, overflow, output equality to serial recognition, stable job ownership, warm buffers, mixed→balanced→mixed resets, reduced demand, explicit workers, batching and owner accounting. Native and x86_64 macOS all-target/all-feature checks and native Clippy passed. Cross-compilation does not validate native x86 performance.

Changed Rust files and whitespace checks pass. Whole-tree formatting still reports the pre-existing unchanged assertion formatting in `src/runtime/nhwc/pointwise/large_channel_tests.rs:25`. Ordinary suites retain ten ignored library diagnostics and one explicit local integration check.

Live qualification against the pinned C# reference passed **48/48 selected requests**: 21 mixed-width requests across all three bundles with automatic, serial, two/four explicit workers, batch two/four and alignment profiles, followed by 27 reuse/recovery requests. All comparisons passed with unchanged tolerances. This is focused qualification; the preceding narrow-convolution version's full 213-request qualification is historical and does not substitute for these final-source checks.

No unchanged-code reruns were made to obtain cleaner timings. The six-process control had 4.5-second process and 13-second per-version bounds, with the original cumulative cap explicitly expanded to 75 seconds beforehand. Final comparison had seven-second process and 24-second new per-version bounds. Before that comparison, the original cumulative cap was expanded to **95 seconds** to check the new correction against both earlier builds across all three bundles and inputs. The retained and candidate caps remained 60 seconds.

Final comparison execution consumed **14.949 s A, 14.487 s B and 14.517 s C**, including startup/model loading. A's cumulative execution was **86.093 s** and B's **41.586 s**. Separate diagnostic versions consumed **6.119 s before** and **4.794 s after** the correction. C's ordinary plus instrumented execution totaled **19.311 s**. There were 45 ordinary OCR calls per version, including first/warm-up calls, plus six diagnostic OCR calls per diagnostic build. Controls added nine OCR calls per frozen version. Builds and correctness qualification are separate from these performance budgets.

Local raw artifacts are in `target/medium-mixed-regression-20261017`. The portable report embeds their data, reproduction scripts and final source so ignored local files are not required to interpret the result. Source/executable fingerprints were fixed during each measurement scope. Three fixed samples and disjoint observed ranges are not a statistical significance test. The scope is one physical M4, single-request latency and the pinned model/corpus; camera acquisition, decoding, serialization, concurrent throughput and other hardware remain unmeasured.
