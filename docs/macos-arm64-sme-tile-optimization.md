# SME convolution tile and crossover optimization

The retained change reduces complete OCR latency by **7.5–21.3% for Small** and **13.4–22.1% for Medium** on the physical Apple M4 in a fresh comparison with `52ed1be`. It combines a 32×32 SME block with a wider set of eligible recognition convolutions. All nine paired OCR outputs match exactly, retained activation/workspace/weight counters are unchanged, and the full **213-request reference qualification passes**.

Tiny has no demonstrated gain: its sparse, dense and mixed medians are **12.3%, 3.2% and 2.0% slower** in this comparison. A separate stage profile and instruction inspection did not establish a cause for the sparse result. It remains an unresolved measurement limitation, not a fixed regression or a speedup claim. The evidence supports adopting this as a Small/Medium optimization; it does not establish an across-the-board improvement for every bundle.

The [preceding pipeline analysis](macos-sme-pipeline-analysis.md) identified pointwise convolution as the main recognition target and highlighted expensive shapes below the original SME threshold. [Portable evidence](reports/macos-arm64-2026-10-17-sme-tile-optimization.json) includes every raw page sample, exact outputs, selected stage profiles, source and binary hashes, kernel probes, instruction inspection, validation logs and reproduction data. Production settings, goldens and tolerances remain unchanged.

The [SME kernel](../src/runtime/nhwc/pointwise/sme.rs) now computes a 32×32 block with four independent 16×16 FP32 accumulators. Each K step loads two input vectors and two weight vectors and reuses each twice. Full blocks therefore use one guarded streaming entry instead of four and halve the vector loads per four outer products relative to the original single-tile loop. This organization uses the four FP32 ZA tiles described in [Arm's SME Programmer's Guide](https://documentation-service.arm.com/static/664f013638084307512bb30c). Bias-first, ascending-K FMA order is retained for every output.

The existing 64-row input-packing workspace and immutable 16-column weight panels are reused. Four tile results occupy a fixed 4 KiB local array before the existing fused stores apply activation/residual operations. Short row or channel tails retain the 16×16 path. Both tile sizes validate SME availability through their caller, 64-byte streaming vector length, inactive streaming/ZA state, a null TPIDR2 save pointer and default FPCR. The guarded assembly has no calls, allocations or unwinding, exits streaming mode before returning, and declares vector/predicate clobbers. Failed eligibility or thread-state checks retain NEON behavior.

[Dispatch](../src/runtime/nhwc/pointwise.rs) now requires at least 32 rows per executing band, both channel dimensions at least 192, and at least one channel dimension at least 384. This admits the measured 192↔384 and 256↔512 layers while leaving smaller products on NEON. The gate remains restricted to macOS ARM64 and the NEON/FMA policy. The same eligibility function drives scratch reservation, so admitted layers cannot reach an unreserved packing path. No additional heap workspace was required in the nine measured cases because their larger existing layers already reserve the maximum tile.

Candidate selection began with three fixed synthetic layer shapes, one warm-up and three single-call samples for each kernel, including packing and final identity stores. The original SME tile was 56.0% faster than NEON for rows/K/N = 192/192/384 and 59.8% faster for 1920/256/512. This supported widening the crossover. A subsequent 32×32 prototype reduced SME median time by 38.2%, 18.8% and 46.7% for 192/192/384, 1920/256/512 and 960/512/1024 respectively. The middle prototype sample ranged from 0.903 to 1.425 ms, so its exact gain is uncertain. All full output arrays matched bit-for-bit. These are candidate-selection measurements, not final page speedups or a hardware throughput claim.

Page measurements use native release/default features, pinned `nightly-2026-09-08`, one request, eight shared workers, automatic dispatch/layout/fusion and line ownership, detector limit 960, adaptive widths, batch one, classification on and alignment off. The host is Apple M4 with 10 cores and 16 GiB RAM, macOS 27.0.1, on AC power. macOS scheduling is unpinned; physical-core placement and frequency are not measured. The sparse, dense and mixed inputs are `imgs_1`, `imgs_00018069` and `imgs_00006737`.

Each before/after case uses separate processes with alternating order, identical hash-verified predecoded BGR24 pixels, one first-inference call, one warm-up and three measured owned-result calls. Builds and other validation were idle during benchmarking. Values below are medians and complete measured ranges in milliseconds; negative reduction means slower. Decoding, serialization and service admission waiting are excluded.

| Bundle / page | Before ms | After ms | Latency reduction | Before range ms | After range ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny / sparse | 41.76 | 46.90 | -12.3% | 41.69–42.80 | 46.31–47.11 |
| Tiny / dense | 61.39 | 63.37 | -3.2% | 60.73–62.17 | 63.35–86.30 |
| Tiny / mixed | 73.39 | 74.85 | -2.0% | 72.50–74.67 | 73.45–75.58 |
| Small / sparse | 89.54 | 82.83 | +7.5% | 89.40–89.78 | 81.21–85.41 |
| Small / dense | 169.93 | 138.33 | +18.6% | 165.48–170.18 | 136.40–138.62 |
| Small / mixed | 218.68 | 172.19 | +21.3% | 216.88–223.90 | 170.72–173.05 |
| Medium / sparse | 350.65 | 303.82 | +13.4% | 345.99–357.16 | 302.60–310.43 |
| Medium / dense | 557.11 | 433.83 | +22.1% | 533.38–589.68 | 425.75–449.32 |
| Medium / mixed | 719.30 | 579.25 | +19.5% | 687.81–725.92 | 557.83–583.09 |

All six Small/Medium before/after ranges are disjoint in this local run. Three samples still do not establish tail percentiles or significance across machines. Compared with the earlier pre-SME baseline, these numbers measure an additional change on top of SME; historical samples are not pooled into this comparison.

The Tiny sparse slowdown was investigated rather than removed from the table. Its retained kernel scratch is 16 KiB both before and after, with no SME packing workspace. Its detector has no shapes eligible under either gate. The larger SME kernel is outlined, and the unchanged NEON pointwise arithmetic has the same 265 normalized instructions through return in both executables. A separate current-source profile totals 43.34 ms, versus the ordinary 46.90 ms median; its detector inference is 30.23 ms. This does not establish a stable code cause for the 12.3% ordinary slowdown. No fixture-specific bypass or speculative scheduling change was added. The direct candidate comparison below addresses the choice between implementations; its Tiny samples are not substituted into this original-baseline comparison.

A simpler candidate widened the dispatch gate while retaining the original 16×16 kernel. Its separate nine-case original-baseline comparison is preserved in the JSON. The unchanged baseline varied materially between the two rounds (for example, Small sparse was 89.54 ms in the combined round and 120.18 ms in the dispatch-only round), so comparing candidate medians from different rounds would be misleading. Tiny showed the opposite direction in that round. These samples do not establish a Tiny root cause and are not pooled or substituted for the table above.

To choose between candidates, one bounded direct comparison used three representative cases, alternating candidate order, one first call, one warm-up and three measured calls per binary, with a 12-second execution cap per variant. Both used the same native release settings and hash-verified inputs as above. The results favor retaining the 32×32 kernel in addition to the wider gate:

| Bundle / page | Dispatch only ms | 32×32 + dispatch ms | Additional latency reduction |
| --- | ---: | ---: | ---: |
| Tiny / sparse | 41.47 | 41.76 | -0.7% |
| Small / dense | 149.62 | 137.59 | +8.0% |
| Medium / mixed | 678.94 | 573.96 | +15.5% |

Small dense ranges were 136.67–140.74 ms for the combined implementation and 148.74–150.89 ms for dispatch only; Medium mixed ranges were 568.31–574.57 ms and 654.31–685.81 ms respectively. Tiny ranges overlapped (41.35–42.66 ms and 41.41–42.75 ms). These direct results support the Small/Medium candidate choice without resolving the earlier Tiny-versus-original result. All three candidate output projections and retained-memory diagnostics matched exactly. No further timing rounds were run after this decision.

Four separate production profiles diagnose the retained version. Each has one warm-up and one instrumented pass; these are not components of the ordinary medians. Nested worker work overlaps wall phases and is retained separately in the JSON.

| Bundle / page | DET prep | DET inference | DB | Crop | CLS | REC | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny / sparse | 3.44 | 30.23 | 0.42 | 0.34 | 1.39 | 7.51 | 43.34 |
| Small / dense | 0.68 | 16.94 | 0.29 | 0.78 | 20.46 | 117.43 | 156.60 |
| Medium / dense | 0.74 | 57.75 | 0.31 | 0.86 | 20.38 | 336.26 | 416.34 |
| Medium / mixed | 2.84 | 201.41 | 0.52 | 1.46 | 9.94 | 355.98 | 572.18 |

Recognition remains the largest phase on the sampled dense/mixed Small/Medium pages. The change improves arithmetic without changing stable line ownership or the shared worker pool. The earlier evidence about worker imbalance remains a useful follow-up topic, but this implementation does not claim to identify or solve its hardware/scheduling cause.

All nine before/after diagnostics match exactly for retained activations, workspace, kernel scratch, normalized weights, cached packed weights and request count. RSS changes range from approximately −0.30 to +0.59 MiB; these lifetime process peaks also include allocator/setup effects. The retained Medium mixed request peaks at 481.59 MiB. The 4 KiB local accumulator array replaces the smaller tile-local array and is not persistent request workspace.

Local validation passed **403 default checks** (353 library, 45 integration, five doctests), **341 serial library tests**, and **358 profiling-enabled library tests**. Eleven library diagnostics and one local integration check remain intentionally ignored by ordinary suites. The widened gate has platform/backend/boundary coverage; existing numerical tests now cover 31/32/33-row and channel edges, four-tile bias/reduction order, extremes/subnormals, fused indexed stores, offsets/tails, one/two/four/eight worker budgets, scratch reuse and insufficient-scratch fallback. Both assembly tile sizes are checked for live ZA and nondefault FPCR rejection. Session-level tests cover the newly admitted channel directions and workspace accounting across reshapes.

The full **213/213 Rust/C# qualification** passed with a 600-second external timeout. It exercises the 66 canonical fixture/bundle outputs plus concurrency budgets, exact-width batches, alignment/token boxes, rotations, pixel formats/stride, classification settings, dilation, reuse and recovery. Goldens and score/coordinate tolerances are unchanged. Nine ordinary comparisons also preserve every field of the benchmark result projection exactly, and four profile pairs preserve complete owned results exactly. Clippy with warnings denied and all targets/features passes; all-target/all-feature x86_64 macOS compilation passes. Cross-compilation is not native x86 execution or performance validation. Clippy initially found two issues in the ignored benchmark; its runtime debug guard and fixed sample loop were cleaned up. Rebuilt standard, profile and qualification executables are byte-identical to the measured/qualified binaries, confirming that this test-only cleanup does not change production behavior or invalidate those executable measurements.

Selected-version performance execution totaled **21.652 seconds**: 53 OCR executions from the nine original-baseline pairs and four profiles, plus 15 from the direct candidate comparison, or **68 OCR executions**. The dispatch-only alternative totaled **16.302 seconds**, including its separate nine-case comparison and the direct comparison. Baseline cumulative performance execution, including the preceding current-source analysis, crossover probe and both original-baseline rounds, is **46.849 seconds**. The superseded 32×32 kernel probe took less than 0.1 seconds. Each page/profile subprocess had at most ten seconds and the remaining aggregate cap; the final direct comparison had a 12-second cap per candidate. All variants remained below the 60-second performance-execution limit. Builds and correctness qualification are excluded; process startup/loading and all harness calls are included. There were no sampling-growth loops or retries. The extra candidate comparison was bounded to resolve baseline variability and the implementation choice, not to replace unfavorable results.

After candidate selection, the retained source was restored and rebuilt. Every Rust source hash matches the fully qualified version, and all three rebuilt production executables match the measured/qualified binaries byte-for-byte. The temporary dispatch-only implementation is retained solely as comparison evidence.

Relevant local checks are:

```sh
cargo test --release --locked --lib sme -- --test-threads=1
cargo test --release --locked
cargo test --release --locked --no-default-features --lib
cargo test --release --locked --features profiling --lib
cargo clippy --release --locked --all-targets --all-features -- -D warnings
cargo check --release --locked --all-targets --all-features --target x86_64-apple-darwin
source ../env.sh
target/arm64-sme-crossover-20261017/cpu_qualify
```

Frozen standard/profile binaries, jobs, scripts and raw logs are in `target/arm64-sme-crossover-20261017`. Use a fresh directory at the same depth for reproduction, rebuild/copy standard and profiling executables separately, and retain the hash-checked baseline artifacts expected by the embedded comparison driver. Its source hashes reject measurements after a source change; a new version requires a rebuild and a new run count. The portable report stores results independently of local target artifacts.

This validates one physical M4, three representative inputs and one page thread/concurrency configuration. Unsupported hardware retains the existing fallback. Other SME processors/vector lengths, native x86 performance, concurrent service throughput, model accuracy against human ground truth and p95/p99 latency were not measured. The Tiny timing limitation above remains visible when assessing whether the tradeoff fits a workload.
