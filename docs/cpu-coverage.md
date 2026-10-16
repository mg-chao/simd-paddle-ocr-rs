# Windows x64 CPU optimization coverage

This inventory compares active Windows x64 source with pinned C#
`33ff8a81672452041d110e8ba52dcd2639e54228`.
The active source is
`7f341123f48d3d82ad22b970ac627d296b17d8ffda3a08fe9a6ca38f4149a077`.
Its completed local core checks passed 279/272 all-target tests (264/257 library),
six documentation tests per feature mode, Clippy both modes with warnings
denied, 113 tooling contracts and the Debug example. The actual source/PE/PDB
Debug audit passed with a conservative maximum of 1,997,512 bytes (compact
projection 1,740,264), below the ordinary 2 MiB worker stack. **Release-library,
all-model, variants, batching, unavailable-512, Release static and package gates for
these exact bytes remain pending. Current performance is unmeasured.**
The rejected 1397 candidate failed two column-shard bounds tests (261/263
library passed); its failure evidence is retained separately. The corrected
active source's core tests do not substitute for fresh integrated acceptance.

Source `f7bc2ce6761f4521e9492468fe6c201b6a96e5526b5963250d878ee898c2242c`
and `canonical-stores-20261016` evidence are **historical**: 267/261 all-target,
252 Release-library, 6 docs per mode, 86 tool contracts, Clippy both modes, 7 exact
imports/447 comparisons, 1341 variants, 9 batch cases, unavailable-512 rejection,
scoped Debug/Release and 117-member package gates passed then. Its historical
Debug bound was 1,946,832 bytes. It proves selected canonical SIMD4 stores,
plane-one SIMD8 stores and isolated PadEnd2 reductions under the named audit
conditions. PackedStride2 retains eight vector stack operands and bounds checks;
generic PackedPointwise retains coefficient-tail/helper/spill overhead, and
general >25-tap Portable4 geometry retains its explicit fallback. Scoped
proofs do not establish universal register-only loops or current-source gates.
**Untagged retained full/variant/static passes in the inventory below refer to
historical evidence, not qualification of the active 7f341 source.**

The last measured 54a1 source passed 36/72 performance targets (17/36 latency,
19/36 throughput); prior a0bc passed 65/72. Neither qualifies current performance.
All five authorized original-sample rounds are consumed. The append-only
followup recorder exists, but the new human grant is pending and unrecorded;
no further real-model timing, including phase-only diagnostics, is authorized.
Actual AVX-512 execution remains explicitly user-deferred. Remote main protection
was absent, rulesets empty and only the old Rust core workflow registered in
the 2026-10-16 read-only receipts; the Windows workflow was unpublished.
No source has complete hardware/performance acceptance.

All C# paths below are relative to `src/Sdcb.SimdPaddleOCR/`; Rust paths are
relative to this crate. Large captures remain external/ignored, with producer
source/binary/compiler/input/backend/thread identities and failure evidence.

**Status legend:** `Partial` means the operation works but reference specialization
is missing; `Present` means a corresponding source implementation exists, with
focused source validation established by this inventory; `Missing` means
the specific optimization or policy is absent; `Excluded` means the pinned
reference cannot select it. None of these statuses establish acceptance.

## Required gates

1. **G1: Reproducible baseline.** Run the maintained M1 harness against all seven
   pinned models, both dictionaries, and the hashed raw corpus. Compare C# scalar
   to Rust scalar, C# optimized CPU to Rust optimized CPU, and Rust scalar to
   optimized CPU. Retain import, logical-node, public-head, component, and complete
   tiny/small/medium results plus compiler/executable/backend/thread identities.
2. **G2: Focused correctness.** Test each selector immediately below/at/above its
   shape thresholds; exercise empty bias, cancellation, groups, padding, odd
   dimensions, SIMD tails, fused intermediates with additional consumers, failed
   reshape/allocation recovery, and repeated shape reuse. Preserve accumulation
   and bias order, finite-value errors, signed-zero behavior where specified, and
   lowest-column tie breaking. A thread-count change within one kernel
   configuration must preserve results exactly.
3. **G3: Integrated parity.** Re-run G1 for every changed family. Keep tensor
   `atol=rtol=0.001`, scores `0.01`, coordinates `0.5 px`; compare text, labels,
   counts, order, dimensions, span boundaries, and frozen pixel bytes exactly.
   Do not automatically widen tolerances. Preserve first-divergence evidence.
4. **G4: Hardware and performance.** Execute each ISA on eligible hardware;
   cross-compilation does not qualify. Only after G3, compare frozen workloads at
   total CPU budgets 1/2/4/8, latency and concurrency separately, with loading,
   compilation, reshape, cold/warm inference, and ownership costs separated.
   Require warmed median latency <=110% and throughput >=90% of C#. Follow
   `AGENTS.md` and the explicit user exception; the five-round allowance is now
   exhausted for the original sample, with no further run authorized.

## NCHW convolution families

Canonical dispatch in `runtime/kernels.rs` calls `runtime/nchw.rs` before the
original whole-output/channel fallback. One selector drives execution, panel
consumption statistics, actual-ISA reporting, and fused-store eligibility.
Source-owned immutable 16-column panels are reused where their coefficients
are equivalent to the reference OC4/OC8/OC16 layouts. Scalar policy retains the
original independent executor. Native entries check their runtime CPU features;
loads, gathers, boundary taps, and partial channels remain bounds checked.

| Reference family / active selector | Rust equivalent and status | Focused evidence / remaining acceptance |
| --- | --- | --- |
| Canonical packed ordered SIMD epilogues and isolated PadEnd2 reduction | `nchw.rs::store_four_spatial`/`packed_pointwise_single`, `nchw/pad_end2.rs`; **Present** | Full-four spatial packets preserve ordinary contiguous NCHW operand indexing; full plane-one channel panels are contiguous. Borders/inactive tails remain scalar. Named PadEnd2 is restricted to existing N8/OUT8/group1 dispatch; checked spans prove local loads. Independent five-test capability RED, eight new GREEN tests, historical f7bc core/Release-library/447 parity, 1341 variants and Debug/scoped Release proofs pass; no performance claim. |
| Unpacked pointwise: unit stride/dilation, no padding, matching spatial size, including groups | `nchw.rs::spatial_kind` fixed1x1, grouped8/4/1-output reuse and spatial SIMD; **Present** | Independent bias-first oracle covers grouped planes, tiny widths, channel tails and batches. Retained G3 passes; G4 remains required. |
| Packed pointwise: group1, cout%4=0, reference sample/channel/spatial ownership | Channel-vector `packed_pointwise`; **Present** | Shared16-column panels avoid duplicate coefficient formats. Batched sample sharding and eligible flattened spatial bands preserve input order and every output bit. Tests include cout4/8/16/24/32, height1, 4097-position tails and68-channel windows. |
| OC-major pointwise: cout>=16, cout%16=0, plane<48; AVX-512 plus portable-vector fallback | Distinct `pointwise_oc_major` with8/4/1-position register reuse, bounded `[plane][16]` stack tile and planar transpose; **Present** | Plane47/48 selector tests and all available policies compare to an independent accumulation oracle. No full-model retained weight copy is needed. Real AVX-512 execution remains pending. |
| Pointwise blocked/tiled NHWC working set while final output remains NCHW: AVX2+FMA, cin>=32, cout>=16/%16, plane>=32, per-sample work>=4M | `pointwise_working_set`, bounded128/64-position input panels, ascending32-coefficient reduction blocks; **Present** | Input conversion is reused across output blocks; partial reductions remain raw until final stores. Tests cover plane1024/8193, all K blocks, sample workers and indexed epilogues. Retained G3 passes; G4 remains required. |
| Unpacked dense3x3: group1, unit stride/dilation, pad1, equal input/output spatial size | Const3x3 `spatial_kind`,16/8/4/1-output reuse; **Present** | Odd widths7/8/15/17/23, empty/nonempty bias, batches and tails match an independent per-tap oracle. |
| Packed dense3x3: same geometry, cout%8=0; AVX-512 deliberate single-thread large-cout fallback | `packed_dense<3,3,1>`, output-channel vectors and four-position reuse; **Present** | Thread-aware selectors preserve the AVX-512-to-AVX2 fallback and Count4/16-channel worker fork. Shared panels and consumed-panel reporting use the same selector. Retained G3 passes; native AVX-512 and performance acceptance remain pending. |
| Native paired packed 3x3: group1, dilation1, stride1, pad1, same spatial size, packed cout%8 prerequisite | `nchw.rs::packed_three_pair16`, reported `nchw_packed_3x3_pair16`; **Present** | Selects native policies only, threads==1, cout>=64 and cout%16==0. Portable4 never selects it; cout16 with threads>2 retains Dense3. Sixteen bias-first output chains share safe eight-spatial loads, preserving ci/ky/kx order; edges/tails remain scalar. AVX-512 deliberately uses the selected AVX2/FMA body. `nchw/audit_pair16.rs` tests 56/64/72/80 channels, 1/2/4 threads, short shapes, batches, guards and global epilogue indices. Fresh integrated/static/performance acceptance remains pending. |
| General dense stride1: group1, dilation1, K<=2^20, plain AVX horizontal interior>=8 | Dynamic-tap `spatial`,8/4/1-output reuse with safe narrow fallback; **Present** | Source-eligible AVX2/FMA cout%8<=64 projections split disjoint row bands to retain eight output accumulators. Independent row-band and global-index epilogue tests are bit exact. |
| Packed dense stride1: AVX2+FMA, cin>=16, cout%8=0,5x5/7x7, interior>=4, nonempty vertical interior, work>=4M | Shared-panel `packed_dense`, shape/work/worker gates; **Present** | Selector tests exercise minimum work, channel and worker limits; independent numerical tests cover boundaries and coefficient order. AVX-512 uses the reference AVX2 body. |
| Dense2x2 stride1, pad-bottom/right1, pad-top/left0 | Const2x2 `spatial_kind`,8/4/1-output reuse; **Present** | The source cout8/work>=4M/height>=2*workers row split uses disjoint scoped slices. Tests retain last row/column contributions and exact global epilogue indices. |
| Dense3x3 stride2: group1, dilation1, pad1, packed/unpacked routes | `packed_dense<3,3,2>` plus fixed strided `spatial_kind`; **Present** | Full output dimensions, batch/channel/tail coverage and Count4 worker fork are tested. Safe strided loads do not overread rows. The actual PackedStride2 reduction retains eight vector stack operands and bounds checks; this body is not claimed register-only. |
| Depthwise3x3/5x5: unit stride/dilation, symmetric same padding | Distinct output-vector and tap-major bodies; **Present** | Output-vector guards match source:3x3 batch1/height>=50/width>=10;5x5 batch1/height, width>=10, with8M channel-shard override. Tap-major final-valid-tap fusion needs no extra epilogue pass. Tiny1-pixel maps,49/50 and9/10 thresholds, edge arithmetic, ordered FMA regions and8M sharded5x5 are independently tested. |
| Depthwise7x7 unit stride, pad3 | Const7x7 output-vector strip with full scalar boundary rows and8M channel gate; **Present** | Independent per-tap oracle covers narrow widths, row boundaries, batches and SIMD tails. |
| Depthwise9x9 unit stride, pad4; packed and fixed/general channel paths | `packed_depth9` and four-position/channel-vector scoped row body; fixed9 strip and generic fallback; **Present** | Packed guards retain channels%8, dimensions>=9, plane<=4096, work>=1M. Source row ownership covers batches and low channel counts. All packed edges use FMA, with skipped taps bounds checked. Fixed AVX9 eligibility and forced-ISA guards are explicit. |
| Depthwise1x5;3x3/5x5 height-stride2;3x3 stride2 | Const-specialized tap-major kernels with safe contiguous/strided loads; **Present** | Source per-tap vector-end conditions, scalar tails and final valid-tap epilogues are tested across odd dimensions and output shapes. |
| General depthwise stride1, dilation1, arbitrary kernels | Dynamic-tap reusable row strips with nonempty-interior guard and source4M channel ownership; **Present** | Independent shape/edge/tail tests include1x7,9x9 fallback, batches and per-channel bands. Retained G3 passes; G4 remains required. |
|2x2 stride2 transposed convolution | `transpose_tile`,8/4/1-output reuse, four ordered tap accumulators and source8M channel ownership; **Present** | Bias-first, batches, widths3/4/7/8/9/17 and outputs1/4/8/9/16 match an independent oracle. Indexed stores cover every interleaved tap exactly once. |
| Canonical residual/activation epilogues in final stores | `try_conv_store`/`try_transpose_store` use borrowed `StoreTransform`; **Present** | Completed packed spatial quartets and plane-one channel panels transform contiguous SIMD vectors before their first store; borders, tails and remaining interleaved outputs retain absolute scalar indices. Reduction blocks transform only on their final K block. Independent atomic coverage tests require exact eligible vector packets and one transform per output; fixed policy goldens cover scalar/channel/full residuals and activation order. |

`runtime/nchw/tests.rs`, `runtime/nchw/audit_pair16.rs` and the additional
PreparedStore/graph tests in `runtime/optimized/tests/nchw_epilogue.rs`
establish internal
correctness against independent source-order arithmetic and exact worker/storage
invariants. Prior source `a0bcf050...` full/variant parity is recorded above;
these focused tests alone do not establish performance targets or AVX-512
hardware execution. G4 remains incomplete.

## Channels-last, projection, and tensor kernels

| Reference family / active selector | Rust equivalent and status | Work required / focused evidence |
| --- | --- | --- |
| NHWC graph segment and neutral-storage eligibility | `optimized.rs::conv_nhwc_capable`/`layout_neutral`; **Present** | Constant rank-four weights, optional constant rank-one bias, dilation1; group-one output>=16/%16; exact depthwise channels%8 only continues a segment. C1 or H=W1 uses storage-identical cached aliases; original base inputs, not fusion-appended residuals, decide starts. Independent normalized-graph/layout/arena tests pass; historical f7bc full 447 and variants 1341 passed. |
| Dense/depthwise row-worker policy | `nhwc.rs::conv_row_workers`, existing `parallel::aligned_bands`; **Present** | Dense>=2M/depthwise>=1M, bounded complete output rows with global pixel indices, capacity/workspace caps and no hot job allocation. Packed Scalar pointwise bypasses this row selector and retains six-pixel workers/full K. Threshold, C1, row/tail/batch, indexed stores and K585 bias-order cases pass; historical f7bc full 447 and variants 1341 passed. |
| NHWC pointwise: packed `[oc/16][ic][16]`; six rows, groups of eight tiles, full K | `nhwc/pointwise.rs`; AVX 6×16, Portable4 two paired 6×8 subtiles; **Present** | Native noinline tiles keep twelve accumulators and six presliced rows; retained AVX/FMA K loops have no checks, spills, calls or geometry. Portable4 uses the same broadcast for two four-lane weight vectors; both 6×8 bodies have 12 separate mul/add pairs and no K-loop spills/calls. Final fixed-width stores and borrowed tile iteration retain global channel hints; balanced six-row bands select at work>=2M. Independent oracles cover K1/7/17/513/1024, panel/channel/row tails, nonfused cancellation, guards, offset bands and 1/2/4/8 exactness. Retained full/variant/static gates pass; hardware/performance remain separate. |
| NHWC dense groups=1, packed `[oc/16][ic][tap][16]`; six-row tiles, Kc=512 | `nhwc/dense.rs`; grouped panel traversal and fixed partials, including Portable4 two paired 6×8 subtiles; **Present** | Bias and ascending `[ic][ky][kx]` order persist across K512 blocks. No retained 32-by-K im2row tensor or hot allocation. Independent bitwise tests cover K513/585/612, panel/row/channel tails, padding, dilation, multiple batches and guarded worker bands. Retained full/variant gates pass. |
| Dense interior/border: AVX/AVX-FMA/AVX2-FMA eight-lane and Portable4 paired bodies, kh*kw<=25 | `nhwc/dense/interior.rs`, checked descriptors and noinline tiles; **Present** | Interior spans and six bounded u32 border tables prevalidate loads. Border readers select a checked input pointer or static +0.0. Retained actual AVX/FMA and Portable4 descriptor K loops retain six broadcasts/twelve accumulators with no calls/spills/division/geometry. Scalar/AVX-512 and larger/general shapes keep their explicit fallback; paired Portable4 general geometry still has spills/divides and is not claimed register-only. Retained full/variant/static gates pass. |
| NHWC depthwise, packed `[c/32][tap][32]`; large-kernel strip at kh>=6 | `nhwc.rs` pixel tiles plus `nhwc/strip.rs` row-pass strips at kh>=6; equivalent shared 16-channel panels; **Present** | Fixed strips hold sixteen pixels and up to 32 channels on AVX. Tests compare every bit with independent per-pixel tap accumulation for kh=6/7/9, dilation/stride, bands and channel tails. AVX-512 deliberately reuses AVX2. |
| NHWC transpose 2x2/stride 2, four strided pointwise GEMMs; cout=1 special packing | Four shared panels for cout>=16 and `nhwc/transpose.rs` single-output spatial vectors; **Present** | Both paths apply epilogues at final stores and support bounded independent row bands. Single-output vectors preserve the existing bias-first/input-channel order. Other small channel counts retain a scalar fallback outside the active reference NHWC eligibility. AVX-512 uses AVX2 for this family. |
| Residual + activation in convolution store; ReLU, HardSwish, sigmoid, GELU where eligible | Borrowed ordered `StoreTransform` programs in `optimized.rs`, applied by NHWC and specialized NCHW convolution/transpose final stores; **Present** | `store_fused_steps` and family diagnostics report actual acceptance. Canonical scalar or rejected shapes keep the original fallback. Ordered Add/activation graphs are bit-exact in both layouts, with optional Conv bias, packed/unpacked paths, full/channel/scalar operands, extra consumers, tails and thread bands. Adjacent compound HardSwish/GELU patterns and nonadjacent single-node Add/ReLU/sigmoid groups are covered separately; arbitrary gapped compound patterns are not claimed. |
| Bound common ordered epilogues | `optimized/store.rs::PreparedStore`, stack-only ordered recipes; **Present** | Scalar/channel/full operands borrow disjoint arena slices and bind once per native call; arbitrary longer/multiple-activation programs retain the generic interpreter. Current common/generic policy, order, rebinding, channel-hint and tail goldens pass. Retained actual wrappers preserve arithmetic/register streams with fresh CFG/table bindings; selected full channel stores contain no arena lookup, channel divide, outlined SSE Add or CRT copy. No whole tile IntoIter copy remains. Debug boundaries retain ordinary-stack bounds; Release feature inlining remains intact. Retained full/variant/static gates pass. |
| Blocked NCHW <-> NHWC conversion and reference-eligible workers | `optimized/layout.rs`, `runtime/parallel.rs`, `simd.rs::transpose_matrix_columns`; **Present** | Volume>=262144 selects bounded plane/64 or channel/8 bands with aligned exclusive output slices. Recursive stack jobs allocate no temporary job Vec. Avx2Fma/Avx512 select an actually compiled AVX8 transpose child after feature guards; other policies retain blocked payload copies. Independent raw-bit/selector tests cover thresholds, both directions, 8-aligned bands, tails, batches and 1/2/4/8 exactness. Retained full/variant/static gates pass; native512 remains deferred. |
| NHWC max/average pool, nearest resize and repeated-row reuse | `nhwc/misc.rs`, channel vectors and bounded output-row workers; **Present** | A source row expands horizontally once within its worker band; repeated vertical rows copy that completed payload. Reuse stops at source-row/batch/band boundaries and uses existing output storage. Independent raw-payload tests cover signed zero/NaN/Inf/subnormals, factors1/2/3/4, channel/width tails, arbitrary partial bands, guards and 1/2/4/8 exactness. Pool oracles and retained full/variant gates pass; no measured reuse benefit is claimed. |
| NHWC spatial mean, binary-channel, BatchNorm | Backend-width spatial means/binary channels and reserved multiplier BatchNorm affine in `nhwc/misc.rs`; **Present** | Means preserve pixel summation order. BatchNorm uses separate subtraction/multiply/add, backend-width channel tails and reusable channel workspace; independent arithmetic and graph-layout tests are bit-exact. AVX-512 retains the AVX2 family. |
| NCHW spatial mean `Kernels/ReduceMean.cs` | `simd/tensor.rs::spatial_mean`, reference per-ISA paired accumulators; **Present** | Independent oracles preserve ISA lane/tail summation order. Only HW reductions use this path; trailing-axis LayerNorm moments remain sequential. Retained full/variant gates pass; native AVX-512 execution remains required. |
| NCHW maxpool, elementary math, maximum probability | `simd/tensor.rs` pooling and guarded unary/binary paths in `simd.rs`; **Present** | Pool edges and strided average tails compare bitwise to independent oracles. Arithmetic tests cover tails, NaN and signed zero. Retained model/forced-backend gates pass; AVX-512 hardware acceptance remains pending. |
| Erf / GELU vector approximation and explicit math policy | `simd/transcendental.rs::erf_with_fma` and bound `UnaryMath`; **Present** | Portable4/forced AVX use separate F32 polynomial operations; FMA policies use fused Horner updates. Optimized GELU uses the pinned inverse-sqrt2 multiply. Maintained exact-Fraction primitive/FMA oracles cover intervals, mixed lanes, special values, tails and stores without widening 0.001 graph tolerances. Actual current unary/common-store codegen and full/variant gates pass. |
| Optimized Sigmoid vector exp | `unary_vector_with_math`, pinned nonfused vector exp and negative-input `1 - positive`; **Present** | NaNs are sanitized before two bounds-proven unchecked truncations, then original payloads are restored. Width1/4/8/16 finite/clamp/Inf/NaN/zero goldens pass. Actual current ten standalone/inplace policy paths each use exactly two packed conversions, no scalar conversion/exp call, no FMA in exp and one packed divide; common stores forward the same bound math. Retained full/variant gates pass. |
| Last-axis Softmax: complete Portable4/native vectors and scalar tails | `simd.rs::softmax_native`, vector max/exp/lane sums and reciprocal multiply; **Present** | Portable4 selects four lanes at columns>=4; AVX uses eight at columns>=8; AVX-512 uses sixteen at >=16 or eight at axes8..15. Short axes and scalar tails retain their scalar contract. Independent pinned goldens and row-partition checks pass. Actual current Portable4 plus four native full-vector loops use packed conversion without scalar exp calls. Retained full/variant gates pass; hardware/performance remain separate. |
| Reference-eligible independent unary/elementwise workers | `simd/parallel.rs` and `runtime/parallel.rs::aligned_bands`; **Present** | Work>=2^18 and non-scalar policy select at most16 bounded workers with 16-float-aligned bands. Relu/Sigmoid/Erf/Sqrt/Gelu and equal-shaped binary/inplace calls preserve input order, self aliases and operand direction; other shapes keep their existing selectors. No hot job Vec is allocated. Threshold/limit/alias/tail tests and retained full/variant gates pass. |
| Fused LayerNorm, sequential statistics and SIMD affine stage | `simd.rs::layer_norm_affine`, guarded SIMD widths with sequential mean/variance retained; **Present** | Bit-exact inplace/out-of-place affine tests cover lane tails. Separate subtraction/division/multiply/add order remains unchanged. Real AVX-512 hardware gate remains pending. |
| Packed wide MatMul / REC projection | `simd.rs`, four-row baseline and eligible packed eight-row / AVX-512 sixteen-row tiles; **Present** | Wide tiles match disjoint four-row calls bitwise. AVX-512 requires packed rows>=8/%8=0, inner>=64 and columns>=1024; narrower matrix shapes use their selected pointwise/dynamic/fallback family. Hardware/performance gates remain pending. |
| Constant rank-two MatMul under Auto layout: shared RHS panel, 16<=N<1024, N%16=0 | `nhwc.rs::try_pointwise_matrix` with empty bias/IdentityStore; **Present** | Portable4 reports `nhwc_constant_matmul_6x8_pair2`; other eligible non-scalar policies retain the 6×16 family. Scalar/NCHW/dynamic/ineligible widths keep their existing routes. K starts at zero and a following bias Add stays separate. Exact selector, consumed-panel, worker and graph tests plus retained full/variant gates pass. |
| Dynamic row-major attention MatMul: unpacked RHS, rows>=4, columns>=32, native x64 policy | `simd/matrix.rs`, direct four-row / thirty-two-column kernel with K-order and bounded row/column tails; **Present** | `dynamic_matmul_4x32` is reported only under its exact selector; Scalar/Portable4 and smaller shapes retain the baseline. AVX-512 policy deliberately executes the AVX2/FMA body. Tests cover odd K, 31/32/33-column eligibility, row tails, missing-feature forcing, broadcast batch offsets and no panel allocation. Retained G3/variants pass. |
| Compact projection + ArgMax, packed coefficients with columns<=u32::MAX | `simd/argmax.rs`, up to128 rows sharing 256KiB weight panels, SIMD-resident lane winners; **Present** | Every non-scalar packed compact call uses the blocked route. AVX policies use four-row tiles plus one-row tails; eligible AVX-512 uses sixteen/eight rows before four/one tails. Strict per-lane updates and final lowest-column tie reduction preserve score bits and signed-zero ties. Tests compare 129 rows / K65 / 1041 columns against dense logits across panel/row boundaries and reject hidden nonfinite values. Public `ProjectionStatistics` keeps the actual `sequential`/`row_shards` scheduling family and ISA; these labels do not imply a separate unblocked kernel. All complete 16-column packed tiles, including Portable4, use zero-initialized K then bias Add. Incomplete columns retain scalar bias-first separate multiply/add; native unpacked calls use the scalar path. Complete-tile bounds are required before slicing weights, including column shards. The public bias-first MatMul contract stays unchanged. Two deterministic regressions failed before the fix; prior source `a0bcf050...` core/full/variant gates pass. Retained G3/variants pass; native AVX-512 and performance remain pending. |
| AVX-512 compact projection deterministic column sharding | `simd/columns.rs`, retained winner workspace and ordered shard merge; **Present** | Requires packed AVX-512, threads>1, unit rows>=4/%4=0, inner>=64, columns>=1024 and total work>=2M; reports `packed_column_shards`. Portable algorithm tests cover28 rows, ties across shards,1031-column tails,2/4/8 shards, bias, nonfinite failure/recovery and capacity reuse. Native entry rejects unavailable hardware; real AVX-512 execution remains pending. |
| Immutable shared coefficient panels and retained request/session storage | Versioned 16-column F32 panels and synchronized model-level cache in `packing.rs`; transactional reshape; **Present** | Concurrent compilation policies share identical Arc panels; failures are not published. `shared_cached_weight_bytes` separates model cache from per-compiled consumed panels. New formats remain lazy and eligibility-gated. |
| Observable physical kernel selection | Existing logical counts plus additive `specialized_kernels` family/ISA/store/count entries; **Present** | Shape selectors drive static diagnostics without per-run logging. Thread-aware NCHW selectors cover fallback policies; view-only concat inputs are regression-tested. Update selectors whenever implementation eligibility changes. |

## Pixels and active DB acceleration

| Reference family / active selector | Rust equivalent and status | Work required / focused evidence |
| --- | --- | --- |
| Fixed-point bilinear resize horizontal gathers (`PixelRow`): AVX2 / AVX-512 32-bit pixels, portable route | `pixel_kernels.rs::horizontal`, guarded SIMD gathers with exact fixed-point normalization; **Present** | All formats, padded strides, exact row ends and width tails compare to scalar pixel oracles. AVX-512 paths compile but need capable-hardware execution. |
| Perspective row mapping (`Warp.MapRow`): AVX-512 unrotated width>=8, AVX2>=4, portable double vectors | `pixel_kernels.rs::map_coordinates` used by `CropTransform`, guarded f64 vector mapping; **Present** | Independent scalar coordinate and crop tests preserve separate operations, rotation, row bands, pole/range rejection and tails. Real AVX-512 execution remains pending. |
| Cubic interior sampling: safe interior bounds, AVX2/AVX-512 | `pixel_kernels.rs::sample_interior` plus scalar clamped boundary fallback; **Present** | Ordered independent scalar samples and frozen-pixel crop tests cover format/stride/rotation boundaries. Pixel bytes remain exact; AVX-512 hardware acceptance is pending. |
| DB threshold `Threshold.Binarize`: strict `>`, SIMD finite accumulator | `pixel_kernels.rs::binarize`, guarded width variants consumed by DB; **Present** | Strict equality, negative zero, NaN/infinity, short tails and unsupported-policy forcing are tested. Workspace recovery tests and the retained source `a0bcf050...` all-model DB G3 gate pass; prior source `a0bcf050...` full/variant G3 gates pass. |
| DB interior mask (`ComputeInterior`): vector neighbor AND, bounded edges | `pixel_kernels.rs::interior`, reserved DB interior cache; **Present** | Neighbor-mask tests compare to scalar oracles for tiny dimensions, padded row ends and tails. Contour behavior still passes the pipeline regression suite. |
| DB foreground/background runs and row-extreme contour reduction | `pixel_kernels.rs::scan` accelerates forward/reverse runs; DB scanline geometry retained; **Present** | Bidirectional scans compare to iterator oracles. Candidate ordering, holes, dilation, unclip, score boundaries and reading order stay unchanged; Retained G3 passes; G4 remains required. |
| DB polygon score scanline acceleration | `db.rs::polygon_score` implements scanline interior enumeration; **Present** | Validate against C# fixtures on skewed/small/hole contours and exact threshold-boundary scores; preserve floating point boundary decisions. |

## ISA and feasibility

`KernelPolicy` / `Backend` expose Scalar, Portable4, Avx, AvxFma, Avx2Fma,
and Avx512. Auto chooses the highest eligible policy, then each operator retains
its shape-specific ISA guard. Forced unavailable policies return errors before
native execution. AVX uses separate multiply/add; FMA policies fuse eligible
accumulations. Avx512 requires AVX-512F plus AVX2/FMA for its reference fallback
families. NHWC depthwise/transpose/miscellaneous use 256-bit kernels on that
policy, and wide projection keeps the explicit shape gate. Host tests exercise
Scalar/Portable4/AVX/AVX-FMA/AVX2-FMA and reject unavailable AVX-512.

The Windows host reports `AMD Ryzen 9 5950X 16-Core Processor`, 12 exposed cores /
24 logical processors. A read-only PowerShell/.NET runtime query on 2026-10-15
returned AVX=True, AVX2=True, FMA=True, and AVX512F/BW/DQ/VBMI=False. The pinned
`nightly-2026-09-08-x86_64-pc-windows-msvc` toolchain is installed. This machine can
validate scalar/portable/AVX/AVX2+FMA execution, unsupported AVX-512 forcing, and
compile-time AVX-512 support; it cannot satisfy real AVX-512 execution acceptance.
An AVX-512-capable Windows runner is required for that exit gate.

Excluded reference code:

- `InferenceSession.cs` sets `EnableInt8VnniConv1x1=false`; INT8/VNNI experiment
  files do not create a parity requirement.
- `ConvDenseStride1.cs` deliberately bypasses its plain AVX-512 implementation.
  Keep the enabled AVX/AVX2 route and shape guards; merely present source files
  are not active optimization coverage.
- ARM64, WASM, GPU, and general-purpose ONNX operator expansion are outside this
  Windows x64 CPU inventory.

## Remaining integration and acceptance

1. Preserve source-family selectors, independent ordered arithmetic tests,
   indexed-store ownership and explicit fallback reporting.
2. Preserve current core, Release-library, 447 all-model, nine batching,
   unavailable-512 rejection, 1341 variants and Debug/scoped Release proofs.
   Package qualification is determined by
   `target/core-validation-canonical-stores-20261016/package-report.json`:
   it must bind the observed successful Cargo command, exact current source,
   shipped-member hashes and crate SHA. Missing, mismatched or historical reports
   do not qualify these bytes. Historical 54a1 evidence stays bound to its producer.
3. Execute AVX-512 on capable Windows hardware when the user resumes that
   deferred gate. Unsupported-policy errors and static proof do not replace it.
4. Preserve the complete failed 54a1 and earlier 65/72 timing reports. The original
   sample's five-round allowance is exhausted; no further performance run is
   authorized. Untimed codegen improvements are not measured speedups.
