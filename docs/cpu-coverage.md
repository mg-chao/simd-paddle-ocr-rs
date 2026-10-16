# Windows x64 CPU optimization coverage

This inventory compares active Windows x64 source with pinned C#
`33ff8a81672452041d110e8ba52dcd2639e54228`.
The active source is
`d4a381343691afd097fdaa436cb9306f522d71e996dccd4256aee86bf594adf5`
(83 files). **30 focused NCHW tests pass**, including literal spatial/rounding
contracts, actual recursive child dispatch, Portable4 dual tiles and simulated
ISA plans. **Attempt4 core, Release-library, seven-model, variants, batching,
unavailable-512 rejection and selected Release static gates passed for this
source.** The 131-member saved preperformance package also passed; the final
release package remains dependent on the final documentation and actual round
record. The scoped Debug fixed/logical-chain bound is 1,776,512 bytes under the
unchanged 2 MiB threshold, 11 logical visit levels and a 256 KiB outer entry
allowance. That allowance is an assumption, and physical Rayon helping/reentrant
depth is not globally bounded by this proof. Bounded-worker/concurrency core
regressions passed separately. The initial 2,162,480-byte overinclusive sum and
the exhaustive 632-frame posterior grouping are retained. Actual capable-Windows
AVX-512 remains user-deferred.

The exact source has passed **273/265 library tests (288/280 all-target
checks)** with all features and without default features respectively, six
documentation tests per mode, both all-target Clippy checks with warnings denied,
**113 tooling contracts**, and **273 Release-library tests**. Fresh model gates
passed seven exact imports and **447 comparisons**, **1,341 cases across nine
policy/layout/thread configurations**, nine real-batch comparisons, and explicit
unavailable-AVX-512 rejection. Current core and regression completion receipts
bind the actual logs, source snapshot and closed producing executable/PDB pairs.
These results establish correctness on available hardware, separately from speed
and capable-Windows AVX-512 acceptance.

The current Release static gate is closed by
`target/cpu-active-coverage-attempt4-production-codegen-20261016/complete.json`
(SHA-256 `33aa811584ef731d45dd22e3273d9e5d74f1011924bb9eac7844de0a78df695f`),
with 348 hash-bound evidence files. It binds the actual current executable/PDB,
source, native and Portable4 reductions, common-store/transpose bindings,
17 selected NCHW numerical tiles and 20 selected Gelu/store control-flow paths.
These are selected path proofs: canonical packed paths retain documented spills,
bounds checks and coefficient-tail helper overhead. They do not establish that
every kernel is register-only, all epilogue paths are call-free, or AVX-512 ran.

Prior source `7f341...` failed the strict Auto two-thread trace at medium DET
`Conv.121`, despite an exact final output. Its old Packed3 four-position FMA
envelope disagreed with pinned C# scalar borders/tails. The corrected dedicated
eight-output family uses complete spatial 4/8/16-lane packets on interior rows
and separate scalar multiply/add elsewhere. The literal regression has actual
RED and after-run GREEN evidence; the failure and its superseded passes remain
historical. Earlier 1397 column-shard bounds failures are retained separately.

Source `f7bc2ce6761f4521e9492468fe6c201b6a96e5526b5963250d878ee898c2242c`
and `canonical-stores-20261016` evidence are **historical**: 267/261 all-target,
252 Release-library, 6 docs per mode, 86 tool contracts, Clippy both modes, 7 exact
imports/447 comparisons, 1341 variants, 9 batch cases, unavailable-512 rejection,
scoped Debug/Release and 117-member package gates passed then. Its historical
Debug bound was 1,946,832 bytes. It proves selected canonical SIMD4 stores,
plane-one SIMD8 stores and isolated PadEnd2 reductions under the named audit
conditions. That historical generic PackedStride2 body retained eight vector
stack operands and bounds checks; these findings do not describe its dedicated
replacement. Generic PackedPointwise retains coefficient-tail/helper/spill overhead, and
general >25-tap Portable4 geometry retains its explicit fallback. Scoped
proofs do not establish universal register-only loops or current-source gates.
The active D4 full/variant gates now pass independently. Untagged retained
per-family static findings below retain their original historical scope; only
explicitly current producer-bound reviews qualify current compiled paths.

The last measured historical 54a1 source passed 36/72 performance targets;
prior a0bc passed 65/72. Neither qualifies current performance. The first five
original-sample rounds remain consumed. Grant
`42014eec-5933-488f-8879-41dc67a0da8b` raises the cap to six without rewriting
prior entries. The sixth full 72-case round completed **37/72 targets** (18/36 latency,
19/36 throughput), leaving M5 unmet. Output comparisons and same-kernel thread
checks passed, with zero Rust retained-storage growth in measured repeats.
The complete nonpassing report and separate phase evidence are retained at
`target/cpu-acceptance-active-coverage-attempt4-20261016`; completion receipt
SHA-256 is `07054326e6aca8d3e792a7ea8969913c399bcfdcfa51a338fecb943f1de5e9d0`.
All six authorized rounds are consumed; no further round is authorized.
No untimed kernel review is a measured speedup.

Both named Windows checks passed for current published D4 head `fe639067...`
in draft PR #2. Main requires both from Actions app 15368 with strict/admin
enforcement independently verified. A later final documentation tree requires
its own matching publication and CI receipts. Actual AVX-512 execution remains
explicitly user-deferred; full hardware/performance acceptance is incomplete.

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
   `AGENTS.md` and the recorded user exceptions: five rounds are consumed, and
   the sixth full round completed 37/72 targets under fixed parameters and M5
   remains unmet. All six rounds are consumed; no further round is authorized.

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
| Packed dense3x3: same geometry, cout%8=0; AVX-512 deliberate single-thread large-cout fallback | `packed_three_eight`, eight output chains sharing complete spatial packets; **Present** | Pinned Portable4/AVX/AVX-512 spatial widths are4/8/16. Only interior rows and complete packets use vector accumulation; scalar borders/tails preserve separate multiply/add. `packed3_band_plan` uses the pinned8M work gate and floor OC8 worker bands. Eligible AVX/FMA/AVX2 children reselect paired16 for count>=64/%16; Portable4 and AVX-512 workers retain OC8. Literal callback-order/dyadic tests distinguish actual child dispatch, including128/2 and mixed272/4 bands with an upper-half shared panel. Current integrated447/variant1341 gates and scoped Release reviews pass. |
| Native paired packed 3x3: group1, dilation1, stride1, pad1, same spatial size, packed cout%8 prerequisite | `nchw.rs::packed_three_pair16`, reported `nchw_packed_3x3_pair16`; **Present** | Whole-call native selection requires threads==1, cout>=64/%16; eligible recursive AVX/FMA/AVX2 worker children also select it. Aggregate diagnostics distinguish `nchw_packed_3x3_parallel_pair16` and `nchw_packed_3x3_parallel_mixed16_8`. Sixteen bias-first chains share safe spatial octets in ci/ky/kx order; scalar borders/tails and disjoint global stores remain exact. Shared low/high OC8 views support child starts at co%16==8. Portable4 workers stay OC8; whole-call AVX-512 retains its selected AVX2/FMA fallback. Independent geometry, callback order, panel and planner tests cover these routes. |
| General dense stride1: group1, dilation1, K<=2^20, plain AVX horizontal interior>=8 | Dynamic-tap `spatial`,8/4/1-output reuse with safe narrow fallback; **Present** | Source-eligible AVX2/FMA cout%8<=64 projections split disjoint row bands to retain eight output accumulators. Independent row-band and global-index epilogue tests are bit exact. |
| Packed dense stride1: AVX2+FMA, cin>=16, cout%8=0,5x5/7x7, interior>=4, nonempty vertical interior, work>=4M | Shared-panel `packed_dense`, shape/work/worker gates; **Present** | Selector tests exercise minimum work, channel and worker limits; independent numerical tests cover boundaries and coefficient order. AVX-512 uses the reference AVX2 body. |
| Dense2x2 stride1, pad-bottom/right1, pad-top/left0 | Const2x2 `spatial_kind`,8/4/1-output reuse; **Present** | The source cout8/work>=4M/height>=2*workers row split uses disjoint scoped slices. Tests retain last row/column contributions and exact global epilogue indices. |
| Unpacked dense3x3 stride2: group1, dilation1, pad1 | Fixed strided `spatial_kind`; **Present** | The existing unpacked family retains its source geometry, accumulation and worker rules. The Portable4 cout16/threads>2 fork continues to select it. This route is separate from the packed footprint and child-selection corrections below. |
| Packed dense3x3 stride2 OC8: group1, dilation1, pad1, cout>=8/%8 | `nchw/stride2.rs::execute`/`eight`, reported `nchw_packed_3x3_stride2`; **Present** | Spatial lanes are4/8/16 for Portable4/AVX/AVX-512. Vector packets require oy>0,2oy+1<inputHeight, ox>0,ox+N<=outputWidth and2(ox+N-1)+1<inputWidth; scalar borders/remainders retain separate multiply/add. Two row-bounded loads plus an even-lane swizzle safely handle the final selected input value without reading an unused odd value beyond the row. Literal cancellation and exact-store tests cover six heights and odd/even thresholds10/17/18/33/34. AVX-512 keeps OC8 and its native sixteen-spatial policy. Current scoped Release and model gates pass; historical generic PackedStride2 static findings do not describe this replacement. |
| Native paired packed3x3 stride2: eligible AVX/AVX-FMA/AVX2-FMA cout%16, including tiny shapes | `nchw/stride2.rs::sixteen`, reported `nchw_packed_3x3_stride2_pair16`; **Present** | Direct nonsharded calls select sixteen output chains with no64-channel minimum. For batch1/threads>1 and at least two OC8 blocks, `packed_stride2_band_plan` floors bounded OC8 bands with no work threshold; native child count%16 selects OC16, otherwise OC8. Actual tiny32/2 and mixed80/4 callback traces prove sharing/dispatch independently of equal numerical outputs; upper-half shared panels, nonzero batches, guards and indexed stores are exact. Aggregate names are `nchw_packed_3x3_stride2_parallel_pair16` and `nchw_packed_3x3_stride2_parallel_mixed16_8`. Portable4/AVX-512 children remain OC8. |
| Portable4 packed3x3 stride2 dual spatial tiles: complete adjacent4+4 interior packets | `nchw/stride2.rs::dual4`; **Present** | Preferred dual packets share each loaded coefficient across two adjacent four-spatial values for eight output chains, then single4 packets and scalar tails. Source-order callback stamps prove the two packets for one channel are stored before the next channel and before the next tile. Direct dyadic input lookup, global epilogue indices and one-store counters cover full ownership. This source specialization makes no measured-speed or universal register-only claim. |
| Depthwise3x3/5x5: unit stride/dilation, symmetric same padding | Distinct output-vector and tap-major bodies; **Present** | Output-vector guards match source:3x3 batch1/height>=50/width>=10;5x5 batch1/height, width>=10, with8M channel-shard override. Tap-major final-valid-tap fusion needs no extra epilogue pass. Tiny1-pixel maps,49/50 and9/10 thresholds, edge arithmetic, ordered FMA regions and8M sharded5x5 are independently tested. |
| Depthwise7x7 unit stride, pad3 | Const7x7 output-vector strip with full scalar boundary rows and8M channel gate; **Present** | Independent per-tap oracle covers narrow widths, row boundaries, batches and SIMD tails. |
| Depthwise9x9 unit stride, pad4; packed and fixed/general channel paths | `packed_depth9` and four-position/channel-vector scoped row body; fixed9 strip and generic fallback; **Present** | Packed guards retain channels%8, dimensions>=9, plane<=4096, work>=1M. Source row ownership covers batches and low channel counts. All packed edges use FMA, with skipped taps bounds checked. Fixed AVX9 eligibility and forced-ISA guards are explicit. |
| Depthwise1x5;3x3/5x5 height-stride2;3x3 stride2 | Const-specialized tap-major kernels with safe contiguous/strided loads; **Present** | Source per-tap vector-end conditions, scalar tails and final valid-tap epilogues are tested across odd dimensions and output shapes. |
| General depthwise stride1, dilation1, arbitrary kernels | Dynamic-tap reusable row strips with nonempty-interior guard and source4M channel ownership; **Present** | Independent shape/edge/tail tests include1x7,9x9 fallback, batches and per-channel bands. Retained G3 passes; G4 remains required. |
|2x2 stride2 transposed convolution | `transpose_tile`,8/4/1-output reuse, four ordered tap accumulators and source8M channel ownership; **Present** | Bias-first, batches, widths3/4/7/8/9/17 and outputs1/4/8/9/16 match an independent oracle. Indexed stores cover every interleaved tap exactly once. |
| Canonical residual/activation epilogues in final stores | `try_conv_store`/`try_transpose_store` use borrowed `StoreTransform`; **Present** | Completed packed spatial quartets and plane-one channel panels transform contiguous SIMD vectors before their first store; borders, tails and remaining interleaved outputs retain absolute scalar indices. Reduction blocks transform only on their final K block. Independent atomic coverage tests require exact eligible vector packets and one transform per output; fixed policy goldens cover scalar/channel/full residuals and activation order. |

`runtime/nchw/tests.rs`, `runtime/nchw/audit_pair16.rs`, the independent
`audit_packed3_*`/`audit_stride2_*` children and the additional
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
| Observable physical kernel selection | Existing logical counts plus additive `specialized_kernels` family/ISA/store/count entries; **Present** | Shape selectors drive static diagnostics without per-run logging. Shared packed-channel plans drive both actual worker execution and aggregate paired16/mixed16_8 reporting; context capacity bounds worker counts. Thread-aware NCHW selectors cover fallback policies, and view-only concat inputs are regression-tested. Update selectors whenever implementation eligibility changes. |

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
- The declared sixteen-output AVX-512 packed3x3 and packed3x3-stride2 helpers
  have no active callsites. Their reachable AVX-512 paths use eight output
  chains; native sixteen-spatial coverage is a distinct requirement.
- ARM64, WASM, GPU, and general-purpose ONNX operator expansion are outside this
  Windows x64 CPU inventory.

## Remaining integration and acceptance

1. Preserve source-family selectors, independent ordered arithmetic tests,
   indexed-store ownership and explicit fallback reporting.
2. Preserve the passed current-source core, Release-library, 447 all-model,
   nine batching, unavailable-512 rejection, 1341 variants and scoped Debug/Release
   proof identities. Preserve the separate 131-member preperformance package.
   Final package qualification is determined only by the actual current-source
   `target/core-validation-active-coverage-attempt4-20261016/package-report.json`
   after it exists:
   it must bind the observed successful Cargo command, exact current source,
   shipped-member hashes and crate SHA. Missing, mismatched or historical reports
   do not qualify these bytes. Historical 54a1 evidence stays bound to its producer.
3. Execute AVX-512 on capable Windows hardware when the user resumes that
   deferred gate. Unsupported-policy errors and static proof do not replace it.
4. Preserve the complete failed 54a1 and earlier 65/72 timing reports. The original
   sample has consumed all six authorized rounds. The sixth completed 37/72
   targets under the fixed policy; retain all timing/phase evidence and failures.
   No further round is authorized; untimed codegen improvements are not speedups.
