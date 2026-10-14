# All-model scalar validation

The retained scalar kernel passed expanded native REC and full OCR parity
for tiny, small and medium on Windows and a native Linux container. Both proofs
pin the same Rust source snapshot. Its committed production code passed actual
cold and warm hosted CI on both operating systems at `c1e92b2`. A later public API
gate also passed all three families on native Windows and Linux. The retained
scalar changes passed the complete public tuning screen. Official C# comparisons
and remote CI for the additive public API/workflow/tooling checkpoint remain
**pending**.

## Frozen native proof identity

The [Windows expanded proof](../target/validation/all-model-windows-expanded-v3/proof.json)
and [Linux expanded proof](../target/validation/all-model-linux-expanded-v2/proof.json)
both report `all_native_rec_ocr_gates_passed`, `scope: expanded` and
`source_changed_during_validation: false`. Each ran all six REC/OCR executions and
all six comparisons. Reference manifests and executable hashes were unchanged at
completion.

| Platform | Completed (UTC) | CLI executable SHA-256 |
|---|---|---|
| Windows 11 / AMD64 | 2026-10-14 17:21:24 | `155b0d4af9e870b1c8384121f32a48d449018b579992326bed0320f32fff3e8b` |
| Ubuntu 26.04 native container / x86_64 | 2026-10-14 18:05:02 | `f1a1e5b024df12b7f6a212cf4796ee04bb9e27b9a68248deeebea86c9e35dd0d` |

The Linux proof uses Ubuntu image
`sha256:f144425ff09be612d6d9ad965196e9cdc23dae1f42110a8a11a3e9a8198759f7`,
glibc 2.43 and the native x86_64 WSL2 kernel 6.18.33.2. It reuses the frozen
glibc-2.43 release CLI; no CPU emulation is involved. This container proof is
distinct from the earlier SnowShot WSL baseline and interrupted full-E attempt,
which remain preserved separately.

Both proofs contain the same 48-file source hash map covering Rust source, tests,
Cargo manifests and the toolchain declaration. Its canonical JSON SHA-256 is
`6dba9f1e3887f929f02d2b31b3e24ac2ba4866c477c74b5c42a65d002aa6d4b1`
(UTF-8, sorted keys, separators `,` and `:`, no trailing newline). The per-file
digests in each proof are the authoritative source identities.

## Expanded coverage and strict comparisons

Both operating systems checked the following counts, with `first_difference: null`
in every comparison. REC uses the 32 frozen native lines across the original
sample and its rotation. Full OCR additionally checks the blank and one-pixel
cases from source pixels through public results.

| Family | REC cases / required lines | REC artifacts | Full OCR cases | Full OCR artifacts |
|---|---:|---:|---:|---:|
| tiny | 2 / 32 | 5,248 | 4 | 9,761 |
| small | 2 / 32 | 10,816 | 4 | 15,329 |
| medium | 2 / 32 | 11,424 | 4 | 15,881 |

The comparators validate every declared model, artifact and auxiliary file before
semantic comparison, reject missing coverage, and compare all captured stages.
Tensor comparisons retain `atol=1e-5`, `rtol=1e-4`; shapes, text, token counts,
rotations and CTC boundaries compare exactly. Public compact scores retain their
raw projection-max contract separately from dense Softmax probabilities.

| Family | Expanded reference manifest SHA-256 |
|---|---|
| tiny | `acd4c21cc9f251bf1d25afe88de4e3f6bcdb27154d7d504722761b5c03379267` |
| small | `3eaa62682540105b64b8835c4391c583b2ca02d0155bc3e812d84b36c1084e08` |
| medium | `9b73c13a837e3d101dbd814dc2d6ac646b98d013055161a70af53989af05cd27` |

## Tests, import contracts and native operators

The kernel candidate's ordinary all-feature test runs passed **157 tests on Windows** and
**154 on Linux**, with zero failures and 16 ignored archive-dependent tests on
each platform. The expanded CLI proofs above are separate executed gates, rather
than inferred success from ignored tests. All-target/all-feature Clippy passed
with `-D warnings`. An earlier full Windows Python tooling suite passed **71 tests**;
source hashes matched before and after the run. Its matching Linux tooling ran
71 tests: **65 passed** and six explicit Windows-only process-contract tests were
skipped. The refreshed Windows gate is recorded in
`target/validation/windows-tooling-final-v3/python-tools.record.json`; refreshed
Linux tooling evidence is in `target/validation/all-model-linux-final-checks-v3`.

Linux Rust tests and Clippy used the native Rust 1.97.1 Debian bookworm image
`sha256:2775a09d208ff0d7c1f50490c45b62db929e87ba1dcbc3f2132ac71a704bcdd3`
with glibc 2.36, a separate ABI from the Ubuntu 26.04 parity executor. A writable
private copy matched all 48 Rust source files and all 541 copied fixture files
(8,200,563 bytes). Earlier read-only-workspace and reused-manifest-path failures
are retained as environment attempts; the final fresh private build completed
the entire suite. Final Linux check logs, identities and results are preserved in
`target/validation/all-model-linux-final-checks-v2`. Historical pre-kernel
Windows/Linux test counts were 155/152 and remain in their original evidence.

The latest stable prepared-launcher/helper/tool-test snapshot passed **98 tests
on Windows**, with no skips or failures. Native Linux discovered and ran **98
tests: 80 passed, 18 explicitly Windows-only tests skipped**, with zero failures.
The skipped cases comprise 12 prepared benchmark launcher contracts and six
PowerShell process-runner contracts; every test name and reason is retained.
Both platforms verified an unchanged, identical 42-file tool inventory:
`7e0a555edb67450a4ded1385a7476b9f3456a6416333d17223793b13c31b5502`.
Its fingerprint hashes UTF-8 ordered `path + NUL + sha256 + LF` entries; file byte
counts are independently checked. Native Linux uses Python 3.14.4 in the native
Ubuntu 26.04/glibc-2.43 container; Windows uses Python 3.12.10.

The latest raw records are in `target/validation/final-tooling-v4` (Windows) and
`target/validation/all-model-linux-final-tooling-v4` (Linux). Linux's successful
unittest process exited zero and reported `Ran 98 / OK (skipped=18)`. A prior
fingerprint-normalization preflight ran no tests, and a later recording wrapper
rejected a multiline verbose status after the successful test process. Both
recording attempts are preserved separately; strict postprocessing reconciles
79 inline `ok` statuses, one standalone `ok` and 18 skips without rerunning or
changing test assertions.

The separate [final additive tooling archive](../../reference-archives/evidence/all-model-final-tooling-additive-20261015-v1.zip)
retains both platforms' raw records, before/after inventories, the 42 tool source
files, native environment and skip reasons. Its 70 entries passed CRC and
retained-file byte/SHA readback verification: 200,908 bytes, SHA-256
`55dbd7ead584753cef7e56268cc52d6f6fba91312f32b8e773625ba8eeb17018`.

The explicitly executed import gate matched all seven shipped networks: shared
`cls`, tiny `det`/`rec`, `small_det`/`small_rec`, and `medium_det`/`medium_rec`.
The compact hash-contract archive and original physical-weight archive both
passed on Windows; compact import contracts also passed in both remote OS jobs.
These contracts check exact raw/normalized graph structure, attributes, integer
shape metadata and normalized constant byte counts/SHA-256 hashes. The original
archive additionally validates the declared physical normalized-weight payloads.

[Independent native scalar operator goldens](../tests/fixtures/recognition-scalar-v2/README.md)
preserve 17 C# captures: 15 executed comparisons and two non-unit Slice rejection
cases. They cover Sub/Div, native Pow(2)/Sqrt, trailing-axis mean, activation-RHS
attention MatMul and Softmax, broadcast MatMul, rank-five QKV Slice/Squeeze,
complete unfused LayerNorm including near-constant cancellation, and retained
integer shape programs at two widths. Exact-bit cases and numerical-tolerance
cases retain their declared contracts; native `MathF.Pow(x, 2)` is not claimed
bitwise equal to Rust multiplication.

Other native checks retain separate framework counts: 15 C# public-semantic
selftests, 46 REC-form selftests, 16 replay/export selftests and six Windows
process-contract tests passed. Six native C# public semantic preflight groups
(three families × canonical/CPU profiles) passed both REC and OCR on the first
call and after 30 warm calls. Their raw records are in
`target/validation/native-public-semantic-smoke-v1` and retain their original
tool-source identity. A later additive preflight fix verifies all model roles,
auxiliary files and every case artifact before filtering selected cases. Its
`PublicBenchmarkValidation.cs` SHA-256 is
`fdde7bd36e10161cf1e3abb8ddffc82bcbfad47b2c42233a228edf5036979f02`.
The refreshed native CPU preflight passed, and a corrupted unselected artifact
was rejected. The refreshed 37-file Linux tool hash map is
`42a7baa0b63a33ddb7ce460dc495141e0e72fe0861a24a5b47b259b501fbdeae`;
hashes remained unchanged during the refreshed 71-test tooling gate. No Rust or
core-math source changed, so the frozen native proofs and Rust checks retain
their existing scope. These checks establish public semantic behavior;
synthetic runner smokes are not accepted latency measurements.

## Proven remote CI

The retained production code at
`c1e92b2444a286c076c3f15cae0eb9ecd544e9be` passed the
[cold-cache dispatch 38624701713](https://github.com/mg-chao/simd-paddle-ocr-rs/actions/runs/38624701713)
and [warm-cache dispatch 38626245544](https://github.com/mg-chao/simd-paddle-ocr-rs/actions/runs/38626245544)
on Windows and Linux. Both cold jobs skipped cache restoration. Both warm logs
record actual OS-specific cache hits ending in
`1713d3abb3e1e788babff950fb07de87de9f54deeed93a92bf174625c6318473`,
followed by verified acquisition. All 36 instrumented reports use three warmups
and three samples, and every row has all four allocation/growth counters equal
to zero. All 12 mixed-width reports contain exactly
`32,64,96,160,224,320,480,576,992,32,992`. The kernel source SHA-256 is
`19c70e511b7c22ae93578a844a4a472f7f4afcd89c76e590d13fba6ab2ca3359`.

The separate [retained-production CI archive](../../reference-archives/evidence/all-model-hosted-ci-final-production-20261015-v1.zip)
is 283,795 bytes with SHA-256
`c96e91df5c68c530e68094373bccd203d9c6d474942e22a8d02862e83cdf1521`.
Its downloaded metadata/log/report identities and ZIP CRC passed readback
verification. It certifies `c1e92b2`; it does not claim that the later test and
workflow additions have already run remotely.

The all-model baseline at commit
`51840089891e30e67b622771e64e07f57c6c75af` passed the
[cold-cache dispatch 38608698106](https://github.com/mg-chao/simd-paddle-ocr-rs/actions/runs/38608698106)
and [warm-cache dispatch 38611243863](https://github.com/mg-chao/simd-paddle-ocr-rs/actions/runs/38611243863)
on Windows and Linux. Both warm jobs explicitly restored their OS-specific cache
key ending in
`1713d3abb3e1e788babff950fb07de87de9f54deeed93a92bf174625c6318473`
and then verified the assets. Each OS/run retained nine instrumented reports
(three families × three modes), giving 36 reports across both runs. Every warmed
allocation/growth counter was zero with three warmups and three samples. These
results retain their earlier baseline scope; the retained-production checkpoint
above separately certifies the kernel.

The [all-model hosted CI evidence archive](../../reference-archives/evidence/all-model-hosted-ci-correct-20261015-v1.zip)
is 282,068 bytes with SHA-256
`b6b89a5a7f8b2aae2955eeb0848cd105baa51d4071c06e1ecb0969734d41c030`.

The [cold-cache run 38587687253](https://github.com/mg-chao/simd-paddle-ocr-rs/actions/runs/38587687253)
and [warm-cache run 38592128880](https://github.com/mg-chao/simd-paddle-ocr-rs/actions/runs/38592128880)
both completed successfully on Windows and Linux at commit
`97757677048bf17ef93c876536746e31518178aa`. The cold run bypassed cache restoration.
The warm run's downloaded job logs explicitly record cache hits and successful
restoration on **both** operating systems, followed by verified asset acquisition.
These earlier runs establish tiny hosted-fixture CI at their original commit.

For each OS, the archived instrumented tiny reports for `rec_public_borrowed`,
`ocr_borrowed` and `mixed_widths` use three warmups and three samples. Every input
row records zero for all four counters: `warm_allocation_count`,
`warm_allocation_bytes`, `warm_retained_heap_bytes` and
`warm_peak_heap_growth_bytes`. The REC and OCR reports each cover four input rows;
the mixed-width report covers its full sequence. CI timings are observations,
without latency acceptance thresholds.

The preserved [tiny hosted CI evidence archive](../../reference-archives/evidence/tiny-hosted-ci-20261015-v1.zip)
is 180,948 bytes with SHA-256
`cc0ed1b2a5d0adc9653763276989c1157caa72ca22843a0c9508f9fd0cc30918`.
It retains the remote run identities, downloaded logs and allocation evidence.

## Public API reuse and recovery

The additional ignored test
[`all_shipped_families_public_api_reuse_and_recovery`](../tests/real_public_api.rs)
actually ran on native Windows and native Linux: one test passed on each, with
explicit completion markers for tiny, small and medium. Windows formatting and
both platforms' all-feature Clippy checks for the new test passed. The two runs
pin the same 49-file Rust/Cargo snapshot, adding only this test to the original
48-file proof snapshot; every original per-file digest remains unchanged. The
API source-map SHA-256 is
`ea0c5e395e3d5ff6ffa33ffa0eed7d183779586389949ab0c54d5a14396592f8`
(sorted-key Python JSON with its default separators). This is a later selected
integration gate, separate from the ordinary-suite counts above.

| Check | Evidence scope for every family |
|---|---|
| Existing exported crops/pages, rotations, blank and tiny pages | REC/full OCR public JSON compares with the native compact oracle |
| BGR24, RGB24, BGRA32 and RGBA32 | Padded-stride and varying-alpha inputs preserve the native-compared public output |
| Exact 11-width sequence and revisits | Actual shipped models; reused public results match independently fresh Rust engines |
| Fixed320 wide/narrow/wide | Actual shipped models; width is capped at 320 and the whole source line is resized |
| Changed artificial pixels at identical dimensions | Reused output matches a fresh engine, followed by successful restoration of the original input |
| Alignment true/false/true | Text, scores and other public fields survive capture changes; disabled spans are absent |
| Invalid dimensions/stride | Checked image/width constructors reject them; the next public inference remains correct |
| REC activation/image/result budgets and OCR detector/crop/total-result budgets | Every intended failure must occur, followed by valid public inference or native-compared empty-page recovery |

Native preprocessing remains independently covered by the ordinary 56-capture
bit-exact test, including four formats and fixed-width saturation. Arbitrary
synthetic widths and fixed320 inference are Rust regressions, not new native
numerical oracles. In the expanded native corpus, small reaches width 864;
small's width-992 execution is established by the allocation and public API
regression gates. The original and rotated pages also exercise different bytes
at identical dimensions through one reused full OCR engine in each native proof.

No tracing or dense logits are used by the new gate, and diagnostic storage stays
zero. Observer-failure injection remains separately covered by the synthetic
REC/OCR regressions and medium's real DET/CLS test. Linux uses the existing native
Rust 1.97.1 Debian/glibc-2.36 container, with a fresh private source/target and only
the compact public-test read set. Every copied model, dictionary, source/crop
pixel file, metadata and public JSON matched its descriptor byte count/SHA-256;
whole-bundle integrity was independently checked by hosted acquisition/parity.
Expanded tensor payloads were not copied for this test.

The native test executable SHA-256 values are
`a631de012a35e9e89d03a6c2c89253cd0b13f743e14aac76dccbffad21db47d2`
(Windows) and
`eda7bb8513654d7ba9f1b7be369bb872703e24a99b1c054de74e3360664e26b6`
(Linux). Source, executables, raw command/stdout/stderr records and fixture-read
identities are preserved in the separate
[native public API archive](../../reference-archives/evidence/all-model-public-api-native-20261015-v1.zip):
2,056,437 bytes, SHA-256
`94dc094eee6779c7d6dd71cd8dd20ebc58e70a912c4a86621d877cd837cb637d`.
All 146 entries passed CRC and retained-file byte/SHA readback verification.

## Compact CI versus preserved expanded references

The [fixture index and acquisition contract](../tests/fixtures/step12/README.md)
select one complete four-case hosted bundle per family for both REC and OCR.
Tiny remains on `fixtures-v1`; small and medium use `fixtures-v2`. Compact counts
are 2,092, 2,788 and 2,808 artifacts respectively. Archive and model byte counts
and SHA-256 hashes are checked for supplied files, cold downloads and cache reuse.
The additive workflow guards list only ignored tests before exact execution,
including import, DET/CLS, line-pipeline, public API and REC/OCR corruption gates.
Missing names or a lost ignore marker reject the gate. These guard additions
await remote CI at their later commit. All-family allocation gates are configured
in the Windows/Linux workflow.

Expanded original-page references remain separate immutable local archives:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `tiny-ocr-allnodes-v1.zip` | 2,500,108,049 | `a3355def46d050f8b25357dd3ef3cfbc88d3c60912f48f8b94491b28e020ec41` |
| `small-scalar-ocr-expanded-v1.zip` | 6,672,006,502 | `ce9be678f2584b60bd33a2d793d93144f9ee670c6696468bab72f2bac485805d` |
| `medium-scalar-ocr-expanded-v1.zip` | 14,838,005,009 | `6648d9a53569a8f2725949502b55f5c217714615217c9fbecbf13fd175f7c86f` |

They preserve the wider native numerical corpus used by the expanded proofs.
Synthetic mixed-width allocation coverage does not substitute for native parity
at those widths. Compact and expanded proof identities are preserved separately;
large candidate buffers are excluded from the small proof/log archives.

The separate final native proof archives retain raw invocations/comparisons,
the frozen executable, all 48 source files, all six candidate manifests and all
three native manifests. Generated dense tensor payloads remain local; their
descriptor byte counts and SHA-256 hashes remain in the archived candidate
manifests. Native oracle payloads remain in the immutable expanded archives.

| Evidence archive | Bytes | SHA-256 |
|---|---:|---|
| [Windows expanded final](../../reference-archives/evidence/all-model-windows-expanded-final-20261015-v1.zip) | 6,888,131 | `4a16ba914b712c4c4db3ca4e5a029cbc133a471c32743f336b2dfe645e427380` |
| [Linux container expanded final](../../reference-archives/evidence/all-model-linux-container-expanded-final-20261015-v1.zip) | 6,829,726 | `ae269f18d3050dd189c49d82466d83d794ac11c921936d7a6d40a8b6be48a389` |
| [Linux final Rust/tooling checks](../../reference-archives/evidence/all-model-linux-final-checks-20261015-v1.zip) | 41,065 | `725c9aff973f12727aed88cf5e3f9c28a4378b737ae3e0ddcac8b564e18a6b52` |
| [Linux tooling refresh after preflight fix](../../reference-archives/evidence/all-model-linux-tooling-refresh-20261015-v1.zip) | 8,282 | `3f32810ca455d290b785e371e39a13a6a44adc18b9b6df29330c396a4a6e74c5` |

Proof, final-check and hosted-CI archives retain separate identities and passed
CRC and SHA-256 readback verification.

## Scalar tuning selection

The public tuning screen ran 72 fresh processes across all three families and
borrowed/owned REC/OCR scopes: three alternating baseline/candidate pairs per
scope, ten warmups and twenty samples, with common affinity mask `1`. Every pair
improved at p50 and p95. The p50 speedup ranges across each family's four public
scopes were 1.54–1.68× for tiny, 1.84–1.97× for small and 2.42–2.70× for medium.
These are exploratory selection measurements, not the official six-process,
30-warmup, 100-sample C# comparison protocol. All process records and observed
variation remain preserved, including the small OCR baseline tail spike.

The retained changes use contiguous trailing-axis means, row-scalar LayerNorm
broadcasting and a scalar four-channel/four-pixel Conv1x1 tile. Expanded outputs
remain bitwise equal to the untuned Rust baseline. Separate instrumented probes
retain identical active/retained storage; borrowed REC/OCR has zero warmed
allocation count, bytes, retained growth and peak growth. The larger vocabulary
projection tile was rejected after its microbenchmark regressed. Existing batched
MatMul tiling and contiguous QKV slicing were evaluated without further changes.

## Pending final acceptance

The retained kernel has completed native Windows/Linux parity, Rust checks,
public tuning selection and actual cold/warm kernel-specific hosted CI. The
additional all-family public API gate and stable prepared-launcher tooling suite
have passed locally on both native platforms. Official C# comparisons and remote
CI for the additive test/workflow/tooling checkpoint remain **pending**. Earlier
Rust, Python, baseline CI and WSL evidence retain their recorded source scopes.
