# All-model scalar validation

The frozen scalar kernel candidate passed expanded native REC and full OCR parity
for tiny, small and medium on Windows and a native Linux container. Both proofs
pin the same Rust source snapshot. The committed all-model baseline also passed
cold and warm hosted CI on both operating systems. The retained scalar changes
passed the complete public tuning screen. Official C# comparisons and remote CI
for the retained kernel remain **pending**.

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
with `-D warnings`. The final full Windows Python tooling suite passed **71 tests**;
source hashes matched before and after the run. Final Linux Python tooling ran
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
results certify the committed baseline; they do not replace remote CI for the
later kernel candidate.

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

## Compact CI versus preserved expanded references

The [fixture index and acquisition contract](../tests/fixtures/step12/README.md)
select one complete four-case hosted bundle per family for both REC and OCR.
Tiny remains on `fixtures-v1`; small and medium use `fixtures-v2`. Compact counts
are 2,092, 2,788 and 2,808 artifacts respectively. Archive and model byte counts
and SHA-256 hashes are checked for supplied files, cold downloads and cache reuse.
Named parity/corruption tests are listed before execution to reject zero-test
filters. All-family allocation gates are configured in the Windows/Linux workflow.

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

The frozen kernel candidate has completed native Windows/Linux parity and the
final Rust checks and public tuning selection. Official C# comparisons and
kernel-specific remote CI remain **pending**. Record their actual outcomes after
completion; successful baseline CI and earlier WSL evidence retain their original
source scopes.
