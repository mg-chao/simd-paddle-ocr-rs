# All-model scalar validation

The pre-tuning scalar checkpoint passed expanded native REC and full OCR parity
for tiny, small and medium on Windows and Linux. The proofs below pin the source
snapshot before subsequent public-kernel tuning. They do not certify later source
changes. Final tuned-source proofs and final all-model remote CI results are
**pending**.

## Frozen native proof identity

The [Windows expanded proof](../target/validation/all-model-windows-expanded-v2/proof.json)
and [Linux expanded proof](../target/validation/all-model-linux-expanded-v1/proof.json)
both report `all_native_rec_ocr_gates_passed`, `scope: expanded` and
`source_changed_during_validation: false`. Each ran all six REC/OCR executions and
all six comparisons. Reference manifests and executable hashes were unchanged at
completion.

| Platform | Completed (UTC) | CLI executable SHA-256 |
|---|---|---|
| Windows 11 / AMD64 | 2026-10-14 16:30:21 | `8c28a569a3c560bebf1164b3ac932ae9b89aaa103913bdf525a0265eaf18244b` |
| Linux / WSL2 x86_64 | 2026-10-14 16:34:04 | `00c8156ef2a59bd3a68c5cdf06df5fb42101ce7b77032c367325e2e9e7fe9aae` |

Both proofs contain the same 48-file source hash map covering Rust source, tests,
Cargo manifests and the toolchain declaration. Its canonical JSON SHA-256 is
`def4d7edd37dde0ba42f17d120ecf608f989516f334b72c12a1cd8ecceff4284`
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

The final ordinary all-feature test runs passed **155 tests on Windows** and
**152 on Linux**, with zero failures and 16 ignored archive-dependent tests on
each platform. The expanded CLI proofs above are separate executed gates, rather
than inferred success from ignored tests. All-target/all-feature Clippy passed
with `-D warnings`, and the Python tooling suite passed **64 tests**. Logs are
preserved in `target/validation/final-tooling-v1` and the Linux proof's
`final-checks` directory.

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

## Proven remote tiny CI

The [cold-cache run 38587687253](https://github.com/mg-chao/simd-paddle-ocr-rs/actions/runs/38587687253)
and [warm-cache run 38592128880](https://github.com/mg-chao/simd-paddle-ocr-rs/actions/runs/38592128880)
both completed successfully on Windows and Linux at commit
`97757677048bf17ef93c876536746e31518178aa`. The cold run bypassed cache restoration.
The warm run's downloaded job logs explicitly record cache hits and successful
restoration on **both** operating systems, followed by verified asset acquisition.
These runs establish tiny hosted-fixture CI, not final small/medium remote results.

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

## Pending final acceptance

Final all-model remote CI links, tuned-source Windows/Linux proofs and their
updated source/executable identities remain **pending**. Record them after the
candidate source freezes and the actual gates finish. The pre-tuning results in
this document remain pinned historical evidence.
