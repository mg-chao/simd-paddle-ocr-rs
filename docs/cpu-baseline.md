# CPU correctness baseline

This local baseline compares Rust with SimdPaddleOCR C# predictions. It does
not measure OCR accuracy against human annotations. It covers all 22 fixtures
with ChineseV6Tiny, ChineseV6Small and ChineseV6Medium (66 combinations).
No CI, VM, GPU, benchmarking, warm-up, or sampling is involved.

## Setup and normal verification

Use the repository's `nightly-2026-09-08` Rust toolchain and .NET SDK
`10.0.303` (the adapter's `global.json` disables SDK roll-forward). Keep a clean
C# reference checkout at commit `33ff8a81672452041d110e8ba52dcd2639e54228`.
For a new workspace, from the Rust repository:

```powershell
git clone https://github.com/mg-chao/SimdPaddleOCR.git ../SimdPaddleOCR
git -C ../SimdPaddleOCR checkout 33ff8a81672452041d110e8ba52dcd2639e54228
cargo build --release --locked --examples
cargo test --release --locked
cargo run --release --locked --example cpu_baseline -- verify
```

Existing workspaces can use their sibling checkout directly. Override its
location with `--reference-root PATH`; override the model directory separately
with `--models-dir PATH`. Paths can contain spaces. The C# checkout's HEAD and
non-ignored working tree must match the pin. Model binaries are external; they are
not copied into this Rust repository. No models are downloaded by verification.

All nine model/dictionary hashes, the three bundle mappings, the public corpus
commit, source URLs and source/decoded-pixel hashes are in
[`manifest.json`](../tests/baseline/manifest.json). Verification checks every
input and all golden metadata before starting inference. It fails for absent
or modified assets rather than skipping cases. Dictionary entries explicitly
pin both Git's LF bytes and Windows CRLF checkout bytes; only those two exact
whole-file hashes are accepted, with no input normalization. Manifest and
golden JSON checkout line endings are fixed to LF by `.gitattributes`.
Keep custom `RUSTFLAGS`,
`CARGO_ENCODED_RUSTFLAGS` and `CARGO_BUILD_RUSTFLAGS` unset. No `target-cpu=native`,
custom profile, feature override, or forced instruction set is required.
Validation tools also reject inherited `PPOCR_*`, `SIMD_OCR_*` and CLR
instruction-set/profiling/layout overrides so they cannot silently change the
claimed automatic execution configuration. Stage budgets are set by each tool.

For **offline Rust-only verification**, no .NET SDK or C# source tree is needed:

```powershell
cargo run --release --locked --example cpu_baseline -- verify --rust-only --models-dir ../SimdPaddleOCR/models
```

Build dependencies must already be cached for an offline build; add Cargo's
`--offline` option if network access must be prohibited. Runtime verification
does not access the network.

## Execution and comparison contract

The Rust parent decodes each original exactly once with the locked `image`
crate, sniffing its bytes because upstream file extensions are not always
accurate. It converts RGB to tightly packed BGR24 without resizing or applying
EXIF orientation, checks the decoded hash and dimensions, and writes those
same bytes for both workers. Derived rotations are clockwise, without
resampling. The generated blank image is white, 512x512.

Rust uses `CpuContext::serial()` with one worker in every stage, default
compile options (automatic SIMD/layout, fusion on), and default Cargo
features. C# runs in a separate process with `DOTNET_PROCESSOR_COUNT=1`, all
worker budgets set to one, and DET/CLS/REC explicitly set to `OcrBackend.Cpu`.
The normal C# `Auto` backend can choose a GPU and is deliberately not used.
All other compared settings are read from the shared manifest: direction
classification on, classifier threshold 0, recognition batch 1, adaptive
recognition width with target 320, detector limit 960, bitmap threshold 0.2,
unclip ratio 1.4, max candidates 3000, dilation off, and box thresholds
0.4/0.45/0.45 for tiny/small/medium. CTC alignment capture is off.

The versioned JSON interface records fixture IDs, detected count, detector
dimensions, ordered line text/quadrilaterals, all three scores, classification
labels, applied rotations, emitted counts and recognition content/tensor/time
dimensions. Unknown or missing result fields and non-finite values fail.

- Text (including whitespace), ordering, counts, labels, rotations, and integer
  dimensions must match exactly. Lines and quad vertices are never reordered
  or matched by nearest neighbor.
- Each source-coordinate component permits an absolute difference of 0.5 px.
- Scores require `abs(actual - expected) <= 1e-4 + 1e-4 * abs(expected)`.
- Full verification compares golden → Rust, golden → live C#, and live C# →
  Rust. Every fixture runs once per implementation per invocation, with no
  automatic retries. Workers reuse each model but not prior results.

Each case has a 120-second deadline, including first-case model setup. The
suite has a 30-minute deadline, including the C# adapter build. Timeout,
malformed output, worker failure, incomplete coverage, or any difference
produces a nonzero exit code. Child workers are terminated and reaped on error.
Each invocation builds the adapter and all referenced C# projects into its own
`target/baseline/<run>/csharp-artifacts` directory and launches that adapter's
DLL. Concurrent verification does not rebuild another run's loaded assembly.
Build timeouts also terminate the build process tree.

## Corpus restoration and baseline maintenance

The corpus contains the 13 numerically named JPEG files in PaddleOCR v2.7.3's
`doc/imgs` and the six JPEG files in `doc/imgs_en`, pinned to commit
`ddaa85ddafe1edfbe618b83ca0e8e61dab33155b`. It adds 90° and 180° variants of
`doc/imgs/11.jpg`, plus the blank case. See the preserved
[license and attribution](../tests/baseline/NOTICE.md).

Original images are checked in. If any are missing, restore them with:

```powershell
pwsh -File tools/fetch-baseline-corpus.ps1
```

The script downloads only missing originals from immutable URLs, verifies
each checksum before installing it, and rejects changed existing files.
Routine verification never downloads or rewrites corpus or expectations.

To regenerate C# expectations intentionally:

```powershell
cargo run --release --locked --example cpu_baseline -- record
```

This writes candidates `tiny.json`, `small.json`, and `medium.json` to the
printed run directory under `target/baseline`. Review changes and provenance,
then copy those three files into `tests/baseline/expected` and run full
verification. `record` never overwrites checked-in expectations. Goldens pin
the complete manifest's SHA-256, reference revision, settings, toolchain and
host metadata. Updating inputs or configuration therefore requires explicit
regeneration; changing the manifest cannot silently reuse old expectations.

When deliberately changing the corpus or decoder, first update original
source hashes/provenance as appropriate, then run:

```powershell
cargo run --release --locked --example cpu_baseline -- pin-pixels
```

Review `target/baseline/manifest.candidate.json`, replace the manifest only
after reviewing pixel/dimension changes, then regenerate expectations and
verify. Never bless changed hashes or numerical tolerances just to hide a
regression. New legitimate corpus sizes require updating the coverage contract
and tests as well.

## Failure diagnosis and evidence

Every run prints its unique `target/baseline` directory. `run.json` records
status, source state, toolchains, CPU and configuration identity. Per-model
stderr logs record effective budgets/backend and .NET instruction-set support.
Per-case actual JSON, worker JSONL transcripts, input BGR bytes, the C# build
log and `differences.json` remain available on failure. Differences identify
the model, fixture, line and field. A failed suite is never reported as a
complete baseline.

Start with the first differing field. Dimensions indicate preprocessing or
configuration; counts/quads point to detection/postprocessing; labels and
rotations to classification/cropping; text/emissions to recognition/decoding.
Use existing stage trace APIs for a focused reproduction. Fix demonstrated
causes and retain difficult fixtures; do not replace C# expectations with Rust
outputs or widen tolerances to obtain a pass. Cross-CPU bit identity is not
claimed; automatic SIMD can change floating-point rounding.

## Recorded local validation

Validated on the physical Windows x86_64 host on 2026-10-16 using the pinned
Rust nightly, .NET SDK 10.0.303 / runtime 10.0.11, automatic AVX2/FMA dispatch,
and one inference thread per implementation. Standard release example build
passed. All 274 library tests, seven baseline contract tests, and five doctests
passed (286 total).

The final full comparison ran each model/fixture once in Rust and once in
C#: all 66 combinations passed all three comparisons with zero differences
outside the declared contract. A separate Rust-only invocation passed all
66 combinations while given a nonexistent C# checkout path and an explicit
model directory. No instruction-set or optimization flags were added.

| Model | Cases | Detected lines | Largest coordinate delta (px) | Largest score delta |
| --- | ---: | ---: | ---: | ---: |
| tiny | 22 | 448 | 0.000045313 | 0.000042264 |
| small | 22 | 439 | 0.000049414 | 0.000049222 |
| medium | 22 | 452 | 0.000052735 | 0.000131350 |

Deltas above compare saved C# expectations with Rust output; score acceptance
uses the combined absolute/relative formula, not just its absolute term.
Wrong-reference and custom-Rust-flag CLI checks failed before inference as
intended. Deleting and restoring one original through the download script
reproduced its exact SHA-256. Other CPUs and operating systems have not been
validated by this run.

For the independent Apple M4 / macOS ARM64 validation, see
[macOS CPU support](macos-cpu-validation.md).
