# Real-model CPU configuration qualification

`cpu_qualify` extends the [serial CPU baseline](cpu-baseline.md) without changing
its manifest, goldens, or comparison tolerances. It runs Rust and the pinned C#
CPU implementation on the same hash-verified decoded pixels in independent
reusable worker processes. The C# project and all its references build into the
invocation's artifact directory, so qualification can coexist with other runners.

Use the pinned nightly, physical Windows x64 machine, .NET 10 SDK, clean pinned
sibling `SimdPaddleOCR` checkout, and models/corpus described by the baseline guide:

```powershell
cargo run --release --locked --example cpu_qualify -- --list
cargo run --release --locked --example cpu_qualify
```

The complete suite has **213 requests: 71 per bundle**, each run once. Its overall
deadline is 60 minutes including C# build; each worker response and shutdown has
a 120-second deadline. This is correctness qualification, with no warm-up,
sampling growth, or performance claim. Timeout or worker failure terminates the
worker process tree and leaves a failed report and available transcripts.

| Profile | Cases per bundle | Coverage |
| --- | ---: | --- |
| `default-parallel` | 22 | All canonical fixtures, automatic CPU/SIMD/layout and fusion; also compared with saved serial goldens |
| `serial` | 3 | Capacity and all stage budgets set to one |
| `workers-2`, `workers-4` | 3 each | Capacity and all stage budgets set to two or four |
| `batch-2`, `batch-4` | 2 each | Dense short lines and mixed-width page; same-width recognition batching |
| `alignment` | 6 | Three representative pages, both rotations of `imgs_11`, blank; CTC spans and estimated character boxes |
| `rgb-padded`, `bgra-padded`, `rgba-padded` | 3 each | 13 poisoned padding bytes per row, changing alpha ignored |
| `cls-off` | 3 | Both rotations of `imgs_11` and blank without classification |
| `fixed-width` | 3 | Non-adaptive recognition at width 320 |
| `default-thresholds` | 3 | Model-aware box threshold selected by each implementation |
| `dilation` | 3 | DB bitmap dilation enabled |
| `reuse` | 4 | Sparse → dense → mixed → sparse in one request, exercising width changes and reuse |
| `recovery` | 5 | Success → rejected stride → success with different widths → pixel-limit rejection → success in one request |

The three representative fixtures are `imgs_1` (sparse), `imgs_00018069` (dense
short lines), and `imgs_00006737` (mixed widths). Recovery uses a matched detector
pixel limit of 600,000: the dense and mixed fixtures fit, while `imgs_1` exceeds
it. An expected rejection must mention the relevant stride/pixel reason; other
failures cannot accidentally pass. Later successes must match C# output.

Use selections for focused checks and failed-case reruns; selections never modify
the checked-in matrix. Flags may be repeated. Fixture filters require explicit
profiles excluding the ordered `reuse` and `recovery` scenarios:

```powershell
cargo run --release --locked --example cpu_qualify -- --model tiny --profile alignment --fixture imgs_11_rotate180
cargo run --release --locked --example cpu_qualify -- --model tiny --profile default-parallel --fixture blank
cargo run --release --locked --example cpu_qualify -- --model small --profile recovery
cargo run --release --locked --example cpu_qualify -- --reference-root D:\SimdPaddleOCR --models-dir D:\models
```

Every invocation verifies all model/dictionary hashes, source and decoded fixture
hashes, golden provenance, and C# commit/cleanliness before inference. The default
parallel profile uses the physical machine's available CPU count in both workers
(explicitly passed as `DOTNET_PROCESSOR_COUNT` to .NET); worker logs record resolved
budgets and runtime/SIMD details. Explicit budgets are never silently substituted.
Standard release/default-feature builds are required; profiling and custom
`RUSTFLAGS` are rejected, as are inherited `PPOCR_*`, `SIMD_OCR_*`, processor-count,
.NET hardware-dispatch, and profiler overrides that could change the declared configuration.

Evidence is saved under `target/qualification/<timestamp>-<pid>/`: `run.json`
records selected profiles, case count, manifest/source provenance, build and
machine details, elapsed time, completed/selected case counts, `full` versus `focused`
coverage, and passed/failed status; `differences.json` records
comparison failures; each profile has its exact settings/inputs and each worker's
JSONL responses and diagnostic log. A focused pass qualifies only its recorded
selection. Claim full qualification only for a passing unfiltered 213-case run.

The original exact text, line ordering/counts, rotation, emitted count, and tensor
dimension checks remain intact. Scores use `1e-4 + 1e-4 * abs(expected)` and
coordinates use 0.5 source pixels, also for spans and character boxes. Captured
tokens, half-open time columns, null versus captured alignment, and box token
ordering must agree exactly. Non-finite values and malformed result structures
always fail. Contract checks can run without model inference:

```powershell
cargo test --release --locked --test qualification_contract
```
