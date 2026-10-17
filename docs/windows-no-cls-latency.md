# Latency with orientation classification disabled

Disabling classification reduced Tiny/dense median owned-OCR latency from
**69.2114 to 44.1710 ms (36.2% lower, 1.567× speed)** on the physical Windows
host. All 69 text lines, recognition scores, geometry and recognition dimensions
were identical. Mixed-page latency also fell, but its output changed; the sparse
vertical-text case had overlapping timing ranges and changed all three lines.

These measurements were taken on 2026-10-17 at production revision
`061c33b60fd9d14f63119f19552bdc0278e2f083`, after the scheduling, pointwise and
detector optimizations. Both modes used the same rebuilt release executable.
Older timings in the [stage analysis](windows-cpu-stage-analysis.md) are separate
experiments and are not the baseline for this comparison. Production defaults
were not changed.

| Tiny fixture | Lines | CLS enabled median, ms | CLS disabled median, ms | Time saved, ms | Latency reduction | Text changes |
|---|---:|---:|---:|---:|---:|---|
| Sparse / `imgs_1` | 3 | 43.9571 | 42.8163 | 1.1408 | 2.6%; inconclusive | 3 of 3 lines |
| Dense / `imgs_00018069` | 69 | 69.2114 | 44.1710 | 25.0404 | **36.2%** | None |
| Mixed / `imgs_00006737` | 31 | 80.9221 | 71.7193 | 9.2028 | **11.4%** | 1 of 31 lines |

| Fixture | Three enabled samples, ms | Three disabled samples, ms |
|---|---|---|
| Sparse | 45.1623, 43.6833, 43.9571 | 42.7939, 42.8163, 44.4795 |
| Dense | 69.3443, 69.2114, 68.4106 | 43.8208, 44.1710, 45.2026 |
| Mixed | 80.9221, 80.9112, 81.7919 | 69.8012, 72.3105, 71.7193 |

Dense and mixed observed ranges are separated; sparse ranges overlap. Three
samples describe this bounded local experiment, not a statistical confidence
interval. These are corpus images, not captured screenshots. The dense result
supports skipping classification when orientation correction is unnecessary,
but it does not establish latency for a particular screen-capture application.

With classification enabled, all three sparse crops receive a 180° correction.
Disabling it changes, for example, `华南农业大学` to `知不刘要花`. The mixed page
has one corrected crop whose text changes from `M` to `W` when classification is
disabled; no ground-truth accuracy judgment was made for that line. All cases
retain identical detection counts, coordinates, detection scores, detector
sizes and recognition tensor dimensions. Disabled classification scores,
labels and applied 180° rotations are zero. Geometric vertical-crop rotation
is a separate operation and remains enabled.

The machine identifies as an AMD Ryzen 9 5950X, with 12 enabled cores / 24 logical
processors exposed by Windows and approximately 64 GiB RAM. The benchmark used
eight logical CPUs with affinity `0xff`, one request, automatic SIMD/layout,
fusion, adaptive recognition width, batch size one, detector limit 960, and
alignment capture disabled. `line_workers=0` preserves automatic scheduling:
the static budget prints four owners, while the current AVX2/FMA policy selects
eight recognition owners for the dense and mixed pages with enough units.

Timing covers complete OCR from predecoded BGR24 through conversion to an owned
result. It excludes screen capture, image decoding, model import and compilation.
The existing Rust `_worker` imports classifier weights in both modes; the disabled
builder skips classifier compilation and execution. Thus these results measure
the inference toggle, not the additional cold-load savings available by omitting
`cls.onnx` entirely. The baseline classifier threshold is 0.0 in both manifests.

Validation first passed the focused synthetic builder/optional-classifier test.
Used model assets, source images and decoded pixels passed their pinned SHA-256
checks before timing. Each of six separate worker processes performed one
separately recorded first inference, one warm-up and three measured calls; every
complete result within a mode matched its first-inference control exactly.
There were 18 measured calls and 30 total inference calls. Worker execution and
collection took **2.733 seconds**, within the 60-second budget, with a 10-second
deadline per worker. There were no retries or additional sampling. Source and
executable hashes remained unchanged during measurement. No CI or remote runner
was used. The build's existing Windows unused-import warning is unrelated.

The [raw evidence](reports/windows-x64-2026-10-17-no-cls-latency.json) preserves
all samples, first-call timings, results, line differences, execution order,
settings, jobs and source/model/executable hashes. Local jobs, copied manifests
and the bounded runner remain in `target/no-cls-latency-20261017/`.

To reproduce the method, build `cpu_benchmark` with release/default features,
prepare matching predecoded-pixel jobs, and copy `tests/baseline/manifest.json`
twice, changing only `settings.direction_classification` between `true` and
`false`. Invoke `cpu_benchmark.exe _worker <job.json>` once per mode and fixture,
with `threads=8`, `concurrency=1`, `runs=3`, and `affinity="ff"`. Alternate mode
order across fixtures as recorded in the evidence. This uses the internal
Rust worker as a diagnostic A/B; the normal Rust/C# qualification command
requires classification enabled and is not a no-classification benchmark.

Applications can select the tested behavior with
`.direction_classification(false)` or the OCR example's `--no-cls` option.
