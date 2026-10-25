# Current native Metal evidence

Qualified source: `0badef5b4d61e8be`, Apple M4 / macOS 27. GPU execution uses the
single native Metal implementation; default F32 and explicit F16 are numerical
policies on that executor. CPU behavior and the external C# checkout are preserved.

| Evidence | Result / location |
|---|---|
| Final builds and checks | `build-*-qualified`, `clippy-qualified`, `check-cpu-qualified`, `check-gpu-qualified` JSON/log pairs; all exit 0 |
| Regular tests | `regular-tests-qualified`: 475 passed, 122 ignored |
| Focused hardware | `hardware-suite-qualified`: 40 passed |
| Strict OCR | `strict-corpus/manifest.json`: 9 GPU cases with fresh CPU references; F32 storage/matrix, wave 1, classifier capacity 4 |
| Explicit F16 OCR | `fast-corpus/manifest.json`: 9 GPU cases with fresh CPU references; MixedF16/F16, wave 16, classifier capacity 16 |
| Classifier tensor | `tensor-cls/summary-metal-f32.json`: `[2,3,80,160]`, zero failures, max absolute error `7.748603820800781e-7` |
| Performance | `performance/manifest.json`: typical image, 3 presets, one cold plus 3 warm samples each; all qualified |
| Preserved executables | `artifacts.json`: SHA-256, size and original build path for `gpu-tests`, `gpu_ocr_bench`, `gpu_compare`, `ocr` |
| Cleanup | `cleanup.json` and `cleanup-candidates.json`: exact removed paths; dependency caches, CPU-history data and retained evidence excluded |

The `*-qualified.json` wrappers pin this source and executable identity. Earlier
numbered/frozen attempts are diagnostic history, including failed checks; they
are not additional passing validation. The performance directory's source ZIP
contains only the current implementation and its validation helpers.

The Rust warm medians are 19.649750 / 43.714667 / 159.113708 ms for Tiny / Small /
Medium. The unchanged, reused C# control is 20.1564 / 43.2270 / 161.4153 ms in
`../pre-cleanup-bd26694591978bc1/csharp-control`. Tiny/Small sample ranges overlap;
Small's median is 1.13% slower. Medium's median is 1.43% faster in this sample,
with 0.6993007% less recognition input work. C# uses its original own-cold-output
reference scope; it was not rerun here or qualified against Rust's CPU contract.
Text and line counts match all three presets, but one line in each retains the
preexisting C# label 1 / 180° versus Rust label 0 / 0° difference. Equal dimensions
do not establish identical crop contents. Rust's CPU orientation gates pass.
See [performance scope](../../../docs/gpu-performance.md) for every raw sample.

Profiling uses the same native schedule. It records kernel metadata and host
timing; `device_ns` is null and granularity is unavailable. It cannot provide
per-kernel GPU durations. Ordinary performance is uninstrumented and excludes
image decoding, owned output copies, validation, serialization and profile calls;
CTC and dictionary decoding remain inside the timed inference call.

## Commands after build-cache cleanup

Run from the repository root on the physical Mac, with the unchanged sibling
models present. These are reproducibility instructions, not a request to repeat
unchanged performance measurements. Use a new report directory for any justified
future verification. The three `target/release/examples` aliases are restored
copies of the corresponding preserved executables; Cargo caches are unnecessary.

```sh
validation/metal/current/artifacts/gpu-tests --test-threads=1

python tools/full_gpu_validation.py \
  --executable validation/metal/current/artifacts/gpu_ocr_bench \
  --sample fixtures/metal-ocr/corpus/typical.png \
  --presets tiny small medium --cases typical dense vertical \
  --report validation/metal/recheck-strict --budget-seconds 120

python tools/full_gpu_validation.py \
  --executable validation/metal/current/artifacts/gpu_ocr_bench \
  --sample fixtures/metal-ocr/corpus/typical.png \
  --presets tiny small medium --cases typical dense vertical \
  --precision mixed-f16 --matrix f16 --wave 16 --classification-batch 16 \
  --report validation/metal/recheck-fast --budget-seconds 120

SIMD_OCR_COMPARE_DIR=validation/metal/recheck-cls \
  validation/metal/current/artifacts/gpu_compare \
  ../SimdPaddleOCR/models/cls.onnx 2 3 80 160
```

To select exactly the recorded 40 hardware tests without retaining a temporary
runner, use the names from the qualified log:

```sh
python3 - <<'PY'
from pathlib import Path
import re, subprocess
base = Path('validation/metal/current')
names = re.findall(r'^test (\S+) \.\.\. ok$', (base/'hardware-suite-qualified.log').read_text(), re.M)
assert len(names) == 40
subprocess.run([str(base/'artifacts/gpu-tests'), *names, '--exact', '--ignored', '--test-threads=1'], check=True, timeout=120)
PY
```

The bounded performance command uses the restored current benchmark alias:

```sh
python tools/gpu_performance.py \
  --corpus validation/metal/current/fast-corpus/corpus \
  --report validation/metal/recheck-performance \
  --presets tiny small medium --cases typical --targets gpu --configs 1:1:2 \
  --matrix f16 --storage f16 --wave 16 --classification-batch 16 \
  --alignment --measurements 3 --budget-seconds 25
```
