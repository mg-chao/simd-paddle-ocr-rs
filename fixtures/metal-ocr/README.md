# Metal OCR validation fixtures

Nine deterministic image fixtures and the nine Tiny/Small/Medium typical/dense/vertical CPU reference outputs, preserved before legacy GPU report cleanup. These are input/output data, not backend implementation snapshots. Models remain in the external SimdPaddleOCR checkout. Regenerate and qualify current CPU references with `tools/full_gpu_validation.py` when model or CPU semantics change.
