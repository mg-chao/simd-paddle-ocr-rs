# Testing

- Run validation locally on the physical machine. Do not use or rely on CI, remote runners, or virtual machines.
- Run focused correctness checks first. Benchmark only when needed to answer a concrete performance question about the change.
- Prefer the affected kernel or stage over the full pipeline. Default to 1–3 representative inputs, one configuration, at most one warm-up, and 1–3 measured runs per code version.
- Keep total performance execution within 60 seconds per code version by default, excluding build time. Count harness repetitions and internal loops toward the budget; avoid automatic sampling growth.
- Whenever code changes, reset the run count and rebuild before measuring again. Previous results may serve as a baseline but do not validate the new version. A reset does not require another benchmark.
- Use release builds and consistent inputs and settings for comparisons. Do not pool results from different code versions or test setups.
- Independently decide whether larger tests are necessary; do not request user approval. Expand only to resolve a specific question that focused tests cannot answer, and set a bounded scope, run count, and timeout before starting.
- Stop once the question is answered. Do not rerun unchanged code for cleaner or more favorable numbers; report uncertainty when results are noisy.
- Briefly report the checks performed, sample and run counts, results, and any local validation limitations.

# Commit Conventions

- Use Conventional Commits with lowercase, imperative subjects in the format `<type>(<scope>): <summary>`.
- Use these types: `feat`, `fix`, `refactor`, `perf`, `test`, `style`, and `build`.
- Choose a scope that names the affected feature or component, such as `ocr`, `simd`, or `kernels`.
- Keep the summary concise and lowercase, and start it with an imperative verb; for example, `fix(ocr): handle empty input safely` or `perf(simd): accelerate image normalization`.
