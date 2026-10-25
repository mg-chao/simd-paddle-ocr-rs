# Qualified final result: bd26694591978bc1

All three measured Rust warm medians are below the reused C# control on the
physical Apple M4/Metal machine. This is a bounded result for the qualified
explicit fast mode, not a claim about other GPUs, models or arbitrary inputs.
Tiny has substantial observed run variation; Small's median lead is modest.

| Preset | Rust warm samples (ms) | Rust median | Reused C# samples (ms) | C# median | Rust vs C# |
|---|---|---:|---|---:|---:|
| Tiny | 20.885500, 19.036291, 18.769000 | 19.036291 | 22.2229, 20.1564, 18.7985 | 20.1564 | −5.56% |
| Small | 43.082958, 42.436834, 42.534042 | 42.534042 | 44.2365, 43.2270, 42.8181 | 43.2270 | −1.60% |
| Medium | 154.650542, 153.766625, 153.957416 | 153.957416 | 161.8799, 161.4153, 161.4062 | 161.4153 | −4.62% |

The previous fc06 Rust medians were 20.109250 / 45.313667 / 155.337292 ms. No
individual kernel or stage contribution is inferred from this combined change;
isolated gains are not additive. No unchanged performance rerun was used.

## Measurement scope

- One typical full-page input for each Tiny/Small/Medium model, one cold run and
  three warm samples, one request, one slot, recognition batch 2 / wave 16,
  classifier batch 16; no profiling during measurement.
- Explicit matrix-F16, MixedF16 storage, native Metal kernels/graph execution
  and projection enabled. CPU options: detector 8, line workers 4, preprocessing 16.
- Wall timing runs from group start through the last inference return, excluding
  image decoding, owned-output copy, validation, serialization and profile calls.
  CTC and dictionary decoding remain inside the public OCR call. The
  separately reported process CPU clock includes owned-output copy.
- Rust process CPU samples (ms): Tiny 2.722 / 2.557 / 2.358,
  Small 3.092 / 2.793 / 2.795, Medium 3.274 / 2.859 / 2.847. Raw cold times, CPU options,
  counters and memory are retained in the JSON audit.
- C# was **not rerun** for this final version. The unchanged frozen binaries and
  three samples from `performance-counter-db-paired` were reverified and reused.
  C# qualifies against its own cold output; Rust qualifies against the CPU
  reference. These accuracy scopes differ.
- Tiny/Small actual recognition tensor widths and timeSteps equal C#. Medium
  still has zero-based line 6: width 544 / T68 versus C# 608 / T76, with 1308672 versus 1317888
  recognition input elements, **0.6993007% less Rust work**, already present in
  fc06/62d. Medium's result is an accepted whole-OCR comparison, not identical
  neural matrix dimensions. Full ordered workloads are retained in the audit.

## Accuracy and regression evidence

Final same-source checks passed: 481 ordinary tests, 178 ignored; Clippy
all-features/all-targets `-D warnings`; CPU-only and GPU-only checks; release
test/example builds; 10 focused hardware tests (3 attention, 3 square-DW7,
2 pooled-classifier projection, 1 packed-attention regression, 1 prior depthwise
regression). The nine fast cases (288 lines, wave 16) and nine strict cases
(288 lines, wave 1) all passed their existing contracts.

Every parsed captured output in **all 18 cases is exactly equal to fc06, 62d and
fa484**, including text, geometry, scores, CTC spans and emitted dimensions.
There are zero new captured-output mutations. Fast physical-family recipe
counts changed with the new selection; actual neural output dimensions and
workload remained equal to those historical Rust versions. Strict recipe counts
also remained equal. The strict harness emits no per-line CPU delta audit;
exact historical capture equality does not manufacture one.

The unchanged fast-versus-CPU audit has 9 changed-input geometry/shape lines,
5 lines with CTC-column differences, minimum polygon IoU 0.8489825482, maximum
coordinate error 2 px and maximum corner distance 2.236081626 px. All 279
equivalent-input lines pass the existing confidence gates: maximum raw absolute
recognition/classification/detection score deltas are
0.0555664282 / 0.00462247373 / 0.00113162356. For the 9 changed-input lines the harness
checks finite values and classification/detection probability bounds, retaining
raw deltas instead of applying a same-input confidence gate; their maxima are
0.8351671870 / 0.00233598712 / 0.1344503561. No tolerance was widened for this version.

Normal execution retains three queue submissions per page. Tiny/Small/Medium
dispatch totals are 925 / 1340 / 1427, including 803 / 1218 / 1305 native neural graph
dispatches and 122 native resident dispatches each. Control readback is 400 bytes;
output readback is 27744 / 26496 / 28992 bytes. Retained/peak device bytes remain
33988352 / 39246488, 96440796 / 102976292 and 207284632 / 214258040, matching fc06. These
counters are workload evidence, not additive timing attribution.

## Frozen artifacts

The audit and consolidator executed after the performance window closed.
Fourteen final wrapper records have exit code 0. Five earlier failed attempts
remain separately identified with actual source versions and exit 101: the two
prior fc06 fixture/build attempts, c61d private-helper build, ac899 attention
host-mode fixture and 8ea test-only Clippy style. They are not presented as passes.

- [Accuracy, all raw deltas, capture comparisons and samples](accuracy-attention-dw7-classifier-audit.json)
- [Artifact hashes, all wrapper statuses and CPU/GPU counters](current-attention-dw7-classifier-artifacts.json)
- [Final performance manifest](performance-attention-dw7-classifier-wave16/manifest.json)
- [Fast corpus manifest](correctness-attention-dw7-classifier-corpus/manifest.json)
- [Strict corpus manifest](correctness-strict-attention-dw7-classifier-corpus/manifest.json)

Verified 262 checkpoint/live source entries and 258 shared source entries in each
report ZIP, plus its two Python driver files. The report packager intentionally
omits four diagnostic examples from the checkpoint; that explicit difference is
recorded rather than treated as byte-identical archive packaging.

| Artifact | SHA-256 |
|---|---|
| Source checkpoint `bd26694591978bc1.zip` | `1b10c0a9080c5844a36ee130e03976dd6321dd88b925d4a1afe14f8e96b7e606` |
| All three report source ZIPs | `d6d46b10cb6c21fe34327e9137b00ab94d15a396cccf1db92fa3565bb852cca8` |
| Archived OCR benchmark | `498b45456a3f454fa57d0f67fc750823d889ae540cc65d4a75e6aaa4ec7f599a` |
| Archived test executable | `53db8d9ea8603e40410060a509d0b5bb8d850012633694c789d904c22ff14183` |

The test executable is preserved at
`artifacts/bd26694591978bc1/simd_paddle_ocr_rs_tests`. All nine reused C# artifact
hashes and the original control manifest hash were reverified. No GPU runs,
compilation or production edits were performed by the consolidation helpers.
