//! Literal contracts independent of compact_argmax and reference_logits.
use super::*;

#[test]
fn audit_portable4_columns_complete_tiles_unique_tail_and_failure_recovery() {
    let cpu = CpuContext::serial();
    let inner = 65;
    let big = 16_777_216.;
    for columns in [1, 7, 15, 16, 17, 33, 1031] {
        let complete = columns / 16 * 16;
        let probes = if complete == 0 {
            vec![columns - 1]
        } else {
            vec![15, complete - 1]
        };
        for winner in probes {
            let mut weights = vec![0.; inner * columns];
            let mut bias = vec![-2.; columns];
            weights[winner] = big;
            weights[columns + winner] = -big;
            bias[winner] = 1.;
            // Complete tile: (BIG + -BIG) + 1 = 1. All-tail case:
            // (1 + BIG) + -BIG = 0 because 1 + BIG rounds to BIG.
            let expected_bits = if complete == 0 {
                0x0000_0000
            } else {
                0x3f80_0000
            };
            if complete > 0 && complete != columns {
                // An ordinary tail bias ties a full-tile winner; the complete
                // tile must win at its earlier column. All-tail has no tie.
                bias[columns - 1] = 1.;
            }
            let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
            for rows in [4, 28] {
                let input = vec![1.; rows * inner];
                let mut workspace = ArgMaxWorkspace::default();
                for workers in [1, 2, 4, 8] {
                    workspace.reserve(rows, workers).unwrap();
                    let mut indices = vec![usize::MAX; rows];
                    let mut scores = vec![f32::NAN; rows];
                    execute::<4, 4, false>(
                        &input,
                        &bias,
                        &packed,
                        &mut indices,
                        &mut scores,
                        inner,
                        columns,
                        &cpu,
                        workers,
                        &mut workspace.partials,
                    )
                    .unwrap();
                    assert_eq!(
                        indices,
                        vec![winner; rows],
                        "columns={columns}, workers={workers}"
                    );
                    assert_eq!(
                        scores
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        vec![expected_bits; rows]
                    );
                    let owners = workspace
                        .partials
                        .chunks(rows)
                        .filter(|shard| shard.iter().any(|winner| winner.score.is_finite()))
                        .count();
                    if complete == 0 {
                        // Fresh workspace: an all-tail invocation executes one
                        // owning shard, irrespective of the requested budget.
                        assert_eq!(owners, 1);
                    }
                }
            }
        }
    }
    // Both an all-tail vocabulary and a vocabulary with complete tiles reject
    // every nonfinite remainder before publishing any caller outputs. Reuse
    // the same workspace after each failure, preserving scalar bias order.
    for columns in [7, 33] {
        let rows = 12;
        let input = vec![1.; rows * inner];
        let mut weights = vec![0.; inner * columns];
        weights[columns - 1] = big;
        weights[2 * columns - 1] = -big;
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let mut workspace = ArgMaxWorkspace::default();
        for workers in [1, 2, 4, 8] {
            workspace.reserve(rows, workers).unwrap();
            for invalid in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
                let mut bias = vec![-2.; columns];
                bias[0] = 0.5;
                bias[columns - 1] = invalid;
                let mut indices = vec![91; rows];
                let mut scores = vec![42.; rows];
                let run = |bias: &[f32],
                           indices: &mut [usize],
                           scores: &mut [f32],
                           state: &mut ArgMaxWorkspace| {
                    execute::<4, 4, false>(
                        &input,
                        bias,
                        &packed,
                        indices,
                        scores,
                        inner,
                        columns,
                        &cpu,
                        workers,
                        &mut state.partials,
                    )
                };
                assert!(run(&bias, &mut indices, &mut scores, &mut workspace).is_err());
                assert_eq!(indices, vec![91; rows]);
                assert_eq!(scores, vec![42.; rows]);
                bias[columns - 1] = 1.;
                run(&bias, &mut indices, &mut scores, &mut workspace).unwrap();
                // The tail's bias-first cancellation is zero; class zero's
                // literal half wins. Bias-after in a partial tile would pick
                // the last class with score one instead.
                assert_eq!(indices, vec![0; rows]);
                assert_eq!(
                    scores
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    vec![0x3f00_0000; rows]
                );
            }
        }
    }
}
