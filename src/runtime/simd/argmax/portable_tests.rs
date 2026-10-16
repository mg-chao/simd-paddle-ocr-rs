//! Independent cancellation contracts from the pinned C# packed Vector path.
//!
//! `MatMul.ArgMax.Vector.cs` starts each complete 16-column tile at zero,
//! performs all K products, then adds bias. `FinishScalarTail` instead starts
//! from bias. Literal IEEE-754 values below distinguish those two orders; no
//! production projection helper or `reference_logits` derives expectations.
use super::super::{Backend, PackedMatrix, compact_argmax};

const BIG: f32 = 16_777_216.;
const INNER: usize = 65;
const COLUMNS: usize = 1041;

fn project(input: &[f32], weights: &[f32], bias: &[f32]) -> (Vec<usize>, Vec<u32>) {
    let packed = PackedMatrix::new(weights, INNER, COLUMNS).unwrap();
    let rows = input.len() / INNER;
    let mut indices = vec![usize::MAX; rows];
    let mut scores = vec![f32::NAN; rows];
    compact_argmax(
        Backend::Portable4,
        input,
        weights,
        bias,
        Some(&packed),
        &mut indices,
        &mut scores,
        INNER,
        COLUMNS,
    )
    .unwrap();
    (indices, scores.into_iter().map(f32::to_bits).collect())
}

#[test]
fn portable4_compact_full_tiles_add_bias_after_k_across_rows_and_panels() {
    // With K=65, the implementation's 256 KiB panel boundary is column 1008.
    // Probe both sides and the last complete tile independently, so a hidden
    // losing column cannot make a wrong panel look correct.
    for column in [0, 15, 1008, 1039] {
        let mut weights = vec![0.; INNER * COLUMNS];
        let mut bias = vec![-2.; COLUMNS];
        for cancellation_column in [column, 1040] {
            weights[cancellation_column] = BIG;
            weights[COLUMNS + cancellation_column] = -BIG;
            bias[cancellation_column] = 1.;
        }
        bias[1] = 0.5;
        // Complete tile: (BIG + -BIG) + 1 = 1. Scalar column 1040:
        // (1 + BIG) + -BIG = 0, since 1 + BIG rounds to BIG.
        for rows in [1, 4, 5, 17, 129] {
            let (indices, scores) = project(&vec![1.; rows * INNER], &weights, &bias);
            assert_eq!(indices, vec![column; rows], "column {column}, rows {rows}");
            assert_eq!(scores, vec![0x3f80_0000; rows]);
        }
    }
}

#[test]
fn portable4_compact_biased_ties_choose_the_first_complete_tile() {
    let mut weights = vec![0.; INNER * COLUMNS];
    let mut bias = vec![-2.; COLUMNS];
    for column in [15, 1008] {
        weights[column] = BIG;
        weights[COLUMNS + column] = -BIG;
        bias[column] = 1.;
    }
    // The scalar tail's ordinary bias 1 ties the two complete tiles. The
    // earliest class is 15, independently of lanes, panels and row groups.
    bias[1040] = 1.;
    for rows in [1, 4, 5, 129] {
        let input = vec![1.; rows * INNER];
        let (indices, scores) = project(&input, &weights, &bias);
        assert_eq!(indices, vec![15; rows]);
        assert_eq!(scores, vec![0x3f80_0000; rows]);

        // Unpacked and scalar paths keep their separate bias-first contract;
        // the cancellation columns become zero and the tail alone wins.
        let packed = PackedMatrix::new(&weights, INNER, COLUMNS).unwrap();
        for (backend, packed) in [(Backend::Portable4, None), (Backend::Scalar, Some(&packed))] {
            let mut indices = vec![usize::MAX; rows];
            let mut scores = vec![f32::NAN; rows];
            compact_argmax(
                backend,
                &input,
                &weights,
                &bias,
                packed,
                &mut indices,
                &mut scores,
                INNER,
                COLUMNS,
            )
            .unwrap();
            assert_eq!(indices, vec![1040; rows]);
            assert_eq!(
                scores.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                vec![0x3f80_0000; rows]
            );
        }
    }
}

#[test]
fn portable4_compact_signed_zero_distinguishes_full_tiles_from_scalar_tail() {
    let weights = vec![0.; INNER * COLUMNS];
    let mut bias = vec![-2.; COLUMNS];
    bias[4] = -0.;
    bias[1040] = -0.;
    for rows in [1, 4, 5, 129] {
        let input = vec![-0.; rows * INNER];
        // Complete tiles start at +0: +0 + -0 remains +0, including the final
        // bias. Scalar tails start at -0 and remain -0. Equal values keep 4.
        let (indices, scores) = project(&input, &weights, &bias);
        assert_eq!(indices, vec![4; rows]);
        assert_eq!(scores, vec![0x0000_0000; rows]);
        bias[4] = -2.;
        let (indices, scores) = project(&input, &weights, &bias);
        assert_eq!(indices, vec![1040; rows]);
        assert_eq!(scores, vec![0x8000_0000; rows]);
        bias[4] = -0.;
    }
}

#[test]
fn portable4_compact_reference_oracle_uses_independent_literal_logits() {
    let mut weights = vec![0.; INNER * COLUMNS];
    let mut bias = vec![-2.; COLUMNS];
    for column in [15, 1008, 1040] {
        weights[column] = BIG;
        weights[COLUMNS + column] = -BIG;
        bias[column] = 1.;
    }
    for rows in [1, 4, 5, 129] {
        let actual = super::reference_logits(
            Backend::Portable4,
            &vec![1.; rows * INNER],
            &weights,
            &bias,
            rows,
            INNER,
            COLUMNS,
            true,
        );
        for row in 0..rows {
            assert_eq!(actual[row * COLUMNS + 15].to_bits(), 0x3f80_0000);
            assert_eq!(actual[row * COLUMNS + 1008].to_bits(), 0x3f80_0000);
            assert_eq!(actual[row * COLUMNS + 1040].to_bits(), 0x0000_0000);
            assert_eq!(actual[row * COLUMNS + 1].to_bits(), 0xc000_0000);
        }
    }
}
