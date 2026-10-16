// Independent exact-rational pinned generic width-four softmax goldens.
// Regenerate: python tools/generate_portable_softmax_oracle.py --output-dir target/portable-softmax-oracle
// See docs/cpu-validation.md; no C# execution.
// Generator SHA-256: 19d217092810e3178eb22d8ff466e56b4b0c1a94849ae92e22e61bb7149c4cf5
struct Case {
    inputs: &'static [u32],
    expected: &'static [u32],
}
const CASES: &[Case] = &[
    Case {
        inputs: &[0x00000000, 0x00000000, 0x00000000],
        expected: &[0x3eaaaaab, 0x3eaaaaab, 0x3eaaaaab],
    },
    Case {
        inputs: &[0xbf800000, 0xc0555555, 0xc0b55555, 0x00000000],
        expected: &[0x3e85de43, 0x3ccfb413, 0x3b21219d, 0x3f35f21d],
    },
    Case {
        inputs: &[0xbf800000, 0xc0555555, 0xc0b55555, 0xc1000000, 0x00000000],
        expected: &[0x3e85d617, 0x3ccfa766, 0x3b2117c7, 0x3979f1a0, 0x3f35e702],
    },
    Case {
        inputs: &[
            0xbf800000, 0xc0555555, 0xc0b55555, 0xc1000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: &[
            0x3ddd1d8b, 0x3c2b8935, 0x3a8512ce, 0x38ce7865, 0x3e964374, 0x3e964374, 0x3e964374,
        ],
    },
    Case {
        inputs: &[
            0xbf800000, 0xc0555555, 0xc0b55555, 0xc1000000, 0x00000000, 0xc0155555, 0xc0955555,
            0x00000000,
        ],
        expected: &[
            0x3e15dc5b, 0x3c688446, 0x3ab4617e, 0x390bef5f, 0x3ecbae90, 0x3d1e02f0, 0x3b7529b8,
            0x3ecbae90,
        ],
    },
    Case {
        inputs: &[
            0xbf800000, 0xc0555555, 0xc0b55555, 0xc1000000, 0x00000000, 0xc0155555, 0xc0955555,
            0xc0e00000, 0x00000000,
        ],
        expected: &[
            0x3e15ce72, 0x3c686eb0, 0x3ab450c0, 0x390be262, 0x3ecb9ba8, 0x3d1df445, 0x3b7512f6,
            0x39be1f5b, 0x3ecb9ba8,
        ],
    },
    Case {
        inputs: &[
            0xbf800000, 0xc0555555, 0xc0b55555, 0xc1000000, 0x00000000, 0xc0155555, 0xc0955555,
            0xc0e00000, 0xc1155555, 0xbfaaaaab, 0xc06aaaab, 0x00000000,
        ],
        expected: &[
            0x3e065a3c, 0x3c50746d, 0x3aa1b6d5, 0x38fae868, 0x3eb69a9c, 0x3d0da8e3, 0x3b5bcad9,
            0x39aa8273, 0x380446f8, 0x3dc08917, 0x3c155d47, 0x3eb69a9c,
        ],
    },
    Case {
        inputs: &[
            0xbf800000, 0xc0555555, 0xc0b55555, 0xc1000000, 0x00000000, 0xc0155555, 0xc0955555,
            0xc0e00000, 0xc1155555, 0xbfaaaaab, 0xc06aaaab, 0xc0c00000, 0x00000000,
        ],
        expected: &[
            0x3e063bd9, 0x3c504548, 0x3aa19242, 0x38faafaa, 0x3eb67150, 0x3d0d88da, 0x3b5b9924,
            0x39aa5be3, 0x3804290d, 0x3dc05d8c, 0x3c153b7f, 0x3a678ab3, 0x3eb67150,
        ],
    },
    Case {
        inputs: &[
            0xbf800000, 0xc0555555, 0xc0b55555, 0xc1000000, 0x00000000, 0xc0155555, 0xc0955555,
            0xc0e00000, 0xc1155555, 0xbfaaaaab, 0xc06aaaab, 0xc0c00000, 0xc1055555, 0xbeaaaaab,
            0xc02aaaab, 0x00000000,
        ],
        expected: &[
            0x3dd1b670, 0x3c22b0a3, 0x3a7c6c03, 0x38c3d2a1, 0x3e8e83b7, 0x3cdd1e5e, 0x3b2b89d9,
            0x3985134b, 0x37ce792d, 0x3d964403, 0x3be92518, 0x3a34de41, 0x388c502d, 0x3e4c3b72,
            0x3c9e703a, 0x3e8e83b7,
        ],
    },
    Case {
        inputs: &[
            0xbf800000, 0xc0555555, 0xc0b55555, 0xc1000000, 0x00000000, 0xc0155555, 0xc0955555,
            0xc0e00000, 0xc1155555, 0xbfaaaaab, 0xc06aaaab, 0xc0c00000, 0xc1055555, 0xbeaaaaab,
            0xc02aaaab, 0xc0a00000, 0x00000000,
        ],
        expected: &[
            0x3dd151f0, 0x3c2262ab, 0x3a7bf30b, 0x38c374c9, 0x3e8e3f6b, 0x3cdcb466, 0x3b2b37a5,
            0x3984d385, 0x37ce163a, 0x3d95fc00, 0x3be8b55d, 0x3a348794, 0x388c0cef, 0x3e4bd992,
            0x3c9e244c, 0x3af55d7b, 0x3e8e3f6b,
        ],
    },
    Case {
        inputs: &[
            0xbf800000, 0xc0555555, 0xc0b55555, 0xc1000000, 0x00000000, 0xc0155555, 0xc0955555,
            0xc0e00000, 0xc1155555, 0xbfaaaaab, 0xc06aaaab, 0xc0c00000, 0xc1055555, 0xbeaaaaab,
            0xc02aaaab, 0xc0a00000, 0xc0eaaaab, 0xc11aaaab, 0xbfd55555, 0xc0800000, 0xc0caaaab,
            0xc10aaaab, 0xbf2aaaab, 0xc0400000, 0xc0aaaaab, 0xc0f55555, 0xc1200000, 0xc0000000,
            0xc08aaaab, 0xc0d55555, 0xc1100000, 0x00000000,
        ],
        expected: &[
            0x3da667ac, 0x3c0117cb, 0x3a484b5c, 0x389b622c, 0x3e622af5, 0x3caf749c, 0x3b081d38,
            0x39533025, 0x37a3d5a9, 0x3d6e77ff, 0x3bb8ff8d, 0x3a0f8465, 0x385eaca2, 0x3e220e6a,
            0x3c7b704e, 0x3ac30f60, 0x391752a5, 0x376ac8fe, 0x3d2adece, 0x3b848e9b, 0x39cdab47,
            0x381f8d94, 0x3de83ca2, 0x3c3429e8, 0x3a8bc441, 0x38d8daec, 0x37283b20, 0x3cf4de2c,
            0x3b3df668, 0x39935e4d, 0x37e4a64a, 0x3e622af5,
        ],
    },
    Case {
        inputs: &[
            0xbf800000, 0xc0555555, 0xc0b55555, 0xc1000000, 0x00000000, 0xc0155555, 0xc0955555,
            0xc0e00000, 0xc1155555, 0xbfaaaaab, 0xc06aaaab, 0xc0c00000, 0xc1055555, 0xbeaaaaab,
            0xc02aaaab, 0xc0a00000, 0xc0eaaaab, 0xc11aaaab, 0xbfd55555, 0xc0800000, 0xc0caaaab,
            0xc10aaaab, 0xbf2aaaab, 0xc0400000, 0xc0aaaaab, 0xc0f55555, 0xc1200000, 0xc0000000,
            0xc08aaaab, 0xc0d55555, 0xc1100000, 0xbf800000, 0x00000000,
        ],
        expected: &[
            0x3d99e674, 0x3beec8bb, 0x3a393e32, 0x388fb4fb, 0x3e512c0d, 0x3ca24548, 0x3afbc571,
            0x39435169, 0x379785e4, 0x3d5c8c73, 0x3bab18a5, 0x3a04bb7c, 0x384df0ee, 0x3e15e0da,
            0x3c688b3f, 0x3ab466e7, 0x390bf392, 0x3759244f, 0x3d1e07af, 0x3b753113, 0x39be36b7,
            0x3813902d, 0x3dd6c8f9, 0x3c26a002, 0x3a81437e, 0x38c88f2b, 0x371b96c7, 0x3ce27786,
            0x3b2fb000, 0x39884b4d, 0x37d377a4, 0x3d99e674, 0x3e512c0d,
        ],
    },
];
const SCALAR_UNIFORM: &[(usize, u32)] = &[
    (1, 0x3f800000),
    (3, 0x3eaaaaab),
    (4, 0x3e800000),
    (5, 0x3e4ccccd),
    (7, 0x3e124925),
    (8, 0x3e000000),
    (9, 0x3de38e39),
    (16, 0x3d800000),
    (17, 0x3d70f0f1),
    (33, 0x3cf83e10),
];

use super::*;

fn assert_golden(actual: &[f32], expected: &[u32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&value, &bits)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(value.to_bits(), bits, "element {index}");
    }
}

#[test]
fn portable_softmax_matches_independent_generic_width_four_goldens_and_safe_tails() {
    // Cases cover axis3/4/5 eligibility, complete vectors, scalar remainders,
    // lane-order summation and reciprocal multiplication. Scalar tails were
    // selected as row maxima: MathF.Exp(0)=1 exactly, independently of libm.
    for case in CASES {
        let input: Vec<_> = case.inputs.iter().copied().map(f32::from_bits).collect();
        let canary = f32::from_bits(0x7fc0_1234);
        let mut guarded = vec![canary; input.len() + 4];
        softmax(
            Backend::Portable4,
            &input,
            &mut guarded[2..input.len() + 2],
            1,
            input.len(),
        );
        assert_golden(&guarded[2..input.len() + 2], case.expected);
        for &value in guarded[..2].iter().chain(&guarded[input.len() + 2..]) {
            assert_eq!(value.to_bits(), canary.to_bits());
        }
    }
    softmax(Backend::Portable4, &[], &mut [], 0, 4);
}

#[test]
fn portable_softmax_rows_preserve_golden_bits_under_bounded_worker_partitions() {
    for budget in [1, 2, 4, 8] {
        let cpu = crate::cpu::CpuContext::new(if cfg!(feature = "parallel") {
            budget
        } else {
            1
        })
        .unwrap();
        for case in CASES {
            let input: Vec<_> = case.inputs.iter().copied().map(f32::from_bits).collect();
            let all_input = input.repeat(13);
            let mut full = vec![f32::NAN; all_input.len()];
            softmax(Backend::Portable4, &all_input, &mut full, 13, input.len());
            let mut rows = vec![vec![f32::NAN; input.len()]; 13];
            cpu.for_each_mut(&mut rows, budget, |_, output| {
                softmax(Backend::Portable4, &input, output, 1, input.len());
            });
            for (row, output) in rows.iter().enumerate() {
                assert_golden(output, case.expected);
                for (left, right) in output.iter().zip(&full[row * input.len()..][..input.len()]) {
                    assert_eq!(left.to_bits(), right.to_bits());
                }
            }
        }
    }
}

#[test]
fn scalar_softmax_uniform_contract_is_unchanged_around_generic_thresholds() {
    // Exact rational 1/axis rounded directly to F32; each input is a maximum,
    // so the independent scalar exponential is exactly MathF.Exp(0)=1.
    for &(columns, expected) in SCALAR_UNIFORM {
        let input = vec![-0.0; columns];
        let mut output = vec![f32::NAN; columns];
        softmax(Backend::Scalar, &input, &mut output, 1, columns);
        assert_golden(&output, &vec![expected; columns]);
    }
}
