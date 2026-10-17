use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn compare<const LANES: usize>(epilogues: &[Epilogue], layout: Layout, plane: usize) {
    let channels = 5;
    let length = 2 * channels * plane;
    let scalar = [0.125];
    let channel = [-0., 0., -0.75, 1.25, 0.375];
    let full: Vec<_> = (0..length).map(|i| (i % 13) as f32 / 11. - 0.5).collect();
    let operands = [scalar.as_slice(), channel.as_slice(), full.as_slice()];
    let inputs = [0, 1, 2];
    let data = |id: usize| operands[id];
    let generic = EpilogueStore {
        epilogues,
        inputs: &inputs,
        data: &data,
        channels,
        plane,
        layout,
        math: simd::UnaryMath::for_backend(simd::Backend::Scalar),
    };
    let prepared = PreparedStore::new(EpilogueStore {
        epilogues,
        inputs: &inputs,
        data: &data,
        channels,
        plane,
        layout,
        math: simd::UnaryMath::for_backend(simd::Backend::Scalar),
    });
    for index in 0..length {
        let value = (index % 17) as f32 / 7. - 1.25;
        assert_eq!(
            prepared.scalar(value, index).to_bits(),
            generic.scalar(value, index).to_bits(),
            "scalar index={index} plane={plane} layout={layout:?}"
        );
        let value = Simd::<f32, LANES>::from_array(std::array::from_fn(|lane| {
            ((index + lane) % 17) as f32 / 7. - 1.25
        }));
        let actual = prepared.vector(value, index).to_array();
        let expected = generic.vector(value, index).to_array();
        let hinted = prepared
            .vector_channel(value, index, index % channels)
            .to_array();
        for (lane, (a, b)) in actual.into_iter().zip(expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "lane={lane} index={index} plane={plane} layout={layout:?}"
            );
            assert_eq!(
                hinted[lane].to_bits(),
                b.to_bits(),
                "hinted lane={lane} index={index} plane={plane} layout={layout:?}"
            );
        }
    }
}

#[test]
fn bound_common_recipes_preserve_every_order_broadcast_tail_and_batch_crossing() {
    let mut recipes = vec![vec![]];
    for activation in [
        Activation::Relu,
        Activation::HardSwish,
        Activation::Gelu,
        Activation::Sigmoid,
    ] {
        recipes.push(vec![Epilogue::Activation(activation)]);
        for a in 0..3 {
            recipes.push(vec![Epilogue::Add(a)]);
            recipes.push(vec![Epilogue::Add(a), Epilogue::Activation(activation)]);
            recipes.push(vec![Epilogue::Activation(activation), Epilogue::Add(a)]);
            for b in 0..3 {
                recipes.push(vec![Epilogue::Add(a), Epilogue::Add(b)]);
                recipes.push(vec![
                    Epilogue::Add(a),
                    Epilogue::Add(b),
                    Epilogue::Activation(activation),
                ]);
                recipes.push(vec![
                    Epilogue::Add(a),
                    Epilogue::Activation(activation),
                    Epilogue::Add(b),
                ]);
                recipes.push(vec![
                    Epilogue::Activation(activation),
                    Epilogue::Add(a),
                    Epilogue::Add(b),
                ]);
            }
        }
    }
    for layout in [Layout::Canonical, Layout::Nhwc] {
        for plane in [1, 3, 11] {
            for recipe in &recipes {
                compare::<1>(recipe, layout, plane);
                compare::<4>(recipe, layout, plane);
                compare::<8>(recipe, layout, plane);
                compare::<16>(recipe, layout, plane);
            }
        }
    }
}

#[test]
fn bound_operands_are_resolved_once_and_long_programs_keep_the_generic_path() {
    let scalar = [0.25];
    let channel = [-0., 0., 0.125, -0.5, 0.75];
    let full = [0.03125; 30];
    let operands = [scalar.as_slice(), channel.as_slice(), full.as_slice()];
    let reads = AtomicUsize::new(0);
    let data = |id: usize| {
        reads.fetch_add(1, Ordering::Relaxed);
        operands[id]
    };
    let common = [
        Epilogue::Add(1),
        Epilogue::Add(2),
        Epilogue::Activation(Activation::Gelu),
    ];
    let inputs = [0, 1, 2];
    let store = PreparedStore::new(EpilogueStore {
        epilogues: &common,
        inputs: &inputs,
        data: &data,
        channels: 5,
        plane: 3,
        layout: Layout::Nhwc,
        math: simd::UnaryMath::for_backend(simd::Backend::Scalar),
    });
    assert!(store.common.is_some());
    assert_eq!(reads.load(Ordering::Relaxed), 2);
    for index in 0..30 {
        let _ = store.scalar(0.5, index);
        let _ = store.vector(Simd::<f32, 8>::splat(0.5), index);
        let _ = store.vector_channel(Simd::<f32, 8>::splat(0.5), index, index % 5);
    }
    assert_eq!(reads.load(Ordering::Relaxed), 2);

    // No cap is imposed on the original interpreter. Binding must neither
    // truncate this longer program nor resolve only a prefix's operands.
    let longer = [
        Epilogue::Add(0),
        Epilogue::Activation(Activation::Relu),
        Epilogue::Add(2),
        Epilogue::Activation(Activation::HardSwish),
        Epilogue::Add(1),
        Epilogue::Activation(Activation::Gelu),
        Epilogue::Add(0),
        Epilogue::Activation(Activation::Sigmoid),
    ];
    reads.store(0, Ordering::Relaxed);
    let store = PreparedStore::new(EpilogueStore {
        epilogues: &longer,
        inputs: &inputs,
        data: &data,
        channels: 5,
        plane: 3,
        layout: Layout::Canonical,
        math: simd::UnaryMath::for_backend(simd::Backend::Scalar),
    });
    assert!(store.common.is_none());
    assert_eq!(reads.load(Ordering::Relaxed), 0);
    let _ = store.vector(Simd::<f32, 8>::splat(-0.25), 4);
    assert_eq!(reads.load(Ordering::Relaxed), 4);
    compare::<1>(&longer, Layout::Canonical, 3);
    compare::<4>(&longer, Layout::Canonical, 3);
    compare::<8>(&longer, Layout::Nhwc, 3);
    compare::<16>(&longer, Layout::Nhwc, 3);
    let three_adds = [Epilogue::Add(0), Epilogue::Add(1), Epilogue::Add(2)];
    compare::<8>(&three_adds, Layout::Canonical, 3);
}

#[test]
fn bound_recipes_rebind_current_call_views_and_preserve_noncommuting_order() {
    let epilogues = [Epilogue::Activation(Activation::Relu), Epilogue::Add(0)];
    let inputs = [0];
    let mut scalar = [-0.5];
    for expected in [-0.5, -1.25] {
        scalar[0] = expected;
        let data = |_: usize| scalar.as_slice();
        let store = PreparedStore::new(EpilogueStore {
            epilogues: &epilogues,
            inputs: &inputs,
            data: &data,
            channels: 1,
            plane: 1,
            layout: Layout::Nhwc,
            math: simd::UnaryMath::for_backend(simd::Backend::Scalar),
        });
        // Relu(-1)+negative differs from Relu(-1+negative): retain ordering.
        assert_eq!(store.scalar(-1., 0), expected);
        assert_eq!(
            store.vector(Simd::<f32, 8>::splat(-1.), 0).to_array(),
            [expected; 8]
        );
    }
}

fn check_math_policy<const LANES: usize>(
    backend: simd::Backend,
    activation: Activation,
    inputs_bits: &[u32; 8],
    expected_bits: &[u32; 8],
) {
    let zero = [0.0];
    let data = |_: usize| zero.as_slice();
    let inputs = [0];
    let programs = [
        vec![Epilogue::Activation(activation)],
        vec![Epilogue::Add(0), Epilogue::Activation(activation)],
        vec![Epilogue::Activation(activation), Epilogue::Add(0)],
        vec![
            Epilogue::Add(0),
            Epilogue::Activation(activation),
            Epilogue::Add(0),
        ],
        vec![
            Epilogue::Add(0),
            Epilogue::Add(0),
            Epilogue::Activation(activation),
        ],
        vec![
            Epilogue::Activation(activation),
            Epilogue::Add(0),
            Epilogue::Add(0),
        ],
        // Three Adds retain the generic interpreter rather than a common recipe.
        vec![
            Epilogue::Activation(activation),
            Epilogue::Add(0),
            Epilogue::Add(0),
            Epilogue::Add(0),
        ],
    ];
    for (program_index, epilogues) in programs.iter().enumerate() {
        let generic = EpilogueStore {
            epilogues,
            inputs: &inputs,
            data: &data,
            channels: 1,
            plane: 8,
            layout: Layout::Nhwc,
            math: simd::UnaryMath::for_backend(backend),
        };
        let prepared = PreparedStore::new(EpilogueStore { ..generic });
        assert_eq!(prepared.common.is_some(), program_index < 6);
        for (input, expected) in inputs_bits.iter().zip(expected_bits) {
            assert_eq!(
                prepared.scalar(f32::from_bits(*input), 0).to_bits(),
                *expected
            );
            assert_eq!(
                generic.scalar(f32::from_bits(*input), 0).to_bits(),
                *expected
            );
        }
        let value = Simd::<f32, LANES>::from_array(std::array::from_fn(|lane| {
            f32::from_bits(inputs_bits[lane % 8])
        }));
        for actual in [
            prepared.vector(value, 0),
            prepared.vector_channel(value, 0, 0),
            generic.vector(value, 0),
        ] {
            for (lane, value) in actual.to_array().into_iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    expected_bits[lane % 8],
                    "{backend:?} lane={lane}"
                );
            }
        }
    }
}

#[test]
fn selected_activation_math_reaches_bound_generic_and_tail_stores() {
    // Independent binary32 goldens from the pinned reference's coefficients and
    // expression order, also retained in simd/unary_policy_tests.rs. None of the
    // expected bits are computed by either StoreTransform implementation.
    let gelu_inputs = [
        0x3e027077, 0xbe027077, 0xc0300000, 0xbfe00000, 0xbf90a3d7, 0xbe9eb852, 0x3e9eb852,
        0x3f3d70a4,
    ];
    let gelu_expected = [
        0x3d8fa931, 0xbd6a6f79, 0xbc064180, 0xbd8f926e, 0xbe158b43, 0xbdf029a7, 0x3e455bd1,
        0x3f11ef64,
    ];
    let sigmoid_inputs = [
        0xc1880000, 0xc1800000, 0xc0533333, 0xbdcccccd, 0x3dcccccd, 0x40200000, 0x41000000,
        0x41a00000,
    ];
    let sigmoid_expected = [
        0x00000000, 0x34000000, 0x3d11b310, 0x3ef335ee, 0x3f066509, 0x3f6c948f, 0x3f7fea06,
        0x3f800000,
    ];
    let gelu_fma_inputs = [
        0x3f230c95, 0xbf230c95, 0x3f230c95, 0xbf230c95, 0x3f230c95, 0xbf230c95, 0x3f230c95,
        0xbf230c95,
    ];
    for policy in [
        simd::KernelPolicy::Portable4,
        simd::KernelPolicy::Avx,
        simd::KernelPolicy::AvxFma,
        simd::KernelPolicy::Avx2Fma,
        simd::KernelPolicy::Avx512,
    ] {
        let Ok(backend) = simd::select_backend(policy) else {
            continue;
        };
        let (positive, negative) = if backend.uses_fma() {
            (0x3ef0a176, 0xbe2aef68)
        } else {
            (0x3ef0a177, 0xbe2aef66)
        };
        let gelu_fma_expected = [
            positive, negative, positive, negative, positive, negative, positive, negative,
        ];
        for (activation, inputs, expected) in [
            (Activation::Gelu, &gelu_inputs, &gelu_expected),
            (Activation::Gelu, &gelu_fma_inputs, &gelu_fma_expected),
            (Activation::Sigmoid, &sigmoid_inputs, &sigmoid_expected),
        ] {
            check_math_policy::<1>(backend, activation, inputs, expected);
            check_math_policy::<4>(backend, activation, inputs, expected);
            check_math_policy::<8>(backend, activation, inputs, expected);
            check_math_policy::<16>(backend, activation, inputs, expected);
        }
    }
}
