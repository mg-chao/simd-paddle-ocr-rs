//! Fixed native-policy goldens, independent of convolution and store math.
//! Pinned Tensor.Avx.cs revision33ff8a81672452041d110e8ba52dcd2639e54228.
//! tools/generate_unary_policy_oracle.py SHA-256:
//! bca2b142e49ba3242df149c9968151dda2cfee70f548fc5ba3005dee2db917d8.
//! Oracle JSON SHA-256:
//! 4a5d7bf375ccb694884c637eecde5614bccd512b3aebf3e8c2680aefa4a28945.
//! These literals reuse its existing fixtures; no runtime activation supplies
//! an expected result.
use super::*;
use crate::runtime::{nchw, packing::PackedMatrix};

const INPUTS: [u32; 8] = [
    0x3f230c95, 0xbf230c95, 0x3e027077, 0xbe027077, 0xc0300000, 0xbfe00000, 0xbf90a3d7, 0xbe9eb852,
];
const GELU_SEPARATE: [u32; 8] = [
    0x3ef0a177, 0xbe2aef66, 0x3d8fa931, 0xbd6a6f79, 0xbc064180, 0xbd8f926e, 0xbe158b43, 0xbdf029a7,
];
const GELU_FMA: [u32; 8] = [
    0x3ef0a176, 0xbe2aef68, 0x3d8fa931, 0xbd6a6f79, 0xbc064180, 0xbd8f926e, 0xbe158b43, 0xbdf029a7,
];

fn policies() -> Vec<(KernelPolicy, simd::Backend)> {
    [
        KernelPolicy::Portable4,
        KernelPolicy::Avx,
        KernelPolicy::AvxFma,
        KernelPolicy::Avx2Fma,
        KernelPolicy::Avx512,
    ]
    .into_iter()
    .filter_map(|policy| {
        simd::select_backend(policy)
            .ok()
            .map(|backend| (policy, backend))
    })
    .collect()
}

fn parameters(kernel: usize, width: usize) -> simd::Conv2d {
    simd::Conv2d {
        batch: 2,
        input_channels: 1,
        output_channels: 24,
        height: if kernel == 3 { 3 } else { 1 },
        width,
        output_height: if kernel == 3 { 3 } else { 1 },
        output_width: width,
        kernel_height: kernel,
        kernel_width: kernel,
        stride_height: 1,
        stride_width: 1,
        dilation_height: 1,
        dilation_width: 1,
        pad_top: kernel / 2,
        pad_left: kernel / 2,
        groups: 1,
    }
}

#[derive(Clone, Copy, Debug)]
enum Residual {
    Scalar(usize),
    Channel,
    Full,
}
impl Residual {
    fn slot(self, index: usize, plane: usize) -> usize {
        match self {
            Self::Scalar(slot) => slot,
            Self::Channel => index / plane % 24 % 8,
            Self::Full => index % 8,
        }
    }
}

#[test]
fn packed_epilogues_keep_fixed_math_and_scalar_channel_full_residual_order() {
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 4 } else { 1 }).unwrap();
    for (kernel, width, family) in [(3, 9, "nchw_packed_3x3"), (1, 49, "nchw_pointwise_packed")] {
        let p = parameters(kernel, width);
        let plane = p.output_height * p.output_width;
        let length = p.batch * p.output_channels * plane;
        let input = vec![0.; p.batch * p.input_channels * p.height * p.width];
        let inner = kernel * kernel;
        let weights = vec![0.; p.output_channels * inner];
        let packed = PackedMatrix::new(&weights, inner, p.output_channels).unwrap();
        for (_, backend) in policies() {
            assert_eq!(nchw::family(backend, &p, true), Some(family));
            let goldens = if backend.uses_fma() {
                &GELU_FMA
            } else {
                &GELU_SEPARATE
            };
            for residual in [
                Residual::Scalar(0),
                Residual::Scalar(1),
                Residual::Channel,
                Residual::Full,
            ] {
                let size = match residual {
                    Residual::Scalar(_) => 1,
                    Residual::Channel => p.output_channels,
                    Residual::Full => length,
                };
                let values: Vec<_> = (0..size)
                    .map(|slot| {
                        let index = match residual {
                            Residual::Scalar(slot) => slot,
                            _ => slot % 8,
                        };
                        f32::from_bits(INPUTS[index])
                    })
                    .collect();
                let inputs = [0usize];
                let data = |_: usize| values.as_slice();
                for add_before in [true, false] {
                    let program = if add_before {
                        [Epilogue::Add(0), Epilogue::Activation(Activation::Gelu)]
                    } else {
                        [Epilogue::Activation(Activation::Gelu), Epilogue::Add(0)]
                    };
                    // A constant graph Add could normalize into Conv bias.
                    // Bind the actual production store to retain every Add.
                    let store = store::PreparedStore::new(EpilogueStore {
                        epilogues: &program,
                        inputs: &inputs,
                        data: &data,
                        channels: p.output_channels,
                        plane,
                        layout: Layout::Canonical,
                        math: simd::UnaryMath::for_backend(backend),
                    });
                    let mut output = vec![f32::NAN; length];
                    assert!(nchw::try_conv_store(
                        backend,
                        &p,
                        &input,
                        &weights,
                        &[],
                        Some(&packed),
                        &mut output,
                        &cpu,
                        1,
                        &store
                    ));
                    for (index, actual) in output.iter().enumerate() {
                        let slot = residual.slot(index, plane);
                        // Add->GELU uses fixed goldens; GELU->Add is exactly the
                        // residual because the zero convolution gives GELU(+0).
                        let expected = if add_before {
                            goldens[slot]
                        } else {
                            INPUTS[slot]
                        };
                        assert_eq!(
                            actual.to_bits(),
                            expected,
                            "{backend:?} {residual:?} add_before={add_before} kernel={kernel} index={index}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn delta_conv_gelu_graph_reaches_packed_store_with_independent_policy_bits() {
    let (batch, height, width, channels) = (2, 3, 9, 24);
    let mut weights = vec![0.; channels * 9];
    for channel in 0..channels {
        weights[channel * 9 + 4] = 1.;
    }
    let parsed = model(
        vec![
            tensor("w", &[channels, 1, 3, 3], &weights),
            tensor("sqrt2", &[], &[std::f32::consts::SQRT_2]),
            tensor("one", &[], &[1.]),
            tensor("half", &[], &[0.5]),
            node(
                "Conv",
                &["x", "w"],
                "c",
                [attr("kernel_shape", &[3, 3]), attr("pads", &[1; 4])].concat(),
            ),
            node("Div", &["c", "sqrt2"], "v", vec![]),
            node("Erf", &["v"], "e", vec![]),
            node("Add", &["e", "one"], "a", vec![]),
            node("Mul", &["a", "c"], "m", vec![]),
            node("Mul", &["m", "half"], "y", vec![]),
        ],
        &[batch as i64, 1, height as i64, width as i64],
        &[batch as i64, channels as i64, height as i64, width as i64],
    );
    let input: Vec<_> = (0..batch * height * width)
        .map(|index| f32::from_bits(INPUTS[index % 8]))
        .collect();
    for (kernels, backend) in policies() {
        let mut request = session(
            parsed.clone(),
            CompileOptions {
                kernels,
                layout: LayoutPolicy::Nchw,
                ..CompileOptions::default()
            },
            &[batch, 1, height, width],
        );
        let goldens = if backend.uses_fma() {
            &GELU_FMA
        } else {
            &GELU_SEPARATE
        };
        let output = request.run(&input).unwrap().data().to_vec();
        for (index, value) in output.iter().enumerate() {
            let batch_index = index / (channels * height * width);
            let spatial = index % (height * width);
            assert_eq!(
                value.to_bits(),
                goldens[(batch_index * height * width + spatial) % 8],
                "{backend:?} index={index}"
            );
        }
        assert!(
            request
                .statistics()
                .specialized_kernels
                .iter()
                .any(|selection| selection.family == "nchw_packed_3x3" && selection.store_fused)
        );
        assert_eq!(request.statistics().store_fused_steps, 1);
    }
}
