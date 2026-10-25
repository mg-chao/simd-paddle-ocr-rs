//! FP32 four-pixel depthwise windows for ordinary Metal compute.
//!
//! Scalar depthwise shaders reload each overlapping input window and weight
//! separately. These kernels reuse a packed input window and broadcast every
//! weight across four FP32 accumulators, without a staging buffer or barrier.
/// Generate each register window explicitly so the Metal compiler sees fixed
/// lane selections and a fixed reduction order, never an indexed local array.
fn window_source(kernel: u32, stride: u32) -> String {
    let span = kernel + 3 * stride;
    let mut source = String::new();
    for ky in 0..kernel {
        source.push_str(&format!(
            "{{ const int iy = iy0 + {ky}; if (iy >= 0 && iy < int(height)) {{\n\
             device const float* source = raw_a + plane + uint(iy) * width + uint(ix0);\n"
        ));
        for window in 0..span.div_ceil(4) {
            let offset = window * 4;
            if offset + 4 <= span {
                source.push_str(&format!(
                    "const float4 v{window} = float4(*reinterpret_cast<device const packed_float4*>(source + {offset}u));\n"
                ));
            } else {
                let lanes = (0..4)
                    .map(|lane| {
                        if offset + lane < span {
                            format!("source[{}u]", offset + lane)
                        } else {
                            "0.0f".into()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                source.push_str(&format!("const float4 v{window} = float4({lanes});\n"));
            }
        }
        for kx in 0..kernel {
            let lanes = (0..4)
                .map(|pixel| {
                    let index = pixel * stride + kx;
                    format!("v{}[{}]", index / 4, index % 4)
                })
                .collect::<Vec<_>>()
                .join(", ");
            source.push_str(&format!(
                "sum = sum + float4({lanes}) * weights[{}u];\n",
                ky * kernel + kx
            ));
        }
        source.push_str("}}\n");
    }
    source
}

pub(super) fn source(kernel: u32) -> String {
    // K7 shares the same bounded windows, but is admitted only by the explicit
    // all-Half policy. This generator is not a separate F32 selector.
    assert!(matches!(kernel, 3 | 5 | 7 | 9));
    format!(
        "{}\n{}",
        include_str!("common_native.metal"),
        include_str!("depthwise_native.metal")
            .replace("$KERNEL", &kernel.to_string())
            .replace("$STRIDE1", &window_source(kernel, 1))
            .replace("$STRIDE2", &window_source(kernel, 2))
    )
}
