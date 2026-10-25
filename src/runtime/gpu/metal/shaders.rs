//! Direct MSL catalog. Precision and geometry are native compiler definitions;
//! no runtime translation, source rewriting or implicit binding reflection.
use super::*;
use operators::ShaderKind;

pub(super) fn neural(
    context: &GpuContext,
    kind: ShaderKind,
    storage: [GpuPrecision; 6],
    words: &[u32; 128],
) -> RuntimeResult<Arc<execution::NativePipeline>> {
    let mut threads = 64;
    let (body, defines) = match kind {
        ShaderKind::PointwiseNative64 | ShaderKind::PointwiseF16 | ShaderKind::TailProjection => {
            threads = 128;
            (
                include_str!("shaders/matrix.metal"),
                format!(
                    "#define USE_MPP 1\n#define PACKED {}\n#define TOKEN_OUTPUT {}\n",
                    u8::from(kind == ShaderKind::PointwiseF16),
                    u8::from(kind == ShaderKind::TailProjection)
                ),
            )
        }
        ShaderKind::LinearNative => {
            threads = 256;
            (include_str!("shaders/linear.metal"), String::new())
        }
        ShaderKind::LinearMpp => {
            threads = 128;
            (
                include_str!("shaders/linear_matrix.metal"),
                "#define USE_MPP 1\n".into(),
            )
        }
        ShaderKind::DepthwiseNative3
        | ShaderKind::DepthwiseNative5
        | ShaderKind::DepthwiseNative7
        | ShaderKind::DepthwiseNative9
        | ShaderKind::DepthwiseNative1x7 => {
            (include_str!("shaders/depthwise.metal"), String::new())
        }
        kind if conv_mpp::Kernel::from_shader(kind).is_some()
            || conv_mpp_staged::supports_kind(kind) =>
        {
            threads = 128;
            (
                include_str!("shaders/spatial_matrix.metal"),
                format!(
                    "#define USE_MPP 1\n#define STAGED {}\n#define INPUT_CHANNELS {}\n#define INPUT_HEIGHT {}\n#define INPUT_WIDTH {}\n#define FILTER {}\n#define STRIDE_H {}\n#define STRIDE_W {}\n",
                    u8::from(conv_mpp_staged::supports_kind(kind)),
                    words[3],
                    words[5],
                    words[6],
                    words[11],
                    words[13],
                    words[14]
                ),
            )
        }
        ShaderKind::General => (include_str!("shaders/general.metal"), String::new()),
        ShaderKind::Convolution | ShaderKind::Depthwise3 | ShaderKind::Depthwise5 => {
            (include_str!("shaders/convolution.metal"), String::new())
        }
        ShaderKind::MatMul16 | ShaderKind::MatMul32 | ShaderKind::Conv16 | ShaderKind::Conv32 => (
            include_str!("shaders/tiled.metal"),
            format!(
                "#define TILE {}\n#define IS_CONV {}\n",
                if matches!(kind, ShaderKind::MatMul32 | ShaderKind::Conv32) {
                    32
                } else {
                    16
                },
                u8::from(matches!(kind, ShaderKind::Conv16 | ShaderKind::Conv32))
            ),
        ),
        ShaderKind::Reduce | ShaderKind::Softmax | ShaderKind::LayerNorm => (
            include_str!("shaders/reduce.metal"),
            format!(
                "#define MODE {}\n",
                match kind {
                    ShaderKind::Reduce => 0,
                    ShaderKind::Softmax => 1,
                    _ => 2,
                }
            ),
        ),
        ShaderKind::ReducePartial
        | ShaderKind::ReduceFinish
        | ShaderKind::SoftmaxPartial
        | ShaderKind::SoftmaxFinish => (
            include_str!("shaders/reduce_staged.metal"),
            format!(
                "#define SOFTMAX {}\n#define FINISH {}\n",
                u8::from(matches!(
                    kind,
                    ShaderKind::SoftmaxPartial | ShaderKind::SoftmaxFinish
                )),
                u8::from(matches!(
                    kind,
                    ShaderKind::ReduceFinish | ShaderKind::SoftmaxFinish
                ))
            ),
        ),
        ShaderKind::Attention | ShaderKind::AttentionViews => (
            include_str!("shaders/attention.metal"),
            format!(
                "#define VIEWS {}\n",
                u8::from(kind == ShaderKind::AttentionViews)
            ),
        ),
        ShaderKind::Transposed => (include_str!("shaders/transposed.metal"), String::new()),
        ShaderKind::SqueezeExcite | ShaderKind::SmallProjection => (
            include_str!("shaders/excitation.metal"),
            format!(
                "#define SMALL_PROJECTION {}\n",
                u8::from(kind == ShaderKind::SmallProjection)
            ),
        ),
        ShaderKind::PackNhwc => (include_str!("shaders/pack.metal"), String::new()),
        _ => return Err(unsupported(format!("Metal kernel {kind:?} is unavailable"))),
    };
    let mut source = defines;
    for (name, precision) in ["A", "B", "C", "D", "E", "O"].iter().zip(storage) {
        source.push_str(&format!(
            "#define STORAGE_{name} {}\n",
            if precision == GpuPrecision::F32 {
                "float"
            } else {
                "half"
            }
        ));
    }
    source.push_str(&format!(
        "#define HALF_OUTPUT {}\n",
        u8::from(storage[5] == GpuPrecision::MixedF16)
    ));
    source.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
    source.push_str(include_str!("shaders/half.metal"));
    source.push_str(include_str!("shaders/neural.metal"));
    source.push_str(body);
    source.push_str(include_str!("shaders/entry.metal"));
    execution::compile_constants(
        context,
        &source,
        "neural",
        [threads, 1, 1],
        vec![4, 4, 4, 4, 4, 4, 512],
        vec![true, true, true, true, true, false],
        &[(0, u32::from(storage[5] == GpuPrecision::MixedF16))],
        // The half-storage arithmetic contract uses Metal's native fast math
        // functions. Substituting precise transcendental functions can cross a
        // half-rounding boundary and alter downstream detector decisions.
        // Arithmetic remains Safe and accumulators remain F32 in both modes.
        if storage[5] == GpuPrecision::MixedF16 {
            MTLMathFloatingPointFunctions::Fast
        } else {
            MTLMathFloatingPointFunctions::Precise
        },
    )
}

pub(super) fn resident(
    context: &GpuContext,
    key: KernelKey,
    read_only: &[bool],
) -> RuntimeResult<Arc<execution::NativePipeline>> {
    #[cfg(test)]
    if matches!(
        key.id,
        KernelId::SqueezeExcite | KernelId::SmallProjection | KernelId::HalfStoreProbe
    ) {
        let storage = std::array::from_fn(|i| {
            if key.precision & (1 << i) == 0 {
                GpuPrecision::F32
            } else {
                GpuPrecision::MixedF16
            }
        });
        if key.id != KernelId::HalfStoreProbe {
            return neural(
                context,
                if key.id == KernelId::SqueezeExcite {
                    ShaderKind::SqueezeExcite
                } else {
                    ShaderKind::SmallProjection
                },
                storage,
                &[0; 128],
            );
        }
        let mut source = String::from("#define HALF_OUTPUT 1\n");
        for (name, precision) in ["A", "B", "C", "D", "E", "O"].iter().zip(storage) {
            source.push_str(&format!(
                "#define STORAGE_{name} {}\n",
                if precision == GpuPrecision::F32 {
                    "float"
                } else {
                    "half"
                }
            ));
        }
        source.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
        source.push_str(include_str!("shaders/half.metal"));
        source.push_str(include_str!("shaders/neural.metal"));
        source.push_str("void run(uint3 id,uint3,uint,threadgroup float*)const{if(id.x<p(0))store_output(id.x,load_a(id.x));}\n");
        source.push_str(include_str!("shaders/entry.metal"));
        return execution::compile_constants(
            context,
            &source,
            "neural",
            [64, 1, 1],
            vec![4, 4, 4, 4, 4, 4, 512],
            read_only.to_vec(),
            &[(0, 1)],
            MTLMathFloatingPointFunctions::Fast,
        );
    }
    let (body, entry, threads, access): (&str, &str, u32, &[bool]) = match key.id {
        KernelId::ProjectionMatrixF32 | KernelId::ProjectionMatrixF16 => (
            include_str!("shaders/projection.metal"),
            "projection",
            128,
            &[true, true, true, false],
        ),
        #[cfg(test)]
        KernelId::CounterProbe => (
            "kernel void counter(device uint* value [[buffer(0)]],constant Params& p [[buffer(1)]]){value[0]+=1u;}",
            "counter",
            1,
            &[false],
        ),
        #[cfg(test)]
        KernelId::RoundHalfProbe => (
            "kernel void round_half(device const float* input [[buffer(0)]],device float* output [[buffer(1)]],constant Params& p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p.words[0].x)output[i+p.words[0].y]=round_to_half(input[i]);}",
            "round_half",
            64,
            &[true, false],
        ),
        KernelId::Image => (
            include_str!("shaders/image.metal"),
            "image",
            64,
            &[true, true, false],
        ),
        KernelId::Crop => (
            include_str!("shaders/crop.metal"),
            "crop",
            64,
            &[true, true, false],
        ),
        KernelId::CropMeta => (
            include_str!("shaders/crop_meta.metal"),
            "crop_meta",
            64,
            &[true, false],
        ),
        KernelId::PackBgr => (
            include_str!("shaders/pack_bgr.metal"),
            "pack_bgr",
            64,
            &[true, false],
        ),
        KernelId::Rotate => (
            include_str!("shaders/rotate.metal"),
            "rotate",
            64,
            &[true, true, false],
        ),
        KernelId::DbLabel => (
            include_str!("shaders/db_label.metal"),
            "db_label",
            64,
            &[true, false, false, false, false, false, false],
        ),
        KernelId::DbRows => (
            include_str!("shaders/db_rows.metal"),
            "db_rows",
            64,
            &[true, true, true, false],
        ),
        KernelId::DbGeometry | KernelId::DbGeometryFast => (
            include_str!("shaders/db_geometry.metal"),
            "db_geometry",
            1,
            &[true, true, true, false, false],
        ),
        KernelId::DbSort => (
            include_str!("shaders/db_sort.metal"),
            "db_sort",
            1,
            &[true, false],
        ),
        KernelId::Heads => (
            include_str!("shaders/heads.metal"),
            "heads",
            64,
            &[true, true, true, false],
        ),
        KernelId::Heads4 => (
            include_str!("shaders/heads4.metal"),
            "heads4",
            64,
            &[true, true, true, false],
        ),
        KernelId::HeadsFeaturesF32 => (
            include_str!("shaders/heads_features_f32.metal"),
            "heads_features_f32",
            64,
            &[true, false],
        ),
        KernelId::HeadReduce => (
            include_str!("shaders/head_reduce.metal"),
            "head_reduce",
            64,
            &[true, false],
        ),
        KernelId::HeadReduceParallel => (
            include_str!("shaders/head_reduce_parallel.metal"),
            "head_reduce_parallel",
            64,
            &[true, false],
        ),
        KernelId::Ctc => (
            include_str!("shaders/ctc.metal"),
            "ctc",
            64,
            &[true, true, false],
        ),
        #[cfg(test)]
        _ => {
            return Err(unsupported(format!(
                "Metal resident kernel {:?} is unavailable",
                key.id
            )));
        }
    };
    if access != read_only {
        return Err(RuntimeError::with_kind(
            ErrorKind::InvalidInput,
            "Metal resident access contract differs",
        ));
    }
    let scalar = if key.precision == 0 { "float" } else { "half" };
    let mut source = format!(
        "#define INPUT {scalar}\n#define OUT {scalar}\n#define HALF_OUTPUT {}\n",
        u8::from(key.precision != 0)
    );
    source.push_str(&format!(
        "#define MATRIX_STORAGE {}\n",
        if key.id == KernelId::ProjectionMatrixF16 {
            "half"
        } else {
            "float"
        }
    ));
    source.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
    source.push_str(include_str!("shaders/half.metal"));
    source.push_str(include_str!("shaders/resident.metal"));
    source.push_str(if key.id == KernelId::DbGeometryFast {
        include_str!("shaders/precise_fast.metal")
    } else {
        include_str!("shaders/precise.metal")
    });
    source.push_str(body);
    let minimum = std::iter::repeat_n(4, access.len())
        .chain(std::iter::once(1024))
        .collect();
    execution::compile(
        context,
        &source,
        entry,
        [threads, 1, 1],
        minimum,
        access.to_vec(),
    )
}
