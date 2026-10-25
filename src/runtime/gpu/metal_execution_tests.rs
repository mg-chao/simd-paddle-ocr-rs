//! Production resident graph execution through the concurrent Metal encoder.
use super::*;
use operators::{RecipeKind, ShaderKind};
use resident::Batch;

#[test]
#[ignore = "requires physical Metal GPU; native WGSL/MSL ABI, parameter snapshots and unsupported-source rejection"]
fn gpu_native_resident_kernels_and_compile_rejection() {
    const WGSL: &str = r#"
struct Params { words: array<vec4<u32>, 64> }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < params.words[0].x {
        output[gid.x] = input[gid.x] + f32(params.words[0].y);
    }
}"#;
    const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Params { uint4 words[64]; };
kernel void resident_three_operands(device const float* input [[buffer(0)]],
    device const float* original [[buffer(1)]], device float* output [[buffer(2)]],
    constant Params& params [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < params.words[0].x)
        output[i] = 2.0f * input[i] + original[i] + float(params.words[0].y);
}"#;
    // Overrides require explicit specialization. Reject them before recording;
    // every admitted operation still executes on the sole native executor.
    let fallback = WGSL.replace("@compute", "override unused_scale: f32 = 1.0;\n@compute");
    let context = context();
    let mut batch = Batch::new(context.clone()).unwrap();
    let count = 67usize;
    let bytes = count * 4;
    let result = batch
        .storage("resident ABI combined", (count * 2 + 7) * 4)
        .unwrap();
    let mut expected = Vec::new();
    for graph in 0..2 {
        let values = dyadic(count, graph + 2);
        let input = batch.storage("resident ABI input", bytes).unwrap();
        context
            .upload_resident(&input, bytemuck::cast_slice(&values))
            .unwrap();
        let first = batch.storage("resident ABI first", bytes).unwrap();
        let second = batch.storage("resident ABI second", bytes).unwrap();
        let third = batch.storage("resident ABI third", bytes).unwrap();
        batch
            .dispatch(
                "resident_abi_wgsl",
                WGSL,
                &[(&input, true), (&first, false)],
                &[count as u32, graph as u32 + 3],
                [2, 1, 1],
            )
            .unwrap();
        batch
            .dispatch_native(
                "resident_abi_msl",
                resident::NativeShader {
                    source: MSL,
                    entry_point: "resident_three_operands",
                    workgroup_size: [64, 1, 1],
                },
                &[(&first, true), (&input, true), (&second, false)],
                &[count as u32, graph as u32 + 7],
                [2, 1, 1],
            )
            .unwrap();
        batch
            .dispatch(
                "resident_abi_wgsl",
                WGSL,
                &[(&second, true), (&third, false)],
                &[count as u32, graph as u32 + 11],
                [2, 1, 1],
            )
            .unwrap();
        let output = if graph == 0 {
            let extra = batch.storage("resident ABI extra", bytes).unwrap();
            let error = batch
                .dispatch(
                    "resident_abi_override",
                    &fallback,
                    &[(&third, true), (&extra, false)],
                    &[count as u32, 13],
                    [2, 1, 1],
                )
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            batch
                .dispatch(
                    "resident_abi_extra",
                    WGSL,
                    &[(&third, true), (&extra, false)],
                    &[count as u32, 13],
                    [2, 1, 1],
                )
                .unwrap();
            extra
        } else {
            third
        };
        expected.push(
            values
                .iter()
                .map(|x| {
                    3.0 * x
                        + 2.0 * (graph + 3) as f32
                        + (graph + 7) as f32
                        + (graph + 11) as f32
                        + if graph == 0 { 13.0 } else { 0.0 }
                })
                .collect::<Vec<_>>(),
        );
        batch.copy(
            &output,
            0,
            &result,
            ((graph * count + 3) * 4) as u64,
            bytes as u64,
        );
    }
    assert_eq!(batch.statistics.native_graph_dispatches, 0);
    assert_eq!(batch.statistics.native_resident_dispatches, 7);
    assert_eq!(batch.statistics.dispatches, 7);
    // Only queued owners remain; the completion callback must retain them
    // through native chunks and the final partial copies.
    let (actual, statistics) = batch.read(&result, result.bytes(), false).unwrap();
    let values = bytemuck::cast_slice::<u8, f32>(&actual);
    assert!(values[..3].iter().all(|&x| x == 0.0));
    assert_eq!(&values[3..3 + count], expected[0]);
    assert_eq!(&values[3 + count..3 + 2 * count], expected[1]);
    assert!(values[3 + 2 * count..].iter().all(|&x| x == 0.0));
    assert_eq!(statistics.native_resident_dispatches, 7);
    assert_eq!(statistics.diagnostic_readback_bytes, 0);

    // Profiling executes the identical native kernel and reports unavailable
    // device timestamps rather than selecting another shader implementation.
    let mut profiled = Batch::new(context.clone()).unwrap();
    profiled.profiling = true;
    let input = profiled.storage("profiled ABI input", 4).unwrap();
    let output = profiled.storage("profiled ABI output", 4).unwrap();
    context
        .upload_resident(&input, bytemuck::cast_slice(&[2.0f32]))
        .unwrap();
    profiled
        .dispatch(
            "resident_abi_wgsl",
            WGSL,
            &[(&input, true), (&output, false)],
            &[1, 3],
            [1, 1, 1],
        )
        .unwrap();
    assert_eq!(profiled.statistics.native_resident_dispatches, 1);
    let (actual, stats) = profiled.read(&output, 4, false).unwrap();
    assert_eq!(stats.profiling_readback_bytes, 0);
    assert!(
        stats
            .profile
            .unwrap()
            .kernels
            .iter()
            .all(|k| k.device_ns.is_none() && k.granularity == GpuTimingGranularity::Unavailable)
    );
    assert_eq!(f32::from_le_bytes(actual.try_into().unwrap()), 5.0);
}

#[test]
#[ignore = "requires physical Metal GPU; native binding validation, ownership and immutable source layouts"]
fn gpu_native_resident_binding_validation_and_context() {
    const WGSL: &str = r#"
struct Params { words: array<vec4<u32>, 64> }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < params.words[0].x {
        output[gid.x] = input[gid.x] + f32(params.words[0].y);
    }
}"#;
    const NAME: &str = "resident_native_binding_validation";
    let context = context();
    let mut batch = Batch::new(context.clone()).unwrap();
    let input = batch.storage("native validated input", 4).unwrap();
    let output = batch.storage("native validated output", 4).unwrap();
    context
        .upload_resident(&input, bytemuck::cast_slice(&[2.0f32]))
        .unwrap();
    // Prime the compiled shader/layout cache and prove native admission. This
    // operation retains its operands without constructing a WGPU bind group.
    batch
        .dispatch(
            NAME,
            WGSL,
            &[(&input, true), (&output, false)],
            &[1, 3],
            [1, 1, 1],
        )
        .unwrap();
    assert_eq!(batch.statistics.dispatches, 1);
    assert_eq!(batch.statistics.native_resident_dispatches, 1);

    let other = GpuContext::new_with_options(context.options()).unwrap();
    let foreign = other
        .create_buffer("foreign native input", 4, wgpu::BufferUsages::STORAGE)
        .unwrap();
    let copy_only = context
        .create_buffer(
            "native input missing STORAGE",
            4,
            wgpu::BufferUsages::COPY_SRC,
        )
        .unwrap();
    let unaligned = context
        .create_buffer("unaligned native storage", 3, wgpu::BufferUsages::STORAGE)
        .unwrap();
    for (invalid, kind) in [
        (&foreign, ErrorKind::InvalidInput),
        (&copy_only, ErrorKind::InvalidInput),
        (&unaligned, ErrorKind::ResourceLimit),
    ] {
        let error = batch
            .dispatch(
                NAME,
                WGSL,
                &[(invalid, true), (&output, false)],
                &[1, 3],
                [1, 1, 1],
            )
            .unwrap_err();
        assert_eq!(error.kind(), kind);
        assert_eq!(batch.statistics.dispatches, 1);
        assert_eq!(batch.statistics.native_resident_dispatches, 1);
    }
    // Cache hits may not silently substitute a different read/write signature
    // or operand count: scheduling depends on this same immutable signature.
    for buffers in [
        vec![(&input, false), (&output, false)],
        vec![(&input, true)],
    ] {
        let error = batch
            .dispatch(NAME, WGSL, &buffers, &[1, 3], [1, 1, 1])
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(batch.statistics.dispatches, 1);
        assert_eq!(batch.statistics.native_resident_dispatches, 1);
    }

    // Every buffer has a mandatory native handle; foreign contexts above are
    // errors, never a route to a second executor.
    drop(input);
    let (actual, statistics) = batch.read(&output, 4, false).unwrap();
    assert_eq!(f32::from_le_bytes(actual.try_into().unwrap()), 5.0);
    assert_eq!(statistics.native_resident_dispatches, 1);
}

#[test]
#[ignore = "requires physical Metal GPU; reject undersized WGSL bindings and conflicting aliases before recording"]
fn gpu_native_resident_reflected_sizes_and_alias_validation() {
    let context = context();
    let mut batch = Batch::new(context).unwrap();
    // Valid fixtures prime the native pipeline and require the exact reflected
    // minimum. Invalid fixtures must never be recorded or submitted. Include
    // padded vec3 array strides and a runtime-array struct's first element.
    let cases = [
        (
            "resident_native_minimum_fixed_array",
            "@group(0) @binding(0) var<storage, read> input: array<f32>;\n\
             @group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>,4>;",
            "output[3] = vec4<f32>(input[0] + f32(params.words[0].x));",
            4,
            64,
            1,
            4,
            64,
        ),
        (
            "resident_native_minimum_fixed_struct",
            "struct Data { header: vec4<f32>, body: array<vec4<f32>,4> };\n\
             @group(0) @binding(0) var<storage, read> input: Data;\n\
             @group(0) @binding(1) var<storage, read_write> output: array<f32>;",
            "output[0] = input.header.x + input.body[3].x + f32(params.words[0].x);",
            80,
            4,
            0,
            64,
            80,
        ),
        (
            "resident_native_minimum_runtime_struct",
            "struct Data { header: vec4<f32>, body: array<vec4<f32>> };\n\
             @group(0) @binding(0) var<storage, read> input: Data;\n\
             @group(0) @binding(1) var<storage, read_write> output: array<f32>;",
            "output[0] = input.header.x + input.body[0].x + f32(params.words[0].x);",
            32,
            4,
            0,
            16,
            32,
        ),
        (
            "resident_native_minimum_vec3_array",
            "@group(0) @binding(0) var<storage, read> input: array<vec3<f32>,3>;\n\
             @group(0) @binding(1) var<storage, read_write> output: array<f32>;",
            "output[0] = input[2].z + f32(params.words[0].x);",
            48,
            4,
            0,
            36,
            48,
        ),
    ];
    for (name, declarations, body, input_bytes, output_bytes, bad_binding, bad_bytes, minimum) in
        cases
    {
        let source = format!(
            "{}{declarations}\n\
             @group(0) @binding(2) var<uniform> params: Params;\n\
             @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane: u32) {{ if lane == 0u {{ {body} }} }}",
            resident::PARAMETERS
        );
        let input = batch
            .storage("minimum-sized native input", input_bytes)
            .unwrap();
        let output = batch
            .storage("minimum-sized native output", output_bytes)
            .unwrap();
        let before = batch.statistics.native_resident_dispatches;
        batch
            .dispatch(
                name,
                &source,
                &[(&input, true), (&output, false)],
                &[1],
                [1, 1, 1],
            )
            .unwrap();
        assert_eq!(batch.statistics.native_resident_dispatches, before + 1);
        let count = batch.statistics.dispatches;
        let invalid = batch
            .storage("undersized native binding", bad_bytes)
            .unwrap();
        let operands = if bad_binding == 0 {
            [(&invalid, true), (&output, false)]
        } else {
            [(&input, true), (&invalid, false)]
        };
        let error = batch
            .dispatch(name, &source, &operands, &[1], [1, 1, 1])
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(error.to_string().contains(&format!("at least {minimum}")));
        assert_eq!(batch.statistics.dispatches, count);
        assert_eq!(batch.statistics.native_resident_dispatches, before + 1);
    }

    let input = batch.storage("native alias input", 4).unwrap();
    let output = batch.storage("native alias output", 8).unwrap();
    let count = batch.statistics.dispatches;
    let native_count = batch.statistics.native_resident_dispatches;
    let oversized_uniform = r#"
struct Params { words: array<vec4<u32>,65> }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane: u32) {
    if lane == 0u { output[0] = input[0] + f32(params.words[64].x); }
}"#;
    let error = batch
        .dispatch(
            "resident_native_minimum_oversized_uniform",
            oversized_uniform,
            &[(&input, true), (&output, false)],
            &[1],
            [1, 1, 1],
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(error.to_string().contains("at least 1040"));
    assert_eq!(batch.statistics.dispatches, count);
    assert_eq!(batch.statistics.native_resident_dispatches, native_count);

    let alias_source = format!(
        "{}\n\
         @group(0) @binding(0) var<storage, read> input: array<f32>;\n\
         @group(0) @binding(1) var<storage, read_write> output: array<f32>;\n\
         @group(0) @binding(2) var<uniform> params: Params;\n\
         @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane: u32) {{ if lane == 0u {{ output[0] = input[0] + f32(params.words[0].x); }} }}",
        resident::PARAMETERS
    );
    let error = batch
        .dispatch(
            "resident_native_conflicting_alias",
            &alias_source,
            &[(&input, true), (&input, false)],
            &[1],
            [1, 1, 1],
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(error.to_string().contains("conflicting read/write aliases"));
    assert_eq!(batch.statistics.dispatches, count);
    assert_eq!(batch.statistics.native_resident_dispatches, native_count);

    for read_only in [true, false] {
        let (name, second, body) = if read_only {
            (
                "resident_native_readonly_alias_allowed",
                "@group(0) @binding(1) var<storage, read> second: array<f32>;",
                "output[0] = input[0] + second[0] + f32(params.words[0].x);",
            )
        } else {
            (
                "resident_native_writable_alias_allowed",
                "@group(0) @binding(1) var<storage, read_write> second: array<f32>;",
                "second[0] = input[0]; output[1] = input[0] + f32(params.words[0].x);",
            )
        };
        let source = format!(
            "{}\n\
             @group(0) @binding(0) var<storage, read> input: array<f32>;\n\
             {second}\n\
             @group(0) @binding(2) var<storage, read_write> output: array<f32>;\n\
             @group(0) @binding(3) var<uniform> params: Params;\n\
             @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane: u32) {{ if lane == 0u {{ {body} }} }}",
            resident::PARAMETERS
        );
        let duplicate = if read_only { &input } else { &output };
        batch
            .dispatch(
                name,
                &source,
                &[(&input, true), (duplicate, read_only), (&output, false)],
                &[1],
                [1, 1, 1],
            )
            .unwrap();
    }
    let inactive = format!(
        "{}\n\
         @group(0) @binding(0) var<storage, read> unused: array<vec4<f32>,4>;\n\
         @group(0) @binding(1) var<storage, read_write> output: array<f32>;\n\
         @group(0) @binding(2) var<uniform> params: Params;\n\
         @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane: u32) {{ if lane == 0u {{ output[0] = f32(params.words[0].x); }} }}",
        resident::PARAMETERS
    );
    batch
        .dispatch(
            "resident_native_inactive_fixed_array_allowed",
            &inactive,
            &[(&input, true), (&output, false)],
            &[1],
            [1, 1, 1],
        )
        .unwrap();
    assert_eq!(batch.statistics.dispatches, count + 3);
    assert_eq!(
        batch.statistics.native_resident_dispatches,
        native_count + 3
    );
    // No read/submit: invalid work was rejected on the host before entering the
    // schedule, and the positive admissions require no GPU execution here.
}

fn context() -> Arc<GpuContext> {
    static CONTEXT: OnceLock<Arc<GpuContext>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            GpuContext::new_with_options(GpuContextOptions {
                recognition_wave_size: 4,
                ..Default::default()
            })
            .unwrap()
        })
        .clone()
}

fn dyadic(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| (((index * 7 + seed * 11) % 15) as i32 - 7) as f32 / 32.0)
        .collect()
}

fn compile(graph: Arc<Model>, precision: GpuMatrixPrecision) -> Arc<GpuCompiledModel> {
    Arc::new(
        GpuCompiledModel::compile_with_options(
            graph,
            context(),
            CompileOptions {
                gpu: GpuCompileOptions {
                    precision: GpuPrecision::F32,
                    matrix_precision: precision,
                },
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

fn two_resident_graphs(
    graph: Arc<Model>,
    shape: &[usize],
    matrix_precision: GpuMatrixPrecision,
    verify: impl Fn(&GpuSession),
) {
    let context = context();
    let compiled = compile(graph.clone(), matrix_precision);
    let inputs = [
        dyadic(shape.iter().product(), 1),
        dyadic(shape.iter().product(), 3),
    ];
    let expected: Vec<_> = inputs
        .iter()
        .map(|input| reference(graph.clone(), shape, input).1)
        .collect();
    let mut sessions = inputs
        .iter()
        .map(|input| {
            let session =
                GpuSession::new_resident(compiled.clone(), shape, ExecutionLimits::default(), None)
                    .unwrap();
            verify(&session);
            context
                .upload_resident(
                    &session.resident_input().buffer,
                    bytemuck::cast_slice(input),
                )
                .unwrap();
            session
        })
        .collect::<Vec<_>>();

    // Fresh native allocations are initialized. Untouched gaps prove that
    // partial blits and graph lifetimes do not corrupt neighboring output.
    let prefix = 4;
    let gap = 8;
    let suffix = 4;
    let floats = prefix + expected[0].len() + gap + expected[1].len() + suffix;
    let bytes = floats * 4;
    let mut batch = Batch::new(context).unwrap();
    let combined = batch
        .storage("native graph combined result", bytes)
        .unwrap();
    let outputs = sessions
        .iter_mut()
        .map(|session| batch.graph(session, "native-test").unwrap())
        .collect::<Vec<_>>();
    let second = prefix + expected[0].len() + gap;
    batch.copy(
        &outputs[0].buffer,
        0,
        &combined,
        (prefix * 4) as u64,
        (expected[0].len() * 4) as u64,
    );
    batch.copy(
        &outputs[1].buffer,
        0,
        &combined,
        (second * 4) as u64,
        (expected[1].len() * 4) as u64,
    );
    assert!(batch.statistics.native_graph_dispatches > 0);
    assert_eq!(
        batch.statistics.native_graph_dispatches, batch.statistics.dispatches,
        "these fixtures must qualify every graph dispatch for actual native execution"
    );
    // The submitted batch must own every constant, activation, dynamic uniform
    // and packed operand after its originating sessions/model owners disappear.
    drop((outputs, sessions, compiled, graph));
    let (actual, statistics) = batch.read(&combined, bytes, false).unwrap();
    let actual = actual
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_close(
        "native first resident graph",
        &actual[prefix..prefix + expected[0].len()],
        &expected[0],
    );
    assert_close(
        "native second resident graph",
        &actual[second..second + expected[1].len()],
        &expected[1],
    );
    assert!(actual[..prefix].iter().all(|&value| value == 0.0));
    assert!(
        actual[prefix + expected[0].len()..second]
            .iter()
            .all(|&value| value == 0.0)
    );
    assert!(
        actual[second + expected[1].len()..]
            .iter()
            .all(|&value| value == 0.0)
    );
    assert_eq!(statistics.native_graph_dispatches, statistics.dispatches);
    assert_eq!(statistics.diagnostic_readback_bytes, 0);
}

#[test]
#[ignore = "requires physical Metal GPU; concurrent resident WGSL shared memory, dependency and owner lifetime checks"]
fn gpu_native_graph_shared_convolution_reduce_softmax_and_lifetime() {
    let shape = [1, 16, 33, 35];
    let graph = model(
        vec![
            tensor("w", &[64, 16, 3, 3], &dyadic(64 * 16 * 3 * 3, 7)),
            tensor("bias", &[64], &dyadic(64, 11)),
            node(
                "Conv",
                &["x", "w", "bias"],
                "conv",
                &[integers("kernel_shape", &[3; 2]), integers("pads", &[1; 4])],
            ),
            node("Relu", &["conv"], "active", &[]),
            node(
                "ReduceMean",
                &["active"],
                "mean",
                &[integers("axes", &[2, 3]), integer("keepdims", 0)],
            ),
            node("Softmax", &["mean"], "y", &[integer("axis", 1)]),
        ],
        &shape,
        &[1, 64],
    );
    two_resident_graphs(graph, &shape, GpuMatrixPrecision::F32, |session| {
        assert!(session.recipes.iter().any(|recipe| matches!(
            recipe.kind,
            RecipeKind::Compute(ShaderKind::Conv16 | ShaderKind::Conv32)
        )));
        assert!(
            session
                .recipes
                .iter()
                .any(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::Reduce)))
        );
        assert!(
            session
                .recipes
                .iter()
                .any(|recipe| matches!(recipe.kind, RecipeKind::Compute(ShaderKind::Softmax)))
        );
        assert!(
            session
                .dispatches
                .iter()
                .any(|dispatch| matches!(&dispatch.command,
            Command::Compute { native, .. } if native.offsets()[6] != 0))
        );
    });
}

#[test]
#[ignore = "requires physical Metal tensor GPU; concurrent resident FP16 pointwise and canonical output offsets"]
fn gpu_native_graph_f16_pointwise_residual_concat_and_lifetime() {
    let shape = [1, 17, 1, 67];
    let channels = 73;
    let graph = model(
        vec![
            tensor("w1", &[channels, 17, 1, 1], &dyadic(channels * 17, 7)),
            tensor("w2", &[channels, 17, 1, 1], &dyadic(channels * 17, 17)),
            tensor("bias", &[channels], &dyadic(channels, 11)),
            tensor("residual", &[1, channels, 1, 1], &dyadic(channels, 13)),
            node(
                "Conv",
                &["x", "w1", "bias"],
                "a0",
                &[integers("kernel_shape", &[1; 2])],
            ),
            node("Add", &["a0", "residual"], "a1", &[]),
            node("Relu", &["a1"], "a", &[]),
            node(
                "Conv",
                &["x", "w2", "bias"],
                "b0",
                &[integers("kernel_shape", &[1; 2])],
            ),
            node("Relu", &["b0"], "b", &[]),
            node("Concat", &["a", "b"], "y", &[integer("axis", 1)]),
        ],
        &shape,
        &[1, channels * 2, 1, 67],
    );
    two_resident_graphs(graph, &shape, GpuMatrixPrecision::F16, |session| {
        assert_eq!(
            session
                .recipes
                .iter()
                .filter(|recipe| matches!(
                    recipe.kind,
                    RecipeKind::Compute(ShaderKind::PointwiseF16)
                ))
                .count(),
            2
        );
        assert!(
            session
                .recipes
                .iter()
                .any(|recipe| recipe.words[60] == (channels * 67) as u32)
        );
        assert!(
            session
                .dispatches
                .iter()
                .any(|dispatch| matches!(&dispatch.command,
            Command::Compute { native, .. } if native.offsets()[6] != 0))
        );
    });
}
