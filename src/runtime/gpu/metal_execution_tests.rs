use super::*;
use operators::{RecipeKind, ShaderKind};
use resident::Batch;
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
#[ignore = "requires physical Metal GPU; concurrent resident MSL shared memory, dependency and owner lifetime checks"]
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
