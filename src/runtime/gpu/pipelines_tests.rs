use super::*;

#[test]
fn retained_shader_families_validate_for_each_storage_policy() {
    for &kind in ShaderKind::ALL {
        for storage in [
            [GpuPrecision::F32; 6],
            [GpuPrecision::MixedF16; 6],
            [
                GpuPrecision::MixedF16,
                GpuPrecision::F32,
                GpuPrecision::F32,
                GpuPrecision::F32,
                GpuPrecision::F32,
                GpuPrecision::F32,
            ],
        ] {
            let source = wgsl_source(kind, storage);
            let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
                panic!("{kind:?}/{storage:?}: {}", error.emit_to_string(&source))
            });
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::back::msl::supported_capabilities(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{kind:?}/{storage:?}: {error}"));
        }
    }
}

#[test]
#[ignore = "requires physical Metal; native PSO cache identity, storage separation and foreign-device rejection, no dispatch"]
fn gpu_native_pipeline_cache_owns_device_and_arithmetic_signature() {
    let context = GpuContext::new().unwrap();
    let cache = Pipelines::shared(&context).unwrap();
    assert!(Arc::ptr_eq(&cache, &Pipelines::shared(&context).unwrap()));
    let f32_pipeline = cache
        .get(&context, ShaderKind::General, [GpuPrecision::F32; 6])
        .unwrap();
    assert!(Arc::ptr_eq(
        &f32_pipeline,
        &cache
            .get(&context, ShaderKind::General, [GpuPrecision::F32; 6])
            .unwrap()
    ));
    let half_pipeline = cache
        .get(&context, ShaderKind::General, [GpuPrecision::MixedF16; 6])
        .unwrap();
    assert!(!Arc::ptr_eq(&f32_pipeline, &half_pipeline));
    let foreign = GpuContext::new().unwrap();
    assert_eq!(
        cache
            .get(&foreign, ShaderKind::General, [GpuPrecision::F32; 6])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    cache.reject_depthwise7_for_test(RuntimeError::with_kind(
        ErrorKind::Unsupported,
        "fixture unsupported source",
    ));
    assert_eq!(
        cache
            .get_for_recipe(
                &foreign,
                ShaderKind::General,
                [GpuPrecision::F32; 6],
                &[0; 128]
            )
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput,
    );
    // Identity rejection must precede a cached unsupported result or an
    // otherwise unsupported recipe's early return, even if device IDs collide.
    assert_eq!(
        cache
            .supports_half(&foreign, ShaderKind::DepthwiseNative7)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput,
    );
    let unrelated = operators::Recipe {
        node: 0,
        nodes: vec![0],
        kind: operators::RecipeKind::Compute(ShaderKind::General),
        inputs: vec![],
        output: TensorId(0),
        words: [0; 128],
        x: 1,
        y: 1,
    };
    assert_eq!(
        cache
            .supports_matrix(&foreign, &unrelated)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput,
    );

    assert!(
        !cache
            .supports_half(&context, ShaderKind::DepthwiseNative7)
            .unwrap()
    );
    cache.reject_depthwise7_for_test(RuntimeError::with_kind(
        ErrorKind::Allocation,
        "fixture fatal admission",
    ));
    assert_eq!(
        cache
            .supports_half(&context, ShaderKind::DepthwiseNative7)
            .unwrap_err()
            .kind(),
        ErrorKind::Allocation
    );
}
