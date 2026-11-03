use super::super::resident::Batch;
use super::*;

#[test]
#[ignore = "requires physical Apple Silicon; production shader resource inventory"]
fn gpu_metal_production_library_excludes_test_kernels() {
    let context = GpuContext::new().unwrap();
    let owner = &context.backend.owner;
    for (library, test) in [
        (owner.library().unwrap(), false),
        (owner.test_library().unwrap(), true),
    ] {
        let mut names = library
            .functionNames()
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();
        names.sort();
        let mut expected = SHADER_LIBRARY_INVENTORY
            .iter()
            .filter(|(_, probe, _)| *probe == test)
            .map(|(name, _, _)| *name)
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(names, expected);
    }
    let script: serde_json::Value =
        serde_json::from_str(include_str!(concat!(env!("OUT_DIR"), "/ocr.mtlp-json"))).unwrap();
    let mut native = script["pipelines"]["compute_pipelines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pipeline| {
            pipeline["compute_function"]
                .as_str()
                .unwrap()
                .strip_prefix("alias:OCR#")
                .unwrap()
        })
        .collect::<Vec<_>>();
    native.sort();
    let mut expected = SHADER_LIBRARY_INVENTORY
        .iter()
        .filter(|(_, _, native)| *native)
        .map(|(name, _, _)| *name)
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(native, expected);
    assert_eq!(
        SHADER_LIBRARY_INVENTORY
            .iter()
            .filter(|(_, test, _)| *test)
            .map(|(name, _, _)| *name)
            .collect::<Vec<_>>(),
        [
            "ocr_CounterProbe_0",
            "ocr_RoundHalfProbe_0",
            "ocr_HalfStoreProbe_0",
            "ocr_HalfStoreProbe_32",
        ]
    );
    for (id, precision) in [
        (KernelId::CounterProbe, 0),
        (KernelId::RoundHalfProbe, 0),
        (KernelId::HalfStoreProbe, 0),
        (KernelId::HalfStoreProbe, 32),
    ] {
        compile(&context, KernelKey { id, precision }).unwrap();
    }
    assert_eq!(owner.archive_hits.load(Ordering::Relaxed), 0);
    assert_eq!(owner.archive_misses.load(Ordering::Relaxed), 4);
}

#[test]
#[ignore = "requires physical Apple9 GPU; bundled pipeline archive coverage"]
fn gpu_metal_native_archive_hit() {
    let context = GpuContext::new().unwrap();
    assert!(
        context
            .backend
            .owner
            .raw
            .supportsFamily(MTLGPUFamily::Apple9)
    );
    for id in [KernelId::General, KernelId::Conv16] {
        compile(&context, KernelKey::new(id)).unwrap();
    }
    assert_eq!(
        context.backend.owner.archive_hits.load(Ordering::Relaxed),
        2
    );
    assert_eq!(
        context.backend.owner.archive_misses.load(Ordering::Relaxed),
        0
    );
}

#[test]
#[ignore = "requires physical Apple Silicon; native access validation and immutable uploads"]
fn gpu_metal_prepared_ownership_and_failed_submit_preserve_uploads() {
    let context = GpuContext::new().unwrap();
    let foreign = GpuContext::new().unwrap();
    let pipeline = compile(&context, KernelKey::new(KernelId::CounterProbe)).unwrap();
    let value = context
        .create_buffer_init(
            "initial counter",
            &7u32.to_ne_bytes(),
            BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        )
        .unwrap();
    let params = context
        .create_buffer_init("params", &[0; 1024], BufferUsage::UNIFORM)
        .unwrap();
    let other = foreign
        .create_buffer("foreign", 4, BufferUsage::STORAGE)
        .unwrap();
    let short = context
        .create_buffer("short params", 16, BufferUsage::UNIFORM)
        .unwrap();
    let prepare = |buffer, params, access: &[bool]| {
        NativeDispatch::prepare(
            pipeline.clone(),
            vec![buffer, params],
            vec![0, 0],
            [1; 3],
            access,
        )
    };
    assert_eq!(
        prepare(other.clone(), params.clone(), &[false])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        prepare(value.clone(), short, &[false]).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        prepare(value.clone(), params.clone(), &[true])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    let native = prepare(value.clone(), params, &[false]).unwrap();
    let error = submit(
        &context,
        vec![(
            0,
            Operation::Copy {
                source: value.clone(),
                output: other,
                source_offset: 0,
                output_offset: 0,
                bytes: 4,
            },
        )],
        false,
        false,
    )
    .err()
    .unwrap();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    let mut pending = submit(
        &context,
        vec![(
            0,
            Operation::Compute {
                native,
                profile_index: 0,
            },
        )],
        false,
        false,
    )
    .unwrap();
    pending.wait().unwrap();
    let (bytes, _) = Batch::new(context.clone())
        .unwrap()
        .read(&value, 4, false)
        .unwrap();
    assert_eq!(u32::from_ne_bytes(bytes.try_into().unwrap()), 8);
    context.check_alive().unwrap();
}

#[test]
#[ignore = "requires physical Apple Silicon; parameter offsets above 64 KiB and ordered copies"]
fn gpu_metal_parameter_arena_copy_order_and_counter_attribution() {
    let context = GpuContext::new().unwrap();
    let pipeline = compile(&context, KernelKey::new(KernelId::RoundHalfProbe)).unwrap();
    let input = context
        .create_buffer_init("input", bytemuck::bytes_of(&5f32), BufferUsage::STORAGE)
        .unwrap();
    let output = context
        .create_buffer("output", 8, BufferUsage::STORAGE | BufferUsage::COPY_SRC)
        .unwrap();
    let second = 65536usize;
    let mut packet = vec![0; second + 1024];
    packet[..4].copy_from_slice(&1u32.to_ne_bytes());
    packet[second..second + 4].copy_from_slice(&1u32.to_ne_bytes());
    packet[second + 4..second + 8].copy_from_slice(&1u32.to_ne_bytes());
    let params = context
        .create_buffer_init("arena", &packet, BufferUsage::UNIFORM)
        .unwrap();
    let readback = context
        .create_buffer("readback", 8, BufferUsage::COPY_DST | BufferUsage::MAP_READ)
        .unwrap();
    let mut operations = Vec::new();
    for (i, offset) in [0, second].into_iter().enumerate() {
        let native = NativeDispatch::prepare(
            pipeline.clone(),
            vec![input.clone(), output.clone(), params.clone()],
            vec![0, 0, offset as u64],
            [1; 3],
            &[true, false],
        )
        .unwrap();
        operations.push((
            i * 2,
            Operation::Compute {
                native,
                profile_index: i,
            },
        ));
        operations.push((
            i * 2 + 1,
            Operation::Copy {
                source: output.clone(),
                output: readback.clone(),
                source_offset: (i * 4) as u64,
                output_offset: (i * 4) as u64,
                bytes: 4,
            },
        ));
    }
    let mut pending = submit(&context, operations, true, false).unwrap();
    pending.wait().unwrap();
    readback
        .buffer
        .read(8, |data| {
            assert_eq!(bytemuck::cast_slice::<u8, f32>(data), [5., 5.]);
            Ok(())
        })
        .unwrap();
    let timestamps = pending.timestamps().unwrap();
    assert_eq!(timestamps.len(), 2);
    assert!(timestamps.iter().all(|(_, ns)| *ns > 0));
}

#[test]
#[ignore = "requires physical Apple Silicon; more than one timestamp sample buffer"]
fn gpu_metal_counter_buffers_preserve_attribution_across_limit() {
    let context = GpuContext::new().unwrap();
    let pipeline = compile(&context, KernelKey::new(KernelId::CounterProbe)).unwrap();
    let value = context
        .create_buffer_init(
            "counter",
            &0u32.to_ne_bytes(),
            BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        )
        .unwrap();
    let params = context
        .create_buffer_init("params", &[0; 1024], BufferUsage::UNIFORM)
        .unwrap();
    let mut operations = Vec::new();
    for index in 0..2049 {
        let native = NativeDispatch::prepare(
            pipeline.clone(),
            vec![value.clone(), params.clone()],
            vec![0, 0],
            [1; 3],
            &[false],
        )
        .unwrap();
        operations.push((
            index,
            Operation::Compute {
                native,
                profile_index: index * 3 + 7,
            },
        ));
    }
    let mut pending = submit(&context, operations, true, false).unwrap();
    pending.wait().unwrap();
    let timestamps = pending.timestamps().unwrap();
    assert_eq!(timestamps.len(), 2049);
    for (index, &(attributed, ns)) in timestamps.iter().enumerate() {
        assert_eq!(attributed, index * 3 + 7);
        assert!(ns > 0);
    }
    let (bytes, _) = Batch::new(context.clone())
        .unwrap()
        .read(&value, 4, false)
        .unwrap();
    assert_eq!(u32::from_ne_bytes(bytes.try_into().unwrap()), 2049);
}
