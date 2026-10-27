use super::*;
use crate::runtime::gpu::resident::Batch;
use crate::runtime::gpu::resident::schedule::Operation;

#[test]
#[ignore = "requires a physical Vulkan GPU; three cooperative-matrix tail shapes"]
fn vulkan_cooperative_projection_matches_half_operand_oracle() {
    let context = GpuContext::new().unwrap();
    if !context.backend.owner.cooperative_matrix {
        assert_eq!(
            compile_resident(
                &context,
                KernelKey::new(KernelId::ProjectionMatrixF16),
                &[true, true, true, false]
            )
            .unwrap_err()
            .kind(),
            ErrorKind::Unsupported
        );
        eprintln!("Cooperative F16/F32 shape unavailable; portable projection remains selected.");
        return;
    }
    for (rows, inner, classes) in [(3usize, 17usize, 65usize), (17, 31, 129), (35, 64, 257)] {
        let a: Vec<_> = (0..rows * inner)
            .map(|i| half::f16::from_f32(((i * 7 % 29) as f32 - 14.) / 32.))
            .collect();
        let b: Vec<_> = (0..inner * classes)
            .map(|i| half::f16::from_f32(((i * 11 % 37) as f32 - 18.) / 64.))
            .collect();
        let bias: Vec<_> = (0..classes).map(|c| (c % 7) as f32 / 16.).collect();
        let upload_half = |values: &[half::f16]| {
            context
                .create_buffer_init(
                    "half matrix",
                    &values
                        .iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect::<Vec<_>>(),
                    BufferUsage::STORAGE,
                )
                .unwrap()
        };
        let input = upload_half(&a);
        let weights = upload_half(&b);
        let bias_buffer = context
            .create_buffer_init("bias", bytemuck::cast_slice(&bias), BufferUsage::STORAGE)
            .unwrap();
        let tiles = classes.div_ceil(64);
        let mut batch = Batch::new(context.clone()).unwrap();
        let partial = batch
            .storage("cooperative partials", rows * tiles * 16)
            .unwrap();
        batch
            .dispatch(
                "projection_cooperative_test",
                KernelKey::new(KernelId::ProjectionMatrixF16),
                &[
                    (&input, true),
                    (&weights, true),
                    (&bias_buffer, true),
                    (&partial, false),
                ],
                &[rows as u32, inner as u32, classes as u32, tiles as u32, 1],
                [tiles as u32, rows.div_ceil(32) as u32, 1],
            )
            .unwrap();
        let (bytes, _) = batch.read(&partial, rows * tiles * 16, false).unwrap();
        let result: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| u32::from_le_bytes(*v))
            .collect();
        for row in 0..rows {
            for tile in 0..tiles {
                let scores: Vec<_> = (tile * 64..((tile + 1) * 64).min(classes))
                    .map(|c| {
                        let mut sum = 0f32;
                        for k in 0..inner {
                            sum += a[row * inner + k].to_f32() * b[k * classes + c].to_f32();
                        }
                        sum + bias[c]
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(-f32::MAX, f32::max);
                let winner = scores.iter().position(|&v| v == maximum).unwrap() + tile * 64;
                let sum: f32 = scores.iter().map(|v| (v - maximum).exp()).sum();
                let actual = &result[(row * tiles + tile) * 4..][..4];
                assert_eq!(
                    actual[0], winner as u32,
                    "rows={rows}, row={row}, tile={tile}"
                );
                assert!((f32::from_bits(actual[1]) - maximum).abs() < 1e-5);
                assert!((f32::from_bits(actual[2]) - sum).abs() < 1e-4 * sum.max(1.));
                assert_eq!(actual[3], 0);
            }
        }
    }
}

#[test]
#[ignore = "requires a physical Vulkan GPU; zero initialization, alias rejection, and staging retirement"]
fn vulkan_initialization_aliases_and_abandoned_uploads() {
    let context = GpuContext::new().unwrap();
    let buffer = context
        .create_buffer_init("discarded upload", &[9; 128], BufferUsage::STORAGE)
        .unwrap();
    assert_eq!(context.memory_statistics().used_bytes, 256);
    drop(buffer);
    assert_eq!(context.memory_statistics().used_bytes, 0);
    let mut batch = Batch::new(context.clone()).unwrap();
    let zero = batch.storage("zero initialized scratch", 260).unwrap();
    let (bytes, _) = batch.read(&zero, 260, false).unwrap();
    assert!(bytes.iter().all(|&b| b == 0));
    let before = context.memory_statistics().used_bytes;
    let operations = vec![(
        0,
        Operation::Copy {
            source: zero.clone(),
            output: zero.clone(),
            source_offset: 0,
            output_offset: 4,
            bytes: 8,
        },
    )];
    assert!(submit(&context, operations, false, false).is_err());
    context.check_alive().unwrap();
    assert_eq!(context.memory_statistics().used_bytes, before);
    let mut batch = Batch::new(context.clone()).unwrap();
    batch
        .dispatch(
            "counter",
            KernelKey::new(KernelId::CounterProbe),
            &[(&zero, false)],
            &[],
            [1; 3],
        )
        .unwrap();
    let gate = context.acquire_gate().unwrap();
    let pending = batch.submit(&zero, 4, false, &gate).unwrap();
    drop(pending);
    drop(gate);
    let (bytes, _) = Batch::new(context.clone())
        .unwrap()
        .read(&zero, 4, false)
        .unwrap();
    assert_eq!(u32::from_le_bytes(bytes.try_into().unwrap()), 1);
}

#[test]
#[ignore = "requires a physical Vulkan GPU"]
fn vulkan_parameter_arena_and_transfer_order() {
    let context = GpuContext::new().unwrap();
    let pipeline = compile(&context, KernelKey::new(KernelId::BindingProbe)).unwrap();
    let input = context
        .create_buffer_init("input", bytemuck::bytes_of(&5f32), BufferUsage::STORAGE)
        .unwrap();
    let output = context
        .create_buffer("output", 4, BufferUsage::STORAGE | BufferUsage::COPY_SRC)
        .unwrap();
    let alignment = context.limits.min_uniform_buffer_offset_alignment as usize;
    let second = (64 * 1024usize).div_ceil(alignment) * alignment;
    let mut packet = vec![0; second + 512];
    packet[..4].copy_from_slice(&7u32.to_le_bytes());
    packet[second..second + 4].copy_from_slice(&11u32.to_le_bytes());
    let params = context
        .create_buffer_init("parameter arena", &packet, BufferUsage::UNIFORM)
        .unwrap();
    let readback = context
        .create_buffer("readback", 8, BufferUsage::COPY_DST | BufferUsage::MAP_READ)
        .unwrap();
    let mut operations = Vec::new();
    for (index, offset) in [0, second].into_iter().enumerate() {
        let native = NativeDispatch::prepare(
            pipeline.clone(),
            vec![input.clone(), output.clone(), params.clone()],
            vec![0, 0, offset as u64],
            [1; 3],
            &[true, false],
        )
        .unwrap();
        operations.push((
            2 * index,
            Operation::Compute {
                native,
                profile_index: index,
            },
        ));
        operations.push((
            2 * index + 1,
            Operation::Copy {
                source: output.clone(),
                output: readback.clone(),
                source_offset: 0,
                output_offset: (index * 4) as u64,
                bytes: 4,
            },
        ));
    }
    let mut pending = submit(&context, operations, true, false).unwrap();
    pending.wait().unwrap();
    readback
        .buffer
        .read(8, |bytes| {
            assert_eq!(bytemuck::cast_slice::<u8, f32>(bytes), [12., 16.]);
            Ok(())
        })
        .unwrap();
    let timestamps = pending.timestamps().unwrap();
    assert_eq!(timestamps.len(), 2);
    assert!(timestamps.iter().all(|(_, ns)| *ns > 0));
    context.check_alive().unwrap();
}

#[test]
#[ignore = "requires a physical Vulkan GPU"]
fn vulkan_prepared_bindings_validate_ownership_and_shader_access() {
    let context = GpuContext::new().unwrap();
    let pipeline = compile(&context, KernelKey::new(KernelId::BindingProbe)).unwrap();
    let input = context
        .create_buffer("input", 4, BufferUsage::STORAGE)
        .unwrap();
    let output = context
        .create_buffer("output", 4, BufferUsage::STORAGE)
        .unwrap();
    let params = context
        .create_buffer("params", 512, BufferUsage::UNIFORM)
        .unwrap();
    let prepare = |input: GpuBuffer, params: GpuBuffer, access: &[bool]| {
        NativeDispatch::prepare(
            pipeline.clone(),
            vec![input, output.clone(), params],
            vec![0; 3],
            [1; 3],
            access,
        )
    };
    assert_eq!(
        prepare(input.clone(), params.clone(), &[false, false])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    let short = context
        .create_buffer("short", 16, BufferUsage::UNIFORM)
        .unwrap();
    assert_eq!(
        prepare(input.clone(), short, &[true, false])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    let foreign = GpuContext::new().unwrap();
    let other = foreign
        .create_buffer("foreign", 4, BufferUsage::STORAGE)
        .unwrap();
    assert_eq!(
        prepare(other, params.clone(), &[true, false])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    let before = context.memory_statistics().used_bytes;
    let dispatch = prepare(input.clone(), params.clone(), &[true, false]).unwrap();
    drop(input);
    drop(params);
    assert_eq!(context.memory_statistics().used_bytes, before);
    drop(dispatch);
    assert_eq!(context.memory_statistics().used_bytes, before - 516);
    context.check_alive().unwrap();
}

#[test]
#[ignore = "requires a physical Vulkan GPU; deterministic unsignalled timeline and device-loss propagation"]
fn vulkan_timeout_retains_resources_and_device_loss_is_terminal() {
    let context = GpuContext::new().unwrap();
    let source = context
        .create_buffer_init("retained source", &[3; 256], BufferUsage::STORAGE)
        .unwrap();
    let readback = context
        .create_buffer(
            "retained readback",
            256,
            BufferUsage::MAP_READ | BufferUsage::COPY_DST,
        )
        .unwrap();
    let operations = vec![(
        0,
        Operation::Copy {
            source: source.clone(),
            output: readback.clone(),
            source_offset: 0,
            output_offset: 0,
            bytes: 256,
        },
    )];
    let mut pending = submit(&context, operations, false, false).unwrap();
    // Wait on a value beyond the submitted work, guaranteeing a timeout without
    // a slow shader or deliberately hanging the physical device.
    pending.wait_for_unsignalled_test().unwrap_err();
    drop(source);
    drop(readback);
    assert!(context.memory_statistics().used_bytes >= 512);
    assert!(context.acquire_gate().is_err());
    assert!(pending.wait().is_err());
    drop(pending); // drains the real queue before retiring its owners
    assert_eq!(context.memory_statistics().used_bytes, 0);

    let context = GpuContext::new().unwrap();
    let error = context
        .backend
        .owner
        .result::<()>(
            Err(vk::Result::ERROR_DEVICE_LOST),
            "injected native device loss",
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::DeviceLost);
    assert_eq!(
        context.acquire_gate().err().unwrap().kind(),
        ErrorKind::DeviceLost
    );
    assert!(
        context
            .create_buffer("after device loss", 4, BufferUsage::STORAGE)
            .is_err()
    );
}

#[test]
#[ignore = "physical GPU performance experiment: one warmup and three measured dispatches per variant"]
fn profile_projection_variants() {
    let context = GpuContext::new().unwrap();
    let (rows, inner, classes) = (40usize, 120usize, 6906usize);
    let features: Vec<f32> = (0..rows * inner)
        .map(|i| ((i * 7 % 29) as f32 - 14.) / 32.)
        .collect();
    let weights: Vec<f32> = (0..inner * classes)
        .map(|i| ((i * 11 % 37) as f32 - 18.) / 64.)
        .collect();
    let bias: Vec<f32> = (0..classes).map(|i| (i % 7) as f32 / 16.).collect();
    let upload = |name, values: &[f32]| {
        context
            .create_buffer_init(name, bytemuck::cast_slice(values), BufferUsage::STORAGE)
            .unwrap()
    };
    let a = upload("F32 features", &features);
    let b = upload("F32 weights", &weights);
    let bias = upload("F32 bias", &bias);
    let upload_half = |name, values: &[f32]| {
        context
            .create_buffer_init(
                name,
                &values
                    .iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
                    .collect::<Vec<_>>(),
                BufferUsage::STORAGE,
            )
            .unwrap()
    };
    let ah = upload_half("F16 features", &features);
    let bh = upload_half("F16 weights", &weights);
    let tiles = classes.div_ceil(64);
    let mut words = [0u32; 256];
    words[..5].copy_from_slice(&[rows as u32, inner as u32, classes as u32, tiles as u32, 1]);
    let params = context
        .create_buffer_init(
            "projection packet",
            bytemuck::cast_slice(&words),
            BufferUsage::UNIFORM,
        )
        .unwrap();
    let bytes = (rows * tiles * 16) as u64;
    let output = context
        .create_buffer("projection partials", bytes, BufferUsage::STORAGE)
        .unwrap();
    let readback = context
        .create_buffer(
            "projection readback",
            bytes,
            BufferUsage::MAP_READ | BufferUsage::COPY_DST,
        )
        .unwrap();
    let mut reference: Option<Vec<u32>> = None;
    for id in [
        KernelId::Heads4,
        KernelId::Heads4Subgroup,
        KernelId::ProjectionMatrixF16,
    ] {
        if (id == KernelId::Heads4Subgroup && !context.backend.owner.subgroup_reductions)
            || (id == KernelId::ProjectionMatrixF16 && !context.backend.owner.cooperative_matrix)
        {
            continue;
        }
        let half = id == KernelId::ProjectionMatrixF16;
        let pipeline = compile(&context, KernelKey::new(id)).unwrap();
        let native = NativeDispatch::prepare(
            pipeline,
            vec![
                if half { ah.clone() } else { a.clone() },
                if half { bh.clone() } else { b.clone() },
                bias.clone(),
                output.clone(),
                params.clone(),
            ],
            vec![0; 5],
            [
                tiles as u32,
                rows.div_ceil(if half { 32 } else { 4 }) as u32,
                1,
            ],
            &[true, true, true, false],
        )
        .unwrap();
        let mut times = Vec::new();
        for run in 0..4 {
            let operations = vec![
                (
                    0,
                    Operation::Compute {
                        native: native.clone(),
                        profile_index: 0,
                    },
                ),
                (
                    1,
                    Operation::Copy {
                        source: output.clone(),
                        output: readback.clone(),
                        source_offset: 0,
                        output_offset: 0,
                        bytes,
                    },
                ),
            ];
            let mut submission = submit(&context, operations, true, false).unwrap();
            submission.wait().unwrap();
            context.check_alive().unwrap();
            let actual = readback
                .buffer
                .read(bytes as usize, |data| {
                    Ok(data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|v| u32::from_le_bytes(*v))
                        .collect::<Vec<_>>())
                })
                .unwrap();
            if let Some(expected) = &reference {
                for (a, b) in actual
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(expected.as_chunks::<4>().0)
                {
                    assert_eq!(a[0], b[0], "{id:?} winner");
                    assert_eq!(a[3], b[3], "{id:?} invalid");
                    assert!(
                        (f32::from_bits(a[1]) - f32::from_bits(b[1])).abs() < 1e-5,
                        "{id:?} maximum"
                    );
                    assert!(
                        (f32::from_bits(a[2]) - f32::from_bits(b[2])).abs()
                            < 1e-4 * f32::from_bits(b[2]).max(1.),
                        "{id:?} probability"
                    );
                }
            } else {
                reference = Some(actual);
            }
            if run > 0 {
                times.push(submission.timestamps().unwrap()[0].1);
            }
        }
        eprintln!(
            "projection {id:?}: device_ns={times:?}; shape=[{rows},{inner},{classes}]; warmups=1; measurements=3"
        );
    }
}
