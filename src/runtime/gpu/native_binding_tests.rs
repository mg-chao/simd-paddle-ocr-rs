//! Physical Metal checks for immutable prepared graph bindings and ownership.
use super::*;
use crate::runtime::CompileOptions;
use resident::Batch;

fn dynamic_relu() -> Arc<Model> {
    fn number(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 127) as u8;
            value >>= 7;
            out.push(byte | if value == 0 { 0 } else { 128 });
            if value == 0 {
                return out;
            }
        }
    }
    fn integer(field: u64, value: u64) -> Vec<u8> {
        [number(field << 3), number(value)].concat()
    }
    fn bytes(field: u64, value: &[u8]) -> Vec<u8> {
        [
            number(field << 3 | 2),
            number(value.len() as u64),
            value.to_vec(),
        ]
        .concat()
    }
    let info = |name: &str| {
        let dims = [bytes(1, &[]), bytes(1, &[])].concat();
        [
            bytes(1, name.as_bytes()),
            bytes(2, &bytes(1, &[integer(1, 1), bytes(2, &dims)].concat())),
        ]
        .concat()
    };
    let graph = [
        tests::node("Relu", &["x"], "y", &[]),
        bytes(11, &info("x")),
        bytes(12, &info("y")),
    ]
    .concat();
    Arc::new(
        Model::load_bytes(&[integer(1, 8), bytes(8, &integer(2, 13)), bytes(7, &graph)].concat())
            .unwrap(),
    )
}

fn metal_context(quota: Option<usize>) -> Arc<GpuContext> {
    GpuContext::new_with_options(GpuContextOptions {
        max_device_bytes: quota,
        ..Default::default()
    })
    .unwrap()
}

fn prepared(session: &GpuSession) -> Arc<executor::NativeDispatch> {
    match &session.dispatches[0].command {
        Command::Compute { native, .. } => native.clone(),
        _ => panic!("expected actual prepared native graph dispatch"),
    }
}

#[test]
#[ignore = "requires physical Metal; prepared Arc reuse, generation invalidation, quota rollback and two pending waves"]
fn gpu_hardware_prepared_native_bindings_reuse_and_generation() {
    let context = metal_context(Some(128 * 1024));
    let compiled = Arc::new(GpuCompiledModel::compile(dynamic_relu(), context.clone()).unwrap());
    let mut session =
        GpuSession::new_resident(compiled, &[1, 64], ExecutionLimits::default(), None).unwrap();
    let initial = prepared(&session);
    let initial_generation = session.generation;
    assert_eq!(initial.hazards().len(), 6);
    for (i, &(id, read)) in initial.hazards().iter().enumerate() {
        assert_eq!(id, Arc::as_ptr(&initial.owners()[i].reservation) as usize);
        assert_eq!(read, i != 5);
    }
    assert_eq!(initial.offsets().len(), 7);
    assert_eq!(initial.offsets()[6], 0);
    assert_eq!(
        initial.runtime_sizes(),
        initial.owners()[..6]
            .iter()
            .map(|owner| owner.size() as u32)
            .collect::<Vec<_>>(),
        "WGSL runtime-array byte sizes are cached in declaration order",
    );
    assert!(session.prepared_native_bytes() >= initial.retained_bytes());
    session.reshape(&[1, 32]).unwrap();
    assert_eq!(session.generation, initial_generation);
    assert!(!Arc::ptr_eq(&initial, &prepared(&session)));
    session.reshape(&[1, 64]).unwrap();
    assert!(
        Arc::ptr_eq(&initial, &prepared(&session)),
        "LRU restoration must reuse the complete admitted binding snapshot"
    );
    let memory = session.memory;
    let used = context.memory_statistics().used_bytes;
    assert_eq!(
        session.reshape(&[1, 65536]).unwrap_err().kind(),
        ErrorKind::ResourceLimit
    );
    assert_eq!(session.memory, memory);
    assert_eq!(session.generation, initial_generation);
    assert_eq!(context.memory_statistics().used_bytes, used);
    assert!(Arc::ptr_eq(&initial, &prepared(&session)));

    // Exactly one Arc is retained for every recording, with no device allocation.
    let strong = Arc::strong_count(&initial);
    let mut first = Batch::new(context.clone()).unwrap();
    let first_output = first.graph(&mut session, "prepared first").unwrap();
    assert_eq!(Arc::strong_count(&initial), strong + 1);
    let mut second = Batch::new(context.clone()).unwrap();
    second.graph(&mut session, "prepared second").unwrap();
    assert_eq!(Arc::strong_count(&initial), strong + 2);
    assert_eq!(context.memory_statistics().used_bytes, used);
    drop(second);
    assert_eq!(Arc::strong_count(&initial), strong + 1);
    drop(first_output);
    drop(first);

    let gate = context.acquire_gate().unwrap();
    let mut pending = Vec::new();
    for (width, positive) in [(64, 0.25f32), (32, 0.75)] {
        session.reshape(&[1, width]).unwrap();
        let values: Vec<_> = (0..width)
            .map(|i| if i % 3 == 0 { -0.5 } else { positive })
            .collect();
        context
            .upload_resident(
                &session.resident_input().buffer,
                bytemuck::cast_slice(&values),
            )
            .unwrap();
        let mut batch = Batch::new(context.clone()).unwrap();
        let output = batch.graph(&mut session, "prepared pending").unwrap();
        pending.push((
            values.into_iter().map(|v| v.max(0.0)).collect::<Vec<_>>(),
            batch
                .submit(&output.buffer, width * 4, false, &gate)
                .unwrap(),
        ));
    }
    for (expected, pending) in pending {
        let (actual, stats) = pending.finish().unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, f32>(&actual), expected);
        assert_eq!(stats.native_graph_dispatches, 1);
    }
    drop(gate);
    session.reshape(&[1, 1024]).unwrap();
    assert_ne!(session.generation, initial_generation);
    assert!(!Arc::ptr_eq(&initial, &prepared(&session)));
    assert!(
        session.cache.iter().all(|entry| entry.bindings.is_none()),
        "allocation generation changes must discard every old prepared owner"
    );
    drop(initial);
}

#[test]
#[ignore = "requires physical Metal; reservation ownership before submission/after completion and mandatory native profiling"]
fn gpu_hardware_prepared_native_bindings_lifetime_and_profile() {
    let context = metal_context(None);
    let baseline = context.memory_statistics().used_bytes;
    let compiled = Arc::new(GpuCompiledModel::compile(dynamic_relu(), context.clone()).unwrap());
    let mut session =
        GpuSession::new_resident(compiled.clone(), &[1, 67], ExecutionLimits::default(), None)
            .unwrap();
    let native = prepared(&session);
    let allocation = Arc::downgrade(&session.buffers[0].reservation);
    let weak = Arc::downgrade(&native);
    let values: Vec<f32> = (0..67)
        .map(|i| if i % 2 == 0 { -0.25 } else { 0.5 })
        .collect();
    context
        .upload_resident(
            &session.resident_input().buffer,
            bytemuck::cast_slice(&values),
        )
        .unwrap();
    let mut batch = Batch::new(context.clone()).unwrap();
    let output = batch.graph(&mut session, "prepared lifetime").unwrap();
    drop(native);
    drop(session);
    drop(compiled);
    assert!(
        weak.upgrade().is_some(),
        "recorded Arc must outlive its session"
    );
    assert!(
        allocation.upgrade().is_some(),
        "raw Metal handles alone cannot own quota reservations"
    );
    let gate = context.acquire_gate().unwrap();
    let pending = batch.submit(&output.buffer, 67 * 4, false, &gate).unwrap();
    drop(output);
    let (actual, stats) = pending.finish().unwrap();
    assert_eq!(
        bytemuck::cast_slice::<u8, f32>(&actual),
        values.into_iter().map(|v| v.max(0.0)).collect::<Vec<_>>()
    );
    assert_eq!(stats.native_graph_dispatches, 1);
    context
        .backend
        .owner
        .raw
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
    assert!(
        weak.upgrade().is_none(),
        "completion must release prepared snapshots without an Arc cycle"
    );
    assert!(
        allocation.upgrade().is_none(),
        "completion must release reservations"
    );
    drop(gate);
    assert_eq!(context.memory_statistics().used_bytes, baseline);

    let compiled = Arc::new(GpuCompiledModel::compile(dynamic_relu(), context.clone()).unwrap());
    let mut session =
        GpuSession::new_resident(compiled, &[1, 17], ExecutionLimits::default(), None).unwrap();
    context
        .upload_resident(
            &session.resident_input().buffer,
            bytemuck::cast_slice(&[0.5f32; 17]),
        )
        .unwrap();
    let mut batch = Batch::new(context).unwrap();
    batch.profiling = true;
    let output = batch
        .graph(&mut session, "profiled prepared native")
        .unwrap();
    let (actual, stats) = batch.read(&output.buffer, 17 * 4, false).unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, f32>(&actual), &[0.5; 17]);
    assert_eq!(stats.native_graph_dispatches, 1);
    assert_eq!(stats.profiling_readback_bytes, 0);
    assert!(
        stats
            .profile
            .unwrap()
            .kernels
            .iter()
            .all(|record| record.device_ns.is_none()
                && record.granularity == GpuTimingGranularity::Unavailable)
    );
}

#[test]
#[ignore = "requires physical Metal; prepared packed operand retains its accounted immutable allocation"]
fn gpu_hardware_prepared_native_packed_weight_reservation() {
    let context = GpuContext::new_with_options(GpuContextOptions {
        ..Default::default()
    })
    .unwrap();
    let baseline = context.memory_statistics().used_bytes;
    let weights: Vec<f32> = (0..17 * 16 * 9)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) / 64.0)
        .collect();
    let model = tests::model(
        vec![
            tests::tensor("w", &[17, 16, 3, 3], &weights),
            tests::tensor("bias", &[17], &[0.125; 17]),
            tests::node(
                "Conv",
                &["x", "w", "bias"],
                "y",
                &[
                    tests::integers("kernel_shape", &[3, 3]),
                    tests::integers("pads", &[1, 1, 1, 1]),
                ],
            ),
        ],
        &[1, 16, 7, 9],
        &[1, 17, 7, 9],
    );
    let compiled = Arc::new(
        GpuCompiledModel::compile_with_options(
            model,
            context.clone(),
            CompileOptions {
                gpu: GpuCompileOptions {
                    precision: GpuPrecision::MixedF16,
                    matrix_precision: GpuMatrixPrecision::F16,
                },
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let session = GpuSession::new_resident(
        compiled.clone(),
        &[1, 16, 7, 9],
        ExecutionLimits::default(),
        None,
    )
    .unwrap();
    let (recipe, dispatch) = session
        .recipes
        .iter()
        .zip(&session.dispatches)
        .find(|(r, _)| compiled.packed_operand_precision(r).is_some())
        .expect("actual spatial matrix pack");
    let native = match &dispatch.command {
        Command::Compute { native, .. } => native.clone(),
        _ => panic!("packed convolution must use an actual prepared native dispatch"),
    };
    let packed = compiled.specialized_weight(recipe).unwrap().unwrap();
    assert!(
        native.runtime_sizes().is_empty(),
        "native MSL needs no runtime-array table"
    );
    assert!(Arc::ptr_eq(
        &packed.reservation,
        &native.owners()[1].reservation
    ));
    assert_eq!(
        native.hazards()[1].0,
        Arc::as_ptr(&packed.reservation) as usize
    );
    let allocation = Arc::downgrade(&packed.reservation);
    drop(packed);
    drop(session);
    drop(compiled);
    assert!(
        allocation.upgrade().is_some(),
        "the native snapshot must retain the packed buffer's reservation after the cache drops"
    );
    drop(native);
    assert!(allocation.upgrade().is_none());
    assert_eq!(context.memory_statistics().used_bytes, baseline);
}

#[test]
#[ignore = "requires physical Metal; actual raw ranges/minima are checked at preparation and sorted depths remain dynamic"]
fn gpu_hardware_prepared_native_metadata_validation() {
    let context = metal_context(None);
    let compiled = Arc::new(GpuCompiledModel::compile(dynamic_relu(), context.clone()).unwrap());
    let session =
        GpuSession::new_resident(compiled, &[1, 17], ExecutionLimits::default(), None).unwrap();
    let native = prepared(&session);
    let readonly = [true, true, true, true, true, false];
    let prepare = |owners, offsets| {
        executor::NativeDispatch::prepare(
            native.pipeline_for_test(),
            owners,
            offsets,
            [1, 1, 1],
            &readonly,
        )
    };
    assert_eq!(
        prepare(native.owners().to_vec(), vec![0; 6])
            .err()
            .unwrap()
            .kind(),
        ErrorKind::InvalidInput,
    );
    let mut offsets = native.offsets().to_vec();
    offsets[0] = 1;
    assert_eq!(
        prepare(native.owners().to_vec(), offsets)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::ResourceLimit
    );
    let mut offsets = native.offsets().to_vec();
    offsets[0] = native.owners()[0].size();
    assert_eq!(
        prepare(native.owners().to_vec(), offsets)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::ResourceLimit
    );
    let mut owners = native.owners().to_vec();
    owners[6] = context
        .create_buffer(
            "too short native uniform",
            16,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        )
        .unwrap();
    assert_eq!(
        prepare(owners, vec![0; 7]).err().unwrap().kind(),
        ErrorKind::InvalidInput
    );
    let foreign = metal_context(None);
    let mut mixed = native.owners().to_vec();
    mixed[0] =
        resident::storage(&foreign, "foreign preparation storage", mixed[0].bytes()).unwrap();
    assert_eq!(
        prepare(mixed, native.offsets().to_vec())
            .err()
            .unwrap()
            .kind(),
        ErrorKind::InvalidInput
    );
    let foreign_compiled =
        Arc::new(GpuCompiledModel::compile(dynamic_relu(), foreign.clone()).unwrap());
    let foreign_session =
        GpuSession::new_resident(foreign_compiled, &[1, 17], ExecutionLimits::default(), None)
            .unwrap();
    let foreign_native = prepared(&foreign_session);
    assert_eq!(
        prepare(
            foreign_native.owners().to_vec(),
            foreign_native.offsets().to_vec()
        )
        .err()
        .unwrap()
        .kind(),
        ErrorKind::InvalidInput,
        "pipeline owner device must match even when all buffers share one foreign ledger"
    );

    let mut encoder =
        context
            .backend
            .owner
            .raw
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("native depth validation only"),
            });
    assert_eq!(
        executor::encode(&context, &mut encoder, &[(1, &native), (0, &native)], false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput,
    );
    let mut foreign_encoder =
        foreign
            .backend
            .owner
            .raw
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("foreign native encoder validation only"),
            });
    assert_eq!(
        executor::encode(&foreign, &mut foreign_encoder, &[(0, &native)], false)
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    // No command buffer is submitted in this constructor/metadata check.
}
