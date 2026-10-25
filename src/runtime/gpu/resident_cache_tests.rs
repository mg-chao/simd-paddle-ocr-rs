use super::*;
use resident::Batch;

#[test]
fn resident_cached_memory_checks_host_packets_and_preserves_peak() {
    let memory = GpuMemoryStatistics {
        device_activation_bytes: 2048,
        parameter_bytes: 1024,
        host_parameter_bytes: 512,
        diagnostic_bytes: 128,
        transition_peak_bytes: 8192,
        ..Default::default()
    };
    let total = memory.total_bytes();
    assert!(resident_cached_memory(memory, 1024, total + 511).is_none());
    let reused = resident_cached_memory(memory, 1024, total + 512).unwrap();
    assert_eq!(reused.total_bytes(), total + 512);
    assert_eq!(reused.host_parameter_bytes, 1024);
    assert_eq!(reused.transition_peak_bytes, 8192);
    assert_eq!(
        reused.device_activation_bytes,
        memory.device_activation_bytes
    );
    assert_eq!(reused.parameter_bytes, memory.parameter_bytes);
    assert_eq!(reused.diagnostic_bytes, memory.diagnostic_bytes);
    assert!(resident_cached_memory(memory, usize::MAX, usize::MAX).is_none());
    let grown = resident_cached_memory(
        GpuMemoryStatistics {
            transition_peak_bytes: 0,
            ..memory
        },
        1024,
        total + 512,
    )
    .unwrap();
    assert_eq!(grown.transition_peak_bytes, grown.total_bytes());
}

// A symbolic two-dimensional input lets one compiled graph exercise real
// activation growth, cached views and quota compaction without model files.
fn dynamic_relu() -> Arc<Model> {
    fn varint(mut value: u64) -> Vec<u8> {
        let mut result = Vec::new();
        loop {
            let byte = (value & 127) as u8;
            value >>= 7;
            result.push(byte | if value == 0 { 0 } else { 128 });
            if value == 0 {
                return result;
            }
        }
    }
    fn number(field: u64, value: u64) -> Vec<u8> {
        [varint(field << 3), varint(value)].concat()
    }
    fn bytes(field: u64, value: &[u8]) -> Vec<u8> {
        [
            varint(field << 3 | 2),
            varint(value.len() as u64),
            value.to_vec(),
        ]
        .concat()
    }
    let info = |name: &str| {
        let dimensions = [bytes(1, &[]), bytes(1, &[])].concat();
        [
            bytes(1, name.as_bytes()),
            bytes(
                2,
                &bytes(1, &[number(1, 1), bytes(2, &dimensions)].concat()),
            ),
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
        Model::load_bytes(&[number(1, 8), bytes(8, &number(2, 13)), bytes(7, &graph)].concat())
            .unwrap(),
    )
}

fn allocation_ids(session: &GpuSession) -> Vec<usize> {
    session
        .buffers
        .iter()
        .chain(std::iter::once(&session.parameters))
        .map(|buffer| Arc::as_ptr(&buffer.reservation) as usize)
        .collect()
}

#[test]
#[ignore = "requires physical Metal; cached resident packets, quota rollback and two pending shapes"]
fn gpu_hardware_resident_cached_bindings_reuse_and_submission_order() {
    {
        let quota = 128 * 1024;
        let context = GpuContext::new_with_options(GpuContextOptions {
            max_device_bytes: Some(quota),
            ..Default::default()
        })
        .unwrap();
        let compiled =
            Arc::new(GpuCompiledModel::compile(dynamic_relu(), context.clone()).unwrap());
        let mut session =
            GpuSession::new_resident(compiled.clone(), &[1, 64], ExecutionLimits::default(), None)
                .unwrap();
        let minimum = session.activation_bytes();
        session.reshape(&[1, 96]).unwrap();
        let wide_packets = session.parameter_data.clone();
        // Returning to the first width after growth must rebuild its stale
        // binding generation before it can become a reusable cached view.
        session.reshape(&[1, 64]).unwrap();
        let wide_dispatches = session
            .cache
            .iter()
            .find(|entry| entry.shape.dimensions() == [1, 96])
            .unwrap()
            .bindings
            .as_ref()
            .unwrap()
            .1
            .as_ptr();
        let narrow_packets = session.parameter_data.clone();
        let narrow_dispatches = session.dispatches.as_ptr();
        let ids = allocation_ids(&session);
        let used = context.memory_statistics().used_bytes;
        assert!(session.cached_parameter_bytes() > 0);
        assert!(session.retained_planning_bytes() >= session.cached_parameter_bytes());

        // Exhaust the real device quota. A cached view requires no additional
        // GPU allocation; attempted activation growth must fail transactionally.
        let filler = context
            .create_buffer(
                "cached-view quota witness",
                (quota - used) as u64,
                wgpu::BufferUsages::COPY_DST,
            )
            .unwrap();
        session.reshape(&[1, 96]).unwrap();
        assert!(Arc::ptr_eq(&wide_packets, &session.parameter_data));
        assert_eq!(wide_dispatches, session.dispatches.as_ptr());
        session.reshape(&[1, 64]).unwrap();
        assert!(Arc::ptr_eq(&narrow_packets, &session.parameter_data));
        assert_eq!(narrow_dispatches, session.dispatches.as_ptr());
        assert_eq!(allocation_ids(&session), ids);
        assert_eq!(context.memory_statistics().used_bytes, quota);
        let active_memory = session.memory;
        assert_eq!(
            session.reshape(&[1, 8192]).unwrap_err().kind(),
            ErrorKind::ResourceLimit
        );
        assert_eq!(session.input_shape().dimensions(), &[1, 64]);
        assert_eq!(session.memory, active_memory);
        assert!(Arc::ptr_eq(&narrow_packets, &session.parameter_data));
        assert_eq!(context.memory_statistics().used_bytes, quota);
        drop(filler);

        // A missing proof falls back rather than mutating the active view.
        let index = session
            .cache
            .iter()
            .position(|entry| entry.shape.dimensions() == [1, 96])
            .unwrap();
        let packet = session.cache[index].parameter_data.take();
        assert!(!session.restore_resident_cached(index));
        assert_eq!(session.input_shape().dimensions(), &[1, 64]);
        session.cache[index].parameter_data = packet;

        let gate = context.acquire_gate().unwrap();
        let mut pending = Vec::new();
        for (width, positive) in [(64, 0.25), (96, 0.75)] {
            session.reshape(&[1, width]).unwrap();
            let data: Vec<f32> = (0..width)
                .map(|i| if i % 3 == 0 { -0.5 } else { positive })
                .collect();
            context
                .upload_resident(
                    &session.resident_input().buffer,
                    bytemuck::cast_slice(&data),
                )
                .unwrap();
            let mut batch = Batch::new(context.clone()).unwrap();
            let output = batch.graph(&mut session, "cached-shape").unwrap();
            pending.push((
                data.into_iter().map(|v| v.max(0.0)).collect::<Vec<_>>(),
                batch
                    .submit(&output.buffer, width * 4, false, &gate)
                    .unwrap(),
            ));
        }
        for (expected, submitted) in pending {
            let (bytes, statistics) = submitted.finish().unwrap();
            let actual: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|value| f32::from_le_bytes(*value))
                .collect();
            assert_eq!(actual, expected);
            assert_eq!(statistics.native_graph_dispatches, 1);
        }
        drop(gate);

        // Reuse cannot relax the existing mutable-uniform guard within one
        // unsubmitted batch, even though both shape views are cached.
        session.reshape(&[1, 64]).unwrap();
        let mut batch = Batch::new(context.clone()).unwrap();
        batch.graph(&mut session, "first-shape").unwrap();
        session.reshape(&[1, 96]).unwrap();
        assert_eq!(
            batch
                .graph(&mut session, "second-shape")
                .err()
                .unwrap()
                .kind(),
            ErrorKind::InvalidInput
        );
        drop(batch);

        // Tightening the activation budget takes the original rebuilding path
        // and clears every binding that refers to the old allocation generation.
        session
            .reshape_with_limits(
                &[1, 64],
                ExecutionLimits {
                    max_activation_bytes: minimum,
                },
            )
            .unwrap();
        assert_eq!(session.activation_bytes(), minimum);
        assert!(session.cache.iter().all(|entry| entry.bindings.is_none()));
        let compact_packets = session.parameter_data.clone();
        let compact_ids = allocation_ids(&session);
        assert!(
            session
                .reshape_with_limits(
                    &[1, 64],
                    ExecutionLimits {
                        max_activation_bytes: minimum - 1,
                    },
                )
                .is_err()
        );
        assert_eq!(session.execution_limits().max_activation_bytes, minimum);
        assert_eq!(session.activation_bytes(), minimum);
        assert!(Arc::ptr_eq(&compact_packets, &session.parameter_data));
        assert_eq!(allocation_ids(&session), compact_ids);
        session
            .reshape_with_limits(&[1, 64], ExecutionLimits::default())
            .unwrap();
        for width in 1..=10 {
            session.reshape(&[1, width]).unwrap();
        }
        assert_eq!(session.cache.len(), PLAN_CACHE_CAPACITY);
        assert!(
            session.cached_parameter_bytes() <= PLAN_CACHE_CAPACITY * session.parameters.bytes()
        );

        let mut host = GpuSession::new(compiled, &[1, 64], ExecutionLimits::default()).unwrap();
        host.reshape(&[1, 96]).unwrap();
        host.reshape(&[1, 64]).unwrap();
        assert!(
            host.cache
                .iter()
                .all(|entry| entry.parameter_data.is_none())
        );
        assert_eq!(host.cached_parameter_bytes(), 0);
    }
}
