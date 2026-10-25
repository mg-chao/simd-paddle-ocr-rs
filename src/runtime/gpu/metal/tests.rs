use super::*;
use resident::schedule::Operation;
use std::time::Duration;

const AFFINE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Parameters { uint count; uint offset; float scale; float bias; };
kernel void affine(device const float* input [[buffer(0)]], device float* output [[buffer(1)]],
    constant Parameters& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i<p.count) output[i+p.offset]=input[i]*p.scale+p.bias;
}
"#;
fn pipeline(context: &GpuContext) -> Arc<execution::NativePipeline> {
    execution::compile(
        context,
        AFFINE,
        "affine",
        [64, 1, 1],
        vec![4, 4, 16],
        vec![true, false],
    )
    .unwrap()
}
fn buffer(context: &GpuContext, values: &[f32]) -> GpuBuffer {
    context
        .create_buffer_init(
            "Metal test values",
            bytemuck::cast_slice(values),
            BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        )
        .unwrap()
}
fn parameters(context: &GpuContext, count: u32, offset: u32, scale: f32, bias: f32) -> GpuBuffer {
    context
        .create_buffer_init(
            "Metal test parameters",
            bytemuck::cast_slice(&[count, offset, scale.to_bits(), bias.to_bits()]),
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        )
        .unwrap()
}
fn dispatch(
    pipeline: Arc<execution::NativePipeline>,
    input: &GpuBuffer,
    output: &GpuBuffer,
    params: &GpuBuffer,
    count: u32,
) -> Arc<execution::NativeDispatch> {
    execution::NativeDispatch::prepare(
        pipeline,
        vec![input.clone(), output.clone(), params.clone()],
        vec![0; 3],
        [count.div_ceil(64), 1, 1],
        &[true, false],
    )
    .unwrap()
}
fn readback(
    context: &GpuContext,
    source: &GpuBuffer,
    operations: Vec<(usize, Operation)>,
    profiling: bool,
) -> (Vec<f32>, Submission) {
    let read = context
        .create_buffer(
            "Metal test readback",
            source.size(),
            BufferUsage::MAP_READ | BufferUsage::COPY_DST,
        )
        .unwrap();
    let mut operations = operations;
    operations.push((
        operations.len() + 1,
        Operation::Copy {
            source: source.clone(),
            output: read.clone(),
            source_offset: 0,
            output_offset: 0,
            bytes: source.size(),
        },
    ));
    let mut submission = submit(context, operations, profiling, false).unwrap();
    submission.wait().unwrap();
    let values = read
        .buffer
        .read(read.bytes(), |bytes| {
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_ne_bytes(*v))
                .collect())
        })
        .unwrap();
    (values, submission)
}

#[test]
#[ignore = "requires physical Apple Silicon Metal 4"]
fn native_upload_clear_compute_copy_and_parameter_snapshots() {
    let context = GpuContext::new().unwrap();
    let pipeline = pipeline(&context);
    let input = buffer(&context, &[1., 2., 3., 4., 5.]);
    let output = context
        .create_buffer(
            "private zero guards",
            40,
            BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        )
        .unwrap();
    let params = parameters(&context, 5, 2, 2., 3.);
    let command = dispatch(pipeline.clone(), &input, &output, &params, 5);
    let mut first = submit(
        &context,
        vec![(
            0,
            Operation::Compute {
                native: command,
                profile_index: 0,
            },
        )],
        false,
        false,
    )
    .unwrap();
    // Rewriting this packet while the first command is pending stages a new
    // snapshot. It must never overwrite parameters consumed by the first GPU.
    context
        .upload(
            &params,
            0,
            bytemuck::cast_slice(&[5u32, 2, 3f32.to_bits(), 7f32.to_bits()]),
        )
        .unwrap();
    first.wait().unwrap();
    let (values, submission) = readback(&context, &output, vec![], false);
    assert_eq!(values, [0., 0., 5., 7., 9., 11., 13., 0., 0., 0.]);
    assert!(submission.timestamps().unwrap().is_empty());
    let command = dispatch(pipeline, &input, &output, &params, 5);
    let (values, submission) = readback(
        &context,
        &output,
        vec![(
            0,
            Operation::Compute {
                native: command,
                profile_index: 3,
            },
        )],
        true,
    );
    assert_eq!(values, [0., 0., 10., 13., 16., 19., 22., 0., 0., 0.]);
    let timing = submission.timestamps().unwrap();
    assert_eq!(timing.len(), 1);
    assert_eq!(timing[0].0, 3);
    assert!(timing[0].1 > 0);
}

#[test]
#[ignore = "requires physical Apple Silicon Metal 4"]
fn native_context_range_usage_and_quota_validation() {
    let context = GpuContext::new_with_options(GpuContextOptions {
        max_device_bytes: Some(4096),
        ..Default::default()
    })
    .unwrap();
    let pipeline = pipeline(&context);
    let input = buffer(&context, &[2.; 4]);
    let output = buffer(&context, &[0.; 4]);
    let params = parameters(&context, 4, 0, 1., 0.);
    let used = context.memory_statistics().used_bytes;
    assert_eq!(
        context
            .create_buffer("quota", 8192, BufferUsage::STORAGE)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::ResourceLimit
    );
    assert_eq!(context.memory_statistics().used_bytes, used);
    let prepare = |buffers, offsets, groups| {
        execution::NativeDispatch::prepare(
            pipeline.clone(),
            buffers,
            offsets,
            groups,
            &[true, false],
        )
    };
    assert!(
        prepare(
            vec![input.clone(), output.clone(), params.clone()],
            vec![0; 3],
            [0, 1, 1]
        )
        .is_err()
    );
    assert!(
        prepare(
            vec![input.clone(), output.clone(), params.clone()],
            vec![0; 3],
            [65535, 65535, 1]
        )
        .is_err()
    );
    let foreign = GpuContext::new().unwrap();
    let wrong = buffer(&foreign, &[0.; 4]);
    assert_eq!(
        prepare(
            vec![input.clone(), wrong, params.clone()],
            vec![0; 3],
            [1; 3]
        )
        .unwrap_err()
        .kind(),
        ErrorKind::InvalidInput
    );
    assert!(
        prepare(
            vec![input.clone(), output.clone(), params.clone()],
            vec![0, 16, 0],
            [1; 3]
        )
        .is_err()
    );
    assert!(
        prepare(
            vec![input.clone(), output.clone(), params.clone()],
            vec![2, 0, 0],
            [1; 3]
        )
        .is_err()
    );
    assert!(
        prepare(
            vec![input.clone(), output.clone(), params.clone()],
            vec![0; 3],
            [65536, 1, 1]
        )
        .is_err()
    );
    assert!(
        prepare(
            vec![input.clone(), output.clone(), input.clone()],
            vec![0; 3],
            [1; 3]
        )
        .is_err()
    );
    let invalid = Operation::Copy {
        source: input.clone(),
        output: input.clone(),
        source_offset: 0,
        output_offset: 4,
        bytes: 8,
    };
    assert!(submit(&context, vec![(0, invalid)], false, false).is_err());
    assert!(
        execution::compile(
            &context,
            "invalid native source",
            "broken",
            [1; 3],
            vec![],
            vec![]
        )
        .is_err()
    );
    assert!(
        context.check_alive().is_ok(),
        "compile defects propagate without retiring a healthy queue"
    );
}

#[test]
#[ignore = "requires physical Apple Silicon Metal 4"]
fn native_concurrent_recordings_alias_dependencies_and_retirement() {
    let context = GpuContext::new_with_options(GpuContextOptions {
        max_in_flight: 4,
        ..Default::default()
    })
    .unwrap();
    let pipeline = pipeline(&context);
    let input = buffer(&context, &[1.; 129]);
    // Publish immutable shared weights once, then encode independent sessions
    // concurrently while sharing the same device, compiler and pipeline.
    let (_, published) = readback(&context, &input, vec![], false);
    drop(published);
    std::thread::scope(|scope| {
        for index in 0..4 {
            let context = context.clone();
            let pipeline = pipeline.clone();
            let input = input.clone();
            scope.spawn(move || {
                let output = buffer(&context, &[0.; 129]);
                let params = parameters(&context, 129, 0, index as f32 + 1., 2.);
                let first = dispatch(pipeline.clone(), &input, &output, &params, 129);
                let second = dispatch(pipeline, &output, &output, &params, 129);
                let (values, _) = readback(
                    &context,
                    &output,
                    vec![
                        (
                            0,
                            Operation::Compute {
                                native: first,
                                profile_index: 0,
                            },
                        ),
                        (
                            1,
                            Operation::Compute {
                                native: second,
                                profile_index: 1,
                            },
                        ),
                    ],
                    false,
                );
                assert!(
                    values
                        .iter()
                        .all(|&v| v == (index as f32 + 3.) * (index as f32 + 1.) + 2.)
                );
            });
        }
    });
    drop(input);
    drop(pipeline);
    assert_eq!(
        context.memory_statistics().used_bytes,
        0,
        "completed native work releases logical reservations"
    );
    assert!(
        context.memory_statistics().committed_bytes > 0,
        "reusable native arena capacity is reported separately"
    );
}

#[test]
#[ignore = "requires physical Apple Silicon Metal 4"]
fn native_terminal_failure_wakes_admission_and_blocks_publication() {
    let context = GpuContext::new().unwrap();
    let gate = context.acquire_gate().unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    let waiting = context.clone();
    let thread = std::thread::spawn(move || {
        send.send(waiting.acquire_gate().err().unwrap().kind())
            .unwrap();
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while context.waiting.load(Ordering::Acquire) == 0 {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    context.backend.owner.fail(RuntimeError::with_kind(
        ErrorKind::DeviceLost,
        "bounded injected retirement",
    ));
    assert_eq!(
        receive.recv_timeout(Duration::from_secs(2)).unwrap(),
        ErrorKind::DeviceLost
    );
    assert!(submit(&context, vec![], false, false).is_err());
    drop(gate);
    thread.join().unwrap();
}

#[test]
#[ignore = "requires physical Apple Silicon Metal 4 runtime compiler"]
fn native_catalog_compiles_both_storage_abis() {
    let context = GpuContext::new().unwrap();
    assert_eq!(std::mem::size_of::<[u32; 128]>(), 512);
    assert_eq!(std::mem::size_of::<[u32; 256]>(), 1024);
    for &kind in operators::ShaderKind::ALL {
        for storage in [GpuPrecision::F32, GpuPrecision::MixedF16] {
            let pipeline = shaders::neural(&context, kind, [storage; 6], &[0; 128]).unwrap();
            assert_eq!(pipeline.minimum_bytes[6], 512);
            assert_eq!(pipeline.read_only, [true, true, true, true, true, false]);
            assert_eq!(pipeline.workgroup_size, [64, 1, 1]);
        }
    }
}
