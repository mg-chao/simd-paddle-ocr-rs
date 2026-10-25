use super::*;

const SOURCE: &str = r#"
struct Params { words: array<vec4<u32>, 32> }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(1)
fn main() { output[0] = input[0] + f32(params.words[0].x); }
"#;

#[test]
#[ignore = "requires a physical Vulkan GPU; prepared packet offsets and interleaved compute/copies"]
fn vulkan_parameter_arena_and_transfer_order() {
    let context = GpuContext::new().unwrap();
    assert_eq!(context.info.backend, wgpu::Backend::Vulkan);
    let pipeline = compile_wgsl(&context, SOURCE, "main", [0; 3]).unwrap();
    let input = context
        .create_buffer_init(
            "input",
            bytemuck::bytes_of(&5.0f32),
            wgpu::BufferUsages::STORAGE,
        )
        .unwrap();
    let output = context
        .create_buffer(
            "output",
            4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
        .unwrap();
    // The whole arena exceeds max_uniform_buffer_binding_size, but each
    // immutable dispatch binds exactly its 512-byte packet.
    let second = context.limits.max_uniform_buffer_binding_size as usize;
    let mut data = vec![0u8; second + 512];
    data[..4].copy_from_slice(&7u32.to_le_bytes());
    data[second..second + 4].copy_from_slice(&11u32.to_le_bytes());
    let params = context
        .create_buffer_init("large parameter arena", &data, wgpu::BufferUsages::UNIFORM)
        .unwrap();
    let dispatches = context
        .checked("prepare packets", || {
            [0, second]
                .into_iter()
                .map(|offset| {
                    NativeDispatch::prepare(
                        pipeline.clone(),
                        vec![input.clone(), output.clone(), params.clone()],
                        vec![0, 0, offset as u64],
                        [1; 3],
                        &[true, false],
                    )
                })
                .collect::<RuntimeResult<Vec<_>>>()
        })
        .unwrap();
    let readback = context
        .create_buffer(
            "readback",
            8,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        )
        .unwrap();
    let mut encoder = context.device.create_command_encoder(&Default::default());
    for (i, dispatch) in dispatches.iter().enumerate() {
        encode(&context, &mut encoder, &[(i, dispatch.as_ref())], true).unwrap();
        encoder.copy_buffer_to_buffer(&output, 0, &readback, (i * 4) as u64, 4);
    }
    let submission = context
        .checked("submit packets", || {
            Ok(context.queue.submit([encoder.finish()]))
        })
        .unwrap();
    let (send, receive) = mpsc::sync_channel(1);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).unwrap();
        });
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .unwrap();
    receive.recv().unwrap().unwrap();
    let view = readback.slice(..).get_mapped_range().unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, f32>(&view), [12.0, 16.0]);
    drop(view);
    readback.unmap();
    context.check_alive().unwrap();
}

#[test]
#[ignore = "requires a physical Vulkan GPU; rejects foreign resources, short uniforms and invalid access before encoding"]
fn vulkan_prepared_bindings_validate_ownership_and_shader_access() {
    let context = GpuContext::new().unwrap();
    let pipeline = compile_wgsl(&context, SOURCE, "main", [0; 3]).unwrap();
    let input = context
        .create_buffer("input", 4, wgpu::BufferUsages::STORAGE)
        .unwrap();
    let output = context
        .create_buffer("output", 4, wgpu::BufferUsages::STORAGE)
        .unwrap();
    let params = context
        .create_buffer("parameters", 512, wgpu::BufferUsages::UNIFORM)
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
        .create_buffer("short", 16, wgpu::BufferUsages::UNIFORM)
        .unwrap();
    assert_eq!(
        prepare(input.clone(), short, &[true, false])
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidInput
    );
    let foreign = GpuContext::new().unwrap();
    let other = foreign
        .create_buffer("foreign", 4, wgpu::BufferUsages::STORAGE)
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
