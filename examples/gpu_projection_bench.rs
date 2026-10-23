//! Physical timestamp comparison: exactly one warmup and three samples per candidate.
use std::{error::Error, time::Duration};
use wgpu::util::DeviceExt;
fn main() -> Result<(), Box<dyn Error>> {
    assert!(!cfg!(debug_assertions), "release build required");
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    assert!(matches!(
        adapter.get_info().device_type,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
    ));
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::TIMESTAMP_QUERY,
        ..Default::default()
    }))?;
    let buffer = |label, bytes: &[u8], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage,
        })
    };
    let query = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve = buffer(
        "timestamps",
        &[0; 16],
        wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
    );
    let times = buffer(
        "times",
        &[0; 16],
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    for (rows, inner, classes) in [(16u32, 80u32, 6906u32), (80, 80, 6906), (80, 192, 18710)] {
        let tiles = classes.div_ceil(64);
        let values = |n: u32| {
            (0..n)
                .map(|i| ((i * 37 + 19) % 127) as f32 / 63. - 1.)
                .collect::<Vec<_>>()
        };
        let a = buffer(
            "features",
            bytemuck::cast_slice(&values(rows * inner)),
            wgpu::BufferUsages::STORAGE,
        );
        let b = buffer(
            "weights",
            bytemuck::cast_slice(&values(inner * classes)),
            wgpu::BufferUsages::STORAGE,
        );
        let c = buffer(
            "bias",
            bytemuck::cast_slice(&values(classes)),
            wgpu::BufferUsages::STORAGE,
        );
        let bytes = (rows * tiles * 16) as usize;
        let output = buffer(
            "partial",
            &vec![0; bytes],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = buffer(
            "check",
            &vec![0; bytes],
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut words = [0u32; 256];
        words[..5].copy_from_slice(&[rows, inner, classes, tiles, 1]);
        let params = buffer(
            "parameters",
            bytemuck::cast_slice(&words),
            wgpu::BufferUsages::UNIFORM,
        );
        let mut reference: Option<Vec<u8>> = None;
        for (tile, kernel) in [
            (1, include_str!("../src/runtime/gpu/heads.wgsl")),
            (4, include_str!("../src/runtime/gpu/heads4.wgsl")),
        ] {
            let source = format!(
                "struct Params {{words:array<vec4<u32>,64>}};\n{}",
                kernel.replace("$INPUT", "f32")
            );
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[&a, &b, &c, &output, &params]
                    .into_iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            let mut measurements = Vec::new();
            for iteration in 0..4 {
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &query,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        }),
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bindings, &[]);
                    pass.dispatch_workgroups(tiles, rows.div_ceil(tile), 1);
                }
                encoder.resolve_query_set(&query, 0..2, &resolve, 0);
                let mut copy = device.create_command_encoder(&Default::default());
                copy.copy_buffer_to_buffer(&resolve, 0, &times, 0, 16);
                copy.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes as u64);
                let submitted = queue.submit([encoder.finish(), copy.finish()]);
                times
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, |r| r.unwrap());
                readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, |r| r.unwrap());
                device.poll(wgpu::PollType::Wait {
                    submission_index: Some(submitted),
                    timeout: Some(Duration::from_secs(30)),
                })?;
                {
                    let data = times.slice(..).get_mapped_range()?;
                    let start = u64::from_le_bytes(data[..8].try_into()?);
                    let end = u64::from_le_bytes(data[8..].try_into()?);
                    if iteration > 0 {
                        measurements.push(
                            (end - start) as f64 * queue.get_timestamp_period() as f64 / 1000.,
                        );
                    }
                }
                if iteration == 0 {
                    let data = readback.slice(..).get_mapped_range()?;
                    if let Some(expected) = &reference {
                        for (a, b) in data.chunks_exact(16).zip(expected.chunks_exact(16)) {
                            assert_eq!(&a[..4], &b[..4]);
                            for offset in [4, 8] {
                                let x = f32::from_le_bytes(a[offset..offset + 4].try_into()?);
                                let y = f32::from_le_bytes(b[offset..offset + 4].try_into()?);
                                assert!((x - y).abs() <= 1e-3 + 1e-3 * y.abs());
                            }
                        }
                    } else {
                        reference = Some(data.to_vec());
                    }
                }
                times.unmap();
                readback.unmap();
            }
            println!(
                "rows={rows} inner={inner} classes={classes} tile={tile} device_us={measurements:?}"
            );
        }
    }
    Ok(())
}
