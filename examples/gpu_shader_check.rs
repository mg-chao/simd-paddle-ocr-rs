#![recursion_limit = "256"]
//! Developer-only runtime loading for validation of the resident WGSL modules.
use std::{error::Error, path::Path};
#[allow(clippy::assertions_on_constants)]
fn main() -> Result<(), Box<dyn Error>> {
    let backend = match std::env::var("SIMD_OCR_GPU_BACKEND").as_deref() {
        Ok("dx12") => wgpu::Backends::DX12,
        Ok("metal") => wgpu::Backends::METAL,
        _ => wgpu::Backends::VULKAN,
    };
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    assert!(matches!(
        adapter.get_info().device_type,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
    ));
    let features = adapter.features() & wgpu::Features::SHADER_F16;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        ..Default::default()
    }))?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/runtime/gpu");
    let common = std::fs::read_to_string(root.join("common.wgsl"))?;
    let coop = std::fs::read_to_string(root.join("cooperative.wgsl"))?;
    let mut code =
        format!("enable wgpu_cooperative_matrix;\n{common}\n{coop}").replace("$ENABLE", "");
    for token in ["$A", "$B", "$C", "$D", "$E", "$O"] {
        code = code.replace(token, "f32");
    }
    let module = wgpu::naga::front::wgsl::parse_str(&code).map_err(|e| e.emit_to_string(&code))?;
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)?;
    println!(
        "cooperative/f32 Naga validation PASS; hardware execution requires advertised capability"
    );
    for name in [
        "image",
        "heads",
        "heads4",
        "head_reduce",
        "ctc",
        "db_label",
        "db_rows",
        "db_geometry",
        "db_sort",
        "crop_meta",
        "crop",
        "rotate",
        "pack_bgr",
    ] {
        let raw = std::fs::read_to_string(root.join(format!("{name}.wgsl")))?;
        for half in [false, true] {
            if half && (!raw.contains('$') || !features.contains(wgpu::Features::SHADER_F16)) {
                continue;
            }
            let mut source = String::new();
            if half {
                source.push_str("enable f16;\n");
            }
            source.push_str("struct Params { words: array<vec4<u32>,64> };\n");
            if matches!(name, "crop_meta" | "crop" | "db_geometry") {
                source.push_str(&std::fs::read_to_string(root.join("precise.wgsl"))?);
            }
            source.push_str(
                &raw.replace("$INPUT", if half { "f16" } else { "f32" })
                    .replace("$OUT", if half { "f16" } else { "f32" }),
            );
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            if let Some(error) = pollster::block_on(scope.pop()) {
                return Err(format!("{name}: {error}").into());
            }
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            if let Some(error) = pollster::block_on(scope.pop()) {
                return Err(format!("{name}: {error}").into());
            }
            println!("{name}/{} PASS", if half { "f16" } else { "f32" });
        }
    }
    check_rounding(&device, &queue, &root)?;
    Ok(())
}

#[allow(clippy::assertions_on_constants)]
fn check_rounding(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    root: &Path,
) -> Result<(), Box<dyn Error>> {
    use wgpu::util::DeviceExt;
    assert!(
        !cfg!(debug_assertions),
        "physical arithmetic checks require --release"
    );
    let cases = [
        [173.0f32, -0.000001],
        [173.0, 0.000001],
        [-173.0, -0.000001],
        [-173.0, 0.000001],
        [173.5, -0.000001],
        [173.5, 0.000001],
        [-173.5, -0.000001],
        [-173.5, 0.000001],
    ];
    let source = format!(
        "struct Params {{words:array<vec4<u32>,64>}};\n{}
        @group(0) @binding(0) var<storage,read> input:array<vec2<f32>>;
        @group(0) @binding(1) var<storage,read_write> output:array<f32>;
        @group(0) @binding(2) var<uniform> params:Params;
        @compute @workgroup_size(64) #[allow(clippy::assertions_on_constants)]
fn main(@builtin(global_invocation_id) id:vec3<u32>){{
            if id.x>=params.words[0].x{{return;}}let a=input[id.x];
            output[id.x*3u]=dfloor(a);output[id.x*3u+1u]=daway(a);output[id.x*3u+2u]=dround(a);
        }}",
        std::fs::read_to_string(root.join("precise.wgsl"))?
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let init = |bytes: &[u8], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage,
        })
    };
    let input = init(bytemuck::cast_slice(&cases), wgpu::BufferUsages::STORAGE);
    let size = cases.len() * 12;
    let output = init(
        &vec![0; size],
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = init(
        &vec![0; size],
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut words = [0u32; 256];
    words[0] = cases.len() as u32;
    words[255] = u32::MAX;
    let params = init(bytemuck::cast_slice(&words), wgpu::BufferUsages::UNIFORM);
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[&input, &output, &params]
            .into_iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size as u64);
    let submission = queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: Some(std::time::Duration::from_secs(30)),
    })?;
    rx.recv()??;
    let bytes = readback.slice(..).get_mapped_range()?;
    let actual: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    for (i, [hi, lo]) in cases.iter().copied().enumerate() {
        let x = hi as f64 + lo as f64;
        let expected = [
            x.floor() as f32,
            x.round() as f32,
            x.round_ties_even() as f32,
        ];
        if actual[i * 3..i * 3 + 3] != expected {
            return Err(format!(
                "two-FP32 rounding {hi}+{lo}: {:?} vs {expected:?}",
                &actual[i * 3..i * 3 + 3]
            )
            .into());
        }
    }
    println!("two-FP32 floor/round boundary arithmetic PASS (8 cases)");
    Ok(())
}
