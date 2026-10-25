use super::*;

fn packet() -> [u32; 128] {
    let mut words = [0; 128];
    for (i, value) in [
        (0, 20),
        (1, 20),
        (2, 7),
        (3, 17),
        (4, 17),
        (5, 15),
        (6, 15),
        (7, 1f32.to_bits()),
        (8, 5),
        (15, 2),
        (60, 7),
    ] {
        words[i] = value;
    }
    words
}

#[test]
fn attention_simd_admission_preserves_storage_packet_and_resource_guards() {
    let words = packet();
    let limits = crate::runtime::gpu::GpuLimits::default();
    let allowed = |w: &[u32; 128], l: &crate::runtime::gpu::GpuLimits| {
        eligible(
            ShaderKind::AttentionViews,
            [GpuPrecision::MixedF16; 6],
            w,
            l,
        )
    };
    assert!(allowed(&words, &limits));
    for (index, value) in [
        (0, 0),
        (0, 19),
        (1, 0),
        (1, u32::MAX),
        (2, 3),
        (2, 6),
        (3, 0),
        (4, 18),
        (5, 0),
        (5, 33),
        (6, 14),
        (7, f32::NAN.to_bits()),
        (8, 0),
        (8, 4),
        (9, 1),
        (15, 0),
        (15, 3),
        (60, u32::MAX),
        (64, 1),
    ] {
        let mut invalid = words;
        invalid[index] = value;
        assert!(!allowed(&invalid, &limits), "word{index}={value}");
    }
    assert!(!eligible(
        ShaderKind::Attention,
        [GpuPrecision::MixedF16; 6],
        &words,
        &limits
    ));
    for binding in 0..6 {
        let mut storage = [GpuPrecision::MixedF16; 6];
        storage[binding] = GpuPrecision::F32;
        assert!(!eligible(
            ShaderKind::AttentionViews,
            storage,
            &words,
            &limits
        ));
    }
    for field in 0..3 {
        let mut limited = limits.clone();
        match field {
            0 => limited.max_compute_workgroup_size_x = 63,
            1 => limited.max_compute_invocations_per_workgroup = 63,
            _ => limited.max_compute_workgroup_storage_size = 4911,
        }
        assert!(!allowed(&words, &limited));
    }
    let mut large = words;
    large[3] = u32::MAX;
    large[4] = u32::MAX;
    large[8] = u32::MAX.div_ceil(4);
    large[0] = large[8] * 2;
    assert!(!allowed(&large, &limits));
}

fn body<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(name).unwrap();
    let brace = start + source[start..].find('{').unwrap();
    let mut depth = 0;
    for (i, byte) in source.as_bytes().iter().enumerate().skip(brace) {
        if *byte == b'{' {
            depth += 1;
        }
        if *byte == b'}' {
            depth -= 1;
            if depth == 0 {
                return &source[brace..=i];
            }
        }
    }
    panic!("unclosed source body")
}

#[test]
fn attention_simd_source_matches_qualified_bytes_and_current_naga_abi() {
    assert_eq!(source().len(), 31566);
    assert_eq!(
        source()
            .bytes()
            .fold(0xcbf29ce484222325u64, |h, b| (h ^ u64::from(b))
                .wrapping_mul(0x100000001b3)),
        0xb38f47365fb5d588
    );
    let mut wgsl = format!(
        "{}\n{}",
        ShaderKind::Attention.metal_common_source(),
        include_str!("attention.wgsl")
    )
    .replace("$ENABLE", "enable f16;")
    .replace("$HALF_OUTPUT", "true")
    .replace("$VIEWS", "true");
    for operand in ["A", "B", "C", "D", "E", "O"] {
        wgsl = wgsl.replace(&format!("${operand}"), "f16");
    }
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::back::msl::supported_capabilities(),
    )
    .validate(&module)
    .unwrap();
    let entry = info.get_entry_point(0);
    let mut resources = naga::back::msl::EntryPointResources {
        sizes_buffer: Some(30),
        ..Default::default()
    };
    let mut sized = Vec::new();
    let mut memory = Vec::new();
    let mut required = Vec::new();
    for (handle, global) in module.global_variables.iter() {
        if global.space == naga::AddressSpace::WorkGroup && !entry[handle].is_empty() {
            memory.push(
                (module.types[global.ty].inner.size(module.to_ctx()) as usize).div_ceil(16) * 16,
            );
        }
        if let Some(binding) = global.binding {
            let mutable = matches!(global.space, naga::AddressSpace::Storage {access} if access.contains(naga::StorageAccess::STORE));
            resources.resources.insert(
                binding,
                naga::back::msl::BindTarget {
                    buffer: Some(binding.binding as u8),
                    mutable,
                    ..Default::default()
                },
            );
            if !entry[handle].is_empty() {
                required.push((
                    binding.binding as usize,
                    u64::from(module.types[global.ty].inner.size(module.to_ctx())),
                ));
            }
            if module.types[global.ty]
                .inner
                .needs_host_buffer_byte_size(&module.types)
            {
                sized.push(binding.binding as usize);
            }
        }
    }
    assert_eq!(sized, SIZED_BINDINGS);
    assert_eq!(memory, WORKGROUP_MEMORY);
    assert_eq!(required, REQUIRED_BINDINGS);
    assert_eq!(module.entry_points[0].workgroup_size, WORKGROUP_SIZE);
    let options = naga::back::msl::Options {
        lang_version: (4, 0),
        per_entry_point_map: std::collections::BTreeMap::from([("main".to_owned(), resources)]),
        fake_missing_bindings: false,
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Restrict,
            buffer: naga::proc::BoundsCheckPolicy::Restrict,
            image_load: naga::proc::BoundsCheckPolicy::Restrict,
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: true,
        ..Default::default()
    };
    let (baseline, _) = naga::back::msl::write_string(
        &module,
        &info,
        &options,
        &naga::back::msl::PipelineOptions {
            entry_point: Some((naga::ShaderStage::Compute, "main".into())),
            ..Default::default()
        },
    )
    .unwrap();
    let marker = "[[max_total_threads_per_threadgroup(64)]] kernel void main_(";
    assert!(source().starts_with(&baseline[..baseline.find(marker).unwrap()]));
    assert_eq!(
        body(source(), "void attention_original("),
        body(&baseline, "kernel void main_(")
    );
    assert_eq!(
        body(source(), "kernel void main_(")
            .matches("threadgroup_barrier")
            .count(),
        3
    );
    assert!(source().contains("simd_width != 32u"));
    assert!(source().contains("metal::simd_shuffle(score, half_base + k)"));
    assert!(source().contains("metal::simd_shuffle(local_weight, half_base + k)"));
    assert_eq!(ENTRY, "main_");
}
