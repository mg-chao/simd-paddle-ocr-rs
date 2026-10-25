#[cfg(all(feature = "gpu", not(target_os = "macos")))]
fn main() {
    use std::{
        collections::BTreeMap,
        env, fs,
        hash::{Hash, Hasher},
        path::PathBuf,
    };
    // A build script runs on the host; cross-checking the preserved Metal
    // backend from Windows must not compile Windows shaders for that target.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        return;
    }
    let root = PathBuf::from("src/runtime/gpu/vulkan/shaders");
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-changed=build.rs");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let catalog: Vec<serde_json::Value> =
        serde_json::from_slice(&fs::read(root.join("catalog.json")).unwrap()).unwrap();
    let compiler = shaderc::Compiler::new().unwrap();
    let mut base = shaderc::CompileOptions::new().unwrap();
    base.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );
    base.set_target_spirv(shaderc::SpirvVersion::V1_6);
    base.set_optimization_level(shaderc::OptimizationLevel::Performance);
    base.set_warnings_as_errors();
    base.set_include_callback(|name, _, _, _| {
        if !matches!(name, "neural.glsl" | "storage.glsl") {
            return Err(format!("unknown shader include: {name}"));
        }
        Ok(shaderc::ResolvedInclude {
            resolved_name: name.into(),
            content: fs::read_to_string(root.join(name)).map_err(|e| e.to_string())?,
        })
    });
    let common = format!(
        "{}{}{}",
        fs::read_to_string(root.join("neural.glsl")).unwrap(),
        fs::read_to_string(root.join("storage.glsl")).unwrap(),
        "shaderc-0.10.1-vulkan1.3-spirv1.6-performance-v1"
    );
    let mut generated = String::from(
        "// Compiled native Vulkan shader catalog.\nfn binary(key: KernelKey) -> &'static ShaderBinary {\n match (key.id, key.precision) {\n",
    );
    for entry in &catalog {
        let name = entry["name"].as_str().unwrap();
        let id = name
            .split('_')
            .map(|s| {
                let mut c = s.chars();
                c.next().unwrap().to_uppercase().to_string() + c.as_str()
            })
            .collect::<String>();
        let mask = entry["precision_mask"].as_u64().unwrap() as u8;
        let source = fs::read_to_string(root.join(format!("{name}.comp")))
            .unwrap()
            .replace(
                "#version 450 core",
                "#version 450 core\n#extension GL_GOOGLE_include_directive : require",
            );
        let wg: Vec<u32> = serde_json::from_value(entry["workgroup"].clone()).unwrap();
        for precision in 0..64u8 {
            if precision & !mask != 0 {
                continue;
            }
            let file = out.join(format!("{name}-{precision}.spv"));
            let stamp = file.with_extension("stamp");
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            (&source, &common, precision).hash(&mut hash);
            let fingerprint = hash.finish().to_string();
            if !file.exists() || fs::read_to_string(&stamp).ok().as_ref() != Some(&fingerprint) {
                let mut options = base.clone();
                let value = precision.to_string();
                options.add_macro_definition("PRECISION_MASK", Some(&value));
                let compiled = compiler
                    .compile_into_spirv(
                        &source,
                        shaderc::ShaderKind::Compute,
                        name,
                        "main",
                        Some(&options),
                    )
                    .unwrap_or_else(|e| panic!("{name}/{precision}: {e}"));
                fs::write(&file, compiled.as_binary_u8()).unwrap();
                fs::write(stamp, &fingerprint).unwrap();
            }
            let bytes = fs::read(&file).unwrap();
            let words: Vec<u32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            assert_eq!(words[0], 0x07230203, "SPIR-V magic");
            // Reflect the compiled interface, including bindings removed by optimization.
            let mut workgroup = None;
            let mut workgroup_ids = None;
            let mut spec_ids = BTreeMap::new();
            let mut locations = BTreeMap::new();
            let mut sets = BTreeMap::new();
            let mut types = BTreeMap::new();
            let mut variables = BTreeMap::new();
            let mut constants = BTreeMap::new();
            let mut strides = BTreeMap::new();
            let mut offsets = BTreeMap::new();
            let mut read_only = std::collections::BTreeSet::new();
            let mut at = 5;
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                let op = words[at] & 65535;
                assert!(n > 0 && at + n <= words.len(), "invalid SPIR-V instruction");
                if op == 71 && n >= 4 {
                    match words[at + 2] {
                        1 => {
                            spec_ids.insert(words[at + 1], words[at + 3]);
                        }
                        33 => {
                            locations.insert(words[at + 1], words[at + 3]);
                        }
                        34 => {
                            sets.insert(words[at + 1], words[at + 3]);
                        }
                        6 => {
                            strides.insert(words[at + 1], words[at + 3]);
                        }
                        _ => {}
                    }
                }
                if op == 71 && n == 3 && words[at + 2] == 24 {
                    read_only.insert(words[at + 1]);
                }
                if op == 72 && n >= 4 {
                    if words[at + 3] == 35 {
                        offsets.insert((words[at + 1], words[at + 2]), words[at + 4]);
                    }
                    if words[at + 3] == 24 {
                        read_only.insert(words[at + 1]);
                    }
                }
                if (21..=32).contains(&op) {
                    types.insert(words[at + 1], (op, words[at + 2..at + n].to_vec()));
                }
                if op == 43 || op == 50 {
                    constants.insert(words[at + 2], words[at + 3]);
                }
                if op == 16 && n == 6 && words[at + 2] == 17 {
                    workgroup = Some(words[at + 3..at + 6].to_vec());
                }
                if op == 331 && n == 6 && words[at + 2] == 38 {
                    workgroup_ids = Some(words[at + 3..at + 6].to_vec());
                }
                if op == 59 {
                    variables.insert(words[at + 2], (words[at + 1], words[at + 3]));
                }
                at += n;
            }
            if let Some(ids) = workgroup_ids {
                workgroup = Some(ids.iter().map(|id| constants[id]).collect());
            }
            assert_eq!(
                workgroup.as_ref(),
                Some(&wg),
                "{name}/{precision}: workgroup size"
            );
            let expected_specs: BTreeMap<u32, u32> = entry["specialization"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|s| {
                    (
                        s["id"].as_u64().unwrap() as u32,
                        s["value"].as_u64().unwrap() as u32,
                    )
                })
                .collect();
            let actual_specs: BTreeMap<u32, u32> = spec_ids
                .iter()
                .map(|(variable, id)| (*id, constants[variable]))
                .collect();
            assert_eq!(
                actual_specs, expected_specs,
                "{name}/{precision}: specialization interface"
            );
            let mut bindings = Vec::new();
            for (variable, slot) in locations {
                assert_eq!(
                    sets.get(&variable),
                    Some(&0),
                    "only descriptor set zero is supported"
                );
                let meta = entry["bindings"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|b| b["slot"].as_u64() == Some(slot as u64))
                    .expect("undeclared shader binding");
                let uniform = meta["uniform"].as_bool().unwrap();
                let read = uniform || meta["read"].as_bool().unwrap();
                let minimum = meta["minimum"].as_u64().unwrap();
                let (pointer, storage) = variables[&variable];
                assert_eq!(
                    storage,
                    if uniform { 2 } else { 12 },
                    "{name}/{precision} binding {slot}: descriptor type"
                );
                let (_, pointer_type) = &types[&pointer];
                let block = pointer_type[1];
                if !uniform {
                    assert_eq!(
                        read,
                        read_only.contains(&block) || read_only.contains(&variable),
                        "{name}/{precision} binding {slot}: access"
                    );
                }
                if uniform {
                    fn size(
                        id: u32,
                        types: &BTreeMap<u32, (u32, Vec<u32>)>,
                        constants: &BTreeMap<u32, u32>,
                        strides: &BTreeMap<u32, u32>,
                        offsets: &BTreeMap<(u32, u32), u32>,
                    ) -> u64 {
                        let (op, args) = &types[&id];
                        match op {
                            21 | 22 => u64::from(args[0] / 8),
                            23 => {
                                size(args[0], types, constants, strides, offsets)
                                    * u64::from(args[1])
                            }
                            28 => u64::from(strides[&id]) * u64::from(constants[&args[1]]),
                            30 => args
                                .iter()
                                .enumerate()
                                .map(|(i, ty)| {
                                    u64::from(offsets[&(id, i as u32)])
                                        + size(*ty, types, constants, strides, offsets)
                                })
                                .max()
                                .unwrap_or(0),
                            _ => panic!("unsupported uniform type {op}"),
                        }
                    }
                    assert_eq!(
                        minimum,
                        size(block, &types, &constants, &strides, &offsets),
                        "{name}/{precision} binding {slot}: uniform packet size"
                    );
                }
                bindings.push(format!("ShaderBinding{{slot:{slot},uniform:{uniform},read_only:{read},minimum:{minimum}}}"));
            }
            let specialization = entry["specialization"]
                .as_array()
                .map(|values| {
                    values
                        .iter()
                        .map(|v| format!("({}, {})", v["id"], v["value"]))
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            if name.ends_with("_probe") {
                generated += "#[cfg(test)]\n";
            }
            generated += &format!(
                "(KernelId::{id},{precision}) => &ShaderBinary{{bytes:include_bytes!({:?}),workgroup:[{},{},{}],bindings:&[{}],specialization:&[{specialization}]}},\n",
                file.display().to_string(),
                wg[0],
                wg[1],
                wg[2],
                bindings.join(",")
            );
        }
    }
    generated +=
        "_ => unreachable!(\"kernel precision was validated before catalog lookup\"),\n }\n}\n";
    fs::write(out.join("vulkan_shaders.rs"), generated).unwrap();
}

#[cfg(any(not(feature = "gpu"), target_os = "macos"))]
fn main() {}
