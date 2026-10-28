use std::{env, fs, path::PathBuf, process::Command};
#[path = "catalog.rs"]
mod catalog;

fn run(command: &mut Command) {
    let result = command
        .output()
        .unwrap_or_else(|e| panic!("cannot run {command:?}: {e}"));
    assert!(
        result.status.success(),
        "{command:?} failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

pub fn build() {
    println!("cargo:rerun-if-changed=src/runtime/gpu/metal");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=TOOLCHAINS");
    println!("cargo:rerun-if-env-changed=SIMD_OCR_METAL_TOOLCHAIN");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let toolchain = env::var_os("SIMD_OCR_METAL_TOOLCHAIN").map(PathBuf::from);
    let tool = |name: &str| {
        if let Some(root) = &toolchain {
            Command::new(root.join("usr/bin").join(name))
        } else {
            let mut cmd = Command::new("xcrun");
            cmd.args(["--sdk", "macosx", name]);
            cmd
        }
    };
    let version = tool("metal").arg("--version").output().expect("Metal compiler requires Xcode and its Metal Toolchain; run xcodebuild -downloadComponent MetalToolchain");
    assert!(
        version.status.success(),
        "Metal compiler unavailable: {}",
        String::from_utf8_lossy(&version.stderr)
    );
    let fingerprint = format!(
        "{}\nmetal3.1;macos14;neural-relaxed-v1",
        String::from_utf8_lossy(&version.stdout)
    );
    let compiler_changed = fs::read_to_string(out.join("metal-compiler.txt"))
        .ok()
        .as_deref()
        != Some(fingerprint.as_str());
    let mut objects = Vec::new();
    let mut changed = compiler_changed;
    // Relinking also depends on the ordered set of objects. Removing a catalog
    // entry can leave every surviving shader unchanged; the old library must
    // not retain that removed entry (or its embedded resource footprint).
    let inventory = catalog::ENTRIES
        .iter()
        .map(|entry| entry.id)
        .collect::<Vec<_>>()
        .join("\n");
    let inventory_path = out.join("metal-objects.txt");
    changed |= fs::read_to_string(&inventory_path).ok().as_deref() != Some(inventory.as_str())
        || !out.join("ocr.metallib").exists();
    let mut native_pipelines = Vec::new();
    let mut generated = String::from(
        "// Generated from the native Metal catalog.\nfn shader(key: KernelKey) -> RuntimeResult<Shader> {\nlet (name, threads, reads, packet) = match key.id {\n",
    );
    for entry in catalog::ENTRIES {
        let source = catalog::source(entry);
        let math_mode = if entry.neural {
            "-fmetal-math-mode=relaxed"
        } else {
            "-fno-fast-math"
        };
        let input = out.join(format!("{}.metal", entry.id));
        let object = out.join(format!("{}.air", entry.id));
        if compiler_changed
            || fs::read_to_string(&input).ok().as_deref() != Some(source.as_str())
            || !object.exists()
        {
            changed = true;
            fs::write(&input, source).unwrap();
            run(tool("metal")
                .args([
                    "-std=metal3.1",
                    "-mmacosx-version-min=14.0",
                    math_mode,
                    "-c",
                ])
                .arg(&input)
                .arg("-o")
                .arg(&object));
        }
        objects.push(object);
        for precision in 0..64u8 {
            // Ship native binaries for the two homogeneous storage families.
            // All mixed binding signatures remain in portable IR and compile
            // on demand. This keeps the common startup archive compact.
            if precision == 0 || precision == entry.mask {
                // Match the runtime MTLComputePipelineDescriptor defaults
                // explicitly; translator defaults can differ across SDKs.
                native_pipelines.push(serde_json::json!({
                    "compute_function":format!("alias:OCR#ocr_{}_{}", entry.id, precision),
                    "threadgroup_size_is_multiple_of_thread_execution_width":false,
                    "max_total_threads_per_threadgroup":0,
                    "max_call_stack_depth":1,
                    "support_indirect_command_buffers":false,
                    "support_adding_binary_functions":false,
                    "optimize_for_persistent_kernel":false,
                    "forward_progress_usage":"Automatic",
                    "contention_relief":"Automatic"
                }));
            }
        }
        if entry.test {
            generated.push_str("#[cfg(test)]\n");
        }
        generated.push_str(&format!(
            "KernelId::{} => (\"{}\", {}, &{:?}[..], {}),\n",
            entry.id,
            entry.id,
            entry.threads,
            entry.reads,
            if entry.neural { 512 } else { 1024 }
        ));
    }
    generated.push_str("};\nif key.precision & !key.id.precision_mask() != 0 { return Err(resource(\"invalid Metal precision signature\")); }\nOk(Shader { name: format!(\"ocr_{}_{}\", name, key.precision), threads, reads, packet })\n}\n");
    fs::write(out.join("metal_catalog.rs"), generated).unwrap();
    if changed {
        run(tool("metallib")
            .args(&objects)
            .arg("-o")
            .arg(out.join("ocr.metallib")));
        fs::write(inventory_path, inventory).unwrap();
    }
    let script = serde_json::to_string(&serde_json::json!({
        "libraries":{"paths":[{"label":"OCR", "path":out.join("ocr.metallib")}]},
        "pipelines":{"compute_pipelines":native_pipelines}
    }))
    .unwrap();
    let script_path = out.join("ocr.mtlp-json");
    // Native pipeline binaries are OS-version specific, unlike the portable IR.
    let sdk = Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-version"])
        .output()
        .expect("query macOS SDK version");
    assert!(sdk.status.success(), "cannot query macOS SDK version");
    let sdk_version = String::from_utf8(sdk.stdout).unwrap();
    let sdk_major: u32 = sdk_version
        .trim()
        .split('.')
        .next()
        .unwrap()
        .parse()
        .expect("invalid SDK version");
    let native_target = format!("air64-apple-macos{sdk_major}.0");
    let native_fingerprint = format!("{native_target};applegpu_g16g;reflection-v1");
    println!("cargo:rustc-env=SIMD_OCR_METAL_ARCHIVE_OS={sdk_major}");
    if changed
        || fs::read_to_string(out.join("native-target.txt"))
            .ok()
            .as_deref()
            != Some(native_fingerprint.as_str())
        || fs::read_to_string(&script_path).ok().as_deref() != Some(script.as_str())
        || !out.join("ocr.binary.metallib").exists()
    {
        fs::write(&script_path, script).unwrap();
        // The physical M4 target reports applegpu_g16g. Other Apple GPUs retain
        // the embedded IR and compile their pipelines with the native driver.
        run(tool("metal-tt")
            .arg(&script_path)
            .args([
                "-target",
                &native_target,
                "-arch",
                "applegpu_g16g",
                "-no-pack-sources",
                "-compress-resources",
                "-o",
            ])
            .arg(out.join("ocr.binary.metallib")));
        fs::write(out.join("native-target.txt"), native_fingerprint).unwrap();
    }
    fs::write(out.join("metal-compiler.txt"), fingerprint).unwrap();
    // Compress across pipeline variants as well as within Metal resources.
    // This reduces executable IO and first-launch code-signature validation.
    for (name, variable) in [
        ("ocr.metallib", "SIMD_OCR_METAL_IR_BYTES"),
        ("ocr.binary.metallib", "SIMD_OCR_METAL_BINARY_BYTES"),
    ] {
        let input = out.join(name);
        let compressed = out.join(format!("{name}.deflate"));
        let data = fs::read(&input).unwrap();
        println!("cargo:rustc-env={variable}={}", data.len());
        if !compressed.exists()
            || fs::metadata(&input).unwrap().modified().unwrap()
                > fs::metadata(&compressed).unwrap().modified().unwrap()
        {
            fs::write(compressed, miniz_oxide::deflate::compress_to_vec(&data, 6)).unwrap();
        }
    }
}
