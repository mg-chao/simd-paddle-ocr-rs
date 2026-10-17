#[path = "../tools/harness/mod.rs"]
mod harness;
use std::{fs, path::PathBuf, process::Command, time::Duration};

#[test]
fn every_adapter_pins_the_same_reference_sdk() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for adapter in [
        "csharp-reference",
        "csharp-qualification",
        "csharp-benchmark",
        "csharp-profile",
    ] {
        let value: serde_json::Value = serde_json::from_slice(
            &fs::read(root.join("tools").join(adapter).join("global.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(value["sdk"]["version"], "10.0.303", "{adapter}");
        assert_eq!(value["sdk"]["rollForward"], "disable", "{adapter}");
    }
}

#[test]
fn inherited_tuning_cannot_change_declared_default_execution() {
    for key in [
        "PPOCR_NHWC",
        "SIMD_OCR_NOFUSE",
        "PPOCR_DUMP_GRAPH",
        "DOTNET_EnableAVX512F",
        "COMPlus_EnableFMA",
        "complus_enablehwintrinsic",
        "DOTNET_PROCESSOR_COUNT",
        "DOTNET_PreferredVectorBitWidth",
        "COMPlus_MaxVectorTBitWidth",
        "CORECLR_ENABLE_PROFILING",
        "COR_ENABLE_PROFILING",
        "DOTNET_gcServer",
        "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS",
    ] {
        assert!(harness::blocked_standard_environment(key), "{key}");
    }
    assert!(!harness::blocked_standard_environment("DOTNET_ROOT"));
    assert!(!harness::blocked_standard_environment("PATH"));
}

#[test]
fn source_fingerprint_distinguishes_edits_with_the_same_dirty_file_list() -> harness::Result<()> {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
        "target/harness-source-{nonce}-{}",
        std::process::id()
    ));
    fs::create_dir_all(directory.join("src"))?;
    fs::create_dir_all(directory.join("docs"))?;
    harness::command_text(
        Command::new("git")
            .args(["init", "--quiet"])
            .arg(&directory),
    )?;
    let source = directory.join("src/request.rs");
    fs::write(&source, "original")?;
    harness::command_text(
        Command::new("git")
            .current_dir(&directory)
            .args(["add", "src/request.rs"]),
    )?;

    fs::write(&source, "first dirty version")?;
    let status_before = harness::command_text(
        Command::new("git")
            .current_dir(&directory)
            .args(["status", "--porcelain"]),
    )?;
    let before = harness::source_tree_hash(&directory, &["src"])?;
    fs::write(&source, "second dirty version")?;
    let status_after = harness::command_text(
        Command::new("git")
            .current_dir(&directory)
            .args(["status", "--porcelain"]),
    )?;
    let after = harness::source_tree_hash(&directory, &["src"])?;
    assert_eq!(status_before, status_after);
    assert_ne!(
        before, after,
        "dirty filenames alone do not identify source contents"
    );

    fs::write(
        directory.join("docs/note.md"),
        "outside the requested source scope",
    )?;
    assert_eq!(after, harness::source_tree_hash(&directory, &["src"])?);
    let extra = directory.join("src/new-request.rs");
    fs::write(&extra, "untracked source")?;
    assert_ne!(after, harness::source_tree_hash(&directory, &["src"])?);
    fs::remove_file(extra)?;
    assert_eq!(after, harness::source_tree_hash(&directory, &["src"])?);
    fs::remove_file(&source)?;
    assert_ne!(after, harness::source_tree_hash(&directory, &["src"])?);
    fs::write(source, "second dirty version")?;
    assert_eq!(after, harness::source_tree_hash(&directory, &["src"])?);
    Ok(())
}

#[test]
fn process_failures_and_deadlines_remain_errors() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
        "target/harness-contract-{nonce}-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    let mut command = Command::new("never-launch-this-command");
    assert!(
        harness::run_logged(&mut command, &root.join("zero.log"), Duration::ZERO)
            .unwrap_err()
            .to_string()
            .contains("before launch")
    );
    #[cfg(windows)]
    {
        let mut failed = Command::new("powershell");
        failed.args(["-NoProfile", "-NonInteractive", "-Command", "exit 7"]);
        assert!(
            harness::run_logged(
                &mut failed,
                &root.join("failure.log"),
                Duration::from_secs(15)
            )
            .unwrap_err()
            .to_string()
            .contains("exited")
        );
        let pid_file = root.join("timeout.pid");
        let escaped = pid_file.to_string_lossy().replace('\'', "''");
        let mut timed = Command::new("powershell");
        timed.args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &format!("$PID | Set-Content -LiteralPath '{escaped}'; Start-Sleep -Seconds 30"),
        ]);
        assert!(
            harness::run_logged(
                &mut timed,
                &root.join("timeout.log"),
                Duration::from_secs(5)
            )
            .unwrap_err()
            .to_string()
            .contains("deadline")
        );
        let pid = fs::read_to_string(pid_file).unwrap();
        let output = Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                &format!(
                    "if (Get-Process -Id {} -ErrorAction SilentlyContinue) {{ exit 1 }}",
                    pid.trim()
                ),
            ])
            .status()
            .unwrap();
        assert!(output.success(), "timed-out child must be reaped");
    }
}

/// Explicit physical-machine smoke check; needs the pinned sibling checkout/models/.NET.
#[test]
#[ignore = "local .NET/model integration; run explicitly"]
fn isolated_builds_and_live_workers_can_overlap() -> harness::Result<()> {
    use std::{
        io::{BufRead, BufReader, Write},
        process::{Child, Stdio},
        sync::mpsc,
        time::SystemTime,
    };
    struct Worker(Child);
    impl Drop for Worker {
        fn drop(&mut self) {
            harness::kill_tree(&mut self.0);
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let reference = harness::canonical_path(&root.join("../SimdPaddleOCR"))?;
    let nonce = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_millis();
    let run = root.join(format!(
        "target/harness-overlap-{nonce}-{}",
        std::process::id()
    ));
    fs::create_dir_all(&run)?;
    let project = root.join("tools/csharp-reference/Reference.csproj");
    let (a, b) = std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            harness::build_csharp(
                &root,
                &reference,
                &project,
                &run.join("a"),
                Duration::from_secs(180),
            )
        });
        let b = scope.spawn(|| {
            harness::build_csharp(
                &root,
                &reference,
                &project,
                &run.join("b"),
                Duration::from_secs(180),
            )
        });
        (a.join().unwrap(), b.join().unwrap())
    });
    let (a, b) = (a?, b?);
    assert_ne!(a, b);
    let pixels = run.join("blank.bgr");
    fs::write(&pixels, vec![255; 512 * 512 * 3])?;
    let request =
        serde_json::json!({"id":"blank","path":pixels,"width":512,"height":512,"stride":1536});
    let mut workers = Vec::new();
    let (sender, responses) = mpsc::channel();
    for dll in [a, b] {
        let mut child = Command::new("dotnet")
            .arg(dll)
            .arg(root.join("tests/baseline/manifest.json"))
            .arg(reference.join("models"))
            .arg("tiny")
            .env("DOTNET_PROCESSOR_COUNT", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let mut input = child.stdin.take().ok_or("worker input missing")?;
        writeln!(input, "{request}")?;
        input.flush()?;
        let stdout = child.stdout.take().ok_or("worker output missing")?;
        let sender = sender.clone();
        std::thread::spawn(move || {
            let _ = sender.send(BufReader::new(stdout).lines().next());
        });
        // Keep both stdin handles open, so both reference DLLs stay loaded.
        workers.push((Worker(child), input));
    }
    for _ in 0..2 {
        let response = responses
            .recv_timeout(Duration::from_secs(120))?
            .ok_or("worker exited before replying")??;
        let result: serde_json::Value = serde_json::from_str(&response)?;
        assert_eq!(result["id"], "blank");
        assert_eq!(result["detected_count"], 0);
    }
    drop(workers);
    println!(
        "Overlapping builds/workers passed; artifacts: {}",
        run.display()
    );
    Ok(())
}
