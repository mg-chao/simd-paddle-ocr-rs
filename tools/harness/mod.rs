//! Shared process/build boundary for the local CPU validation tools.
#![allow(dead_code)]
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub fn blocked_standard_environment(name: &str) -> bool {
    let key = name.to_ascii_uppercase();
    let dotnet = key
        .strip_prefix("DOTNET_")
        .or_else(|| key.strip_prefix("COMPLUS_"));
    matches!(
        key.as_str(),
        "RUSTFLAGS"
            | "CARGO_ENCODED_RUSTFLAGS"
            | "CARGO_BUILD_RUSTFLAGS"
            | "DOTNET_PROCESSOR_COUNT"
            | "CORECLR_ENABLE_PROFILING"
            | "COR_ENABLE_PROFILING"
    ) || (key.starts_with("CARGO_TARGET_") && key.ends_with("_RUSTFLAGS"))
        || key.starts_with("PPOCR_")
        || key.starts_with("SIMD_OCR_")
        || dotnet.is_some_and(|suffix| {
            suffix.starts_with("ENABLE")
                || suffix.contains("PROFIL")
                || suffix.starts_with("JIT")
                || suffix.starts_with("TIERED")
                || suffix.starts_with("READYTORUN")
                || suffix.starts_with("GC")
                || suffix.contains("VECTORBITWIDTH")
                || suffix.contains("VECTORTBITWIDTH")
        })
}

/// Keep automatic CPU/layout/fusion claims independent of inherited tuning.
pub fn check_standard_environment() -> Result<()> {
    for (name, value) in std::env::vars_os() {
        let key = name.to_string_lossy().to_ascii_uppercase();
        if blocked_standard_environment(&key)
            && (!value.is_empty() || key.starts_with("PPOCR_") || key.starts_with("SIMD_OCR_"))
        {
            return Err(format!(
                "unset {}: CPU validation requires standard build/runtime settings",
                name.to_string_lossy()
            )
            .into());
        }
    }
    Ok(())
}

/// Convert Windows verbatim paths back to paths MSBuild can combine with `/`.
pub fn canonical_path(path: &Path) -> Result<PathBuf> {
    let path = fs::canonicalize(path)?;
    #[cfg(windows)]
    {
        let value = path.to_str().ok_or("path is not UTF-8")?;
        if let Some(rest) = value.strip_prefix(r"\\?\UNC\") {
            return Ok(PathBuf::from(format!(r"\\{rest}")));
        }
        if let Some(rest) = value.strip_prefix(r"\\?\") {
            return Ok(PathBuf::from(rest));
        }
    }
    Ok(path)
}

pub fn command_text(command: &mut Command) -> Result<String> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "{command:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

/// Fingerprint the contents and names of tracked and relevant untracked inputs.
/// Include deleted tracked files so a missing source changes the version too.
pub fn source_tree_hash(root: &Path, paths: &[&str]) -> Result<String> {
    let files = command_text(
        Command::new("git")
            .current_dir(root)
            .args([
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
            ])
            .args(paths),
    )?;
    let mut digest = Sha256::new();
    for path in files
        .split('\0')
        .filter(|s| !s.is_empty())
        .collect::<std::collections::BTreeSet<_>>()
    {
        digest.update(path.as_bytes());
        digest.update([0]);
        if root.join(path).is_file() {
            digest.update(fs::read(root.join(path))?);
        } else {
            digest.update(b"<absent>");
        }
        digest.update([0]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn check_reference(reference: &Path, expected: &str) -> Result<()> {
    let head = command_text(
        Command::new("git")
            .arg("-C")
            .arg(reference)
            .args(["rev-parse", "HEAD"]),
    )?;
    if head != expected {
        return Err(format!("C# HEAD mismatch: expected {expected}, got {head}").into());
    }
    let dirty = command_text(Command::new("git").arg("-C").arg(reference).args([
        "status",
        "--porcelain",
        "--untracked-files=normal",
    ]))?;
    if !dirty.is_empty() {
        return Err(format!("C# reference has modified or untracked files:\n{dirty}").into());
    }
    Ok(())
}

pub fn kill_tree(child: &mut Child) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .creation_flags(0x0800_0000)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

struct Running(Child);
impl Drop for Running {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            kill_tree(&mut self.0);
        }
    }
}

pub fn run_logged(command: &mut Command, log: &Path, timeout: Duration) -> Result<()> {
    if timeout.is_zero() {
        return Err("process deadline exceeded before launch".into());
    }
    let file = fs::File::create(log)?;
    let mut child = Running(command.stdout(file.try_clone()?).stderr(file).spawn()?);
    let start = Instant::now();
    loop {
        if let Some(status) = child.0.try_wait()? {
            return if status.success() {
                Ok(())
            } else {
                Err(format!("process exited with {status}; see {}", log.display()).into())
            };
        }
        if start.elapsed() >= timeout {
            return Err(format!("process deadline exceeded; see {}", log.display()).into());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// All adapter AND ProjectReference bin/obj paths belong to this invocation.
/// A fixed pivot avoids relying on SDK-dependent framework directory names.
pub fn build_csharp(
    root: &Path,
    reference: &Path,
    project: &Path,
    run_dir: &Path,
    timeout: Duration,
) -> Result<PathBuf> {
    let project = canonical_path(&if project.is_absolute() {
        project.to_owned()
    } else {
        root.join(project)
    })?;
    let reference = canonical_path(reference)?;
    fs::create_dir_all(run_dir)?;
    let run_dir = canonical_path(run_dir)?;
    let artifacts = run_dir.join("csharp-artifacts");
    let name = project
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("invalid C# project name")?;
    let mut command = Command::new("dotnet");
    command
        .current_dir(project.parent().ok_or("project has no parent")?)
        .arg("build")
        .arg(&project)
        .args(["-c", "Release", "--nologo", "--artifacts-path"])
        .arg(&artifacts)
        .arg("-p:ArtifactsPivots=cpu")
        .arg(format!("-p:SimdPaddleOcrRoot={}", reference.display()));
    run_logged(&mut command, &run_dir.join("csharp-build.log"), timeout)?;
    let dll = artifacts
        .join("bin")
        .join(name)
        .join("cpu")
        .join(format!("{name}.dll"));
    if !dll.is_file() {
        return Err(format!("C# build did not produce {}", dll.display()).into());
    }
    Ok(dll)
}
