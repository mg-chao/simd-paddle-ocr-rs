//! The archive launcher uses PowerShell 7/.NET on Windows. Other platforms run
//! the cross-platform bundle/import tests without requiring this local tool.
#![cfg(windows)]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{Tolerance, bundle::BundleWriter, compare_bundles, validate_bundle};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct TempRoot(PathBuf);
impl TempRoot {
    fn new() -> Self {
        // Spaces exercise the script's structured path arguments.
        let path = std::env::temp_dir().join(format!(
            "simd ocr archive {} {}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, relative: &str) -> PathBuf {
        self.0.join(relative)
    }
}
impl Drop for TempRoot {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn powershell() -> Option<&'static str> {
    let available = Command::new("pwsh.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "if ($PSVersionTable.PSVersion.Major -lt 7) { exit 1 }",
        ])
        .output()
        .is_ok_and(|output| output.status.success());
    if available {
        Some("pwsh.exe")
    } else {
        eprintln!("Archive integration tests skipped: PowerShell 7 is unavailable");
        None
    }
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fixture(temp: &TempRoot) -> PathBuf {
    let model = temp.path("source model.onnx");
    fs::write(&model, b"tiny frozen model").unwrap();
    let bundle = temp.path("source bundle");
    let mut writer =
        BundleWriter::new(&bundle, json!({"profile":"canonical", "fixture":true})).unwrap();
    writer.copy_model("cls", &model, "models/cls.onnx").unwrap();
    // Deliberately write files out of ordinal order.
    writer
        .write_auxiliary("zzz.bin", b"last", "fixture")
        .unwrap();
    let result = writer
        .write_json(
            "result",
            "cases/sample/result.json",
            &json!({"label":0, "score":0.75}),
        )
        .unwrap();
    let probabilities = writer
        .write_floats(
            "probabilities",
            "cases/sample/probabilities.f32",
            &[0.75, 0.25],
            &[1, 2],
            "row_major",
        )
        .unwrap();
    let input = writer
        .write_bytes(
            "input",
            "cases/sample/input.u8",
            &[1, 2, 3],
            &[1, 1, 3],
            "bgr24",
        )
        .unwrap();
    writer
        .write_auxiliary("aaa.txt", b"first", "fixture")
        .unwrap();
    writer
        .add_case("sample", "classifier", vec![input, probabilities, result])
        .unwrap();
    writer
        .finish(json!({"provenance":{
            "source_inputs":{"tree_sha256":hash(b"fixture source tree")},
            "git":{"head":"fixture-reference-head"}
        }}))
        .unwrap()
}

fn archive(shell: &str, bundle: &Path, id: &str, archives: &Path, catalog: &Path) -> Output {
    Command::new(shell)
        .args(["-NoProfile", "-NonInteractive", "-File"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/archive-reference.ps1"))
        .arg("-Bundle")
        .arg(bundle)
        .arg("-BaselineId")
        .arg(id)
        .arg("-Binary")
        .arg(env!("CARGO_BIN_EXE_simd-paddle-ocr-rs"))
        .arg("-ArchiveDirectory")
        .arg(archives)
        .arg("-Catalog")
        .arg(catalog)
        .output()
        .unwrap()
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn zip_entries(shell: &str, path: &Path) -> Vec<Value> {
    let output = Command::new(shell)
        .args(["-NoProfile", "-NonInteractive", "-Command", r#"
$taskArchiveZip = [IO.Compression.ZipFile]::OpenRead($env:ARCHIVE_TEST_ZIP)
try {
    $taskArchiveEntries = @($taskArchiveZip.Entries | ForEach-Object {
        [pscustomobject]@{name = $_.FullName; timestamp = $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')}
    })
    ConvertTo-Json -InputObject $taskArchiveEntries -Depth 5 -Compress
} finally { $taskArchiveZip.Dispose() }
"#])
        .env("ARCHIVE_TEST_ZIP", path)
        .output().unwrap();
    assert_success(&output);
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn deterministic_archives_restore_validate_and_refuse_existing_ids() {
    let Some(shell) = powershell() else {
        return;
    };
    let temp = TempRoot::new();
    let bundle = fixture(&temp);
    let archives = temp.path("archives");
    let catalog = temp.path("catalog/baselines.json");
    assert_success(&archive(shell, &bundle, "fixture-a", &archives, &catalog));
    assert_success(&archive(shell, &bundle, "fixture-b", &archives, &catalog));

    let first_zip = fs::read(archives.join("fixture-a.zip")).unwrap();
    assert_eq!(first_zip, fs::read(archives.join("fixture-b.zip")).unwrap());
    let entries = zip_entries(shell, &archives.join("fixture-a.zip"));
    let names = entries
        .iter()
        .map(|entry| entry["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            "aaa.txt",
            "cases/sample/input.u8",
            "cases/sample/probabilities.f32",
            "cases/sample/result.json",
            "manifest.json",
            "models/cls.onnx",
            "zzz.bin"
        ]
    );
    assert!(
        entries
            .iter()
            .all(|entry| entry["timestamp"] == "1980-01-01 00:00:00")
    );
    let restored = archives.join("verification/fixture-a");
    assert_eq!(validate_bundle(&restored).unwrap().checked_artifacts, 3);
    assert!(
        compare_bundles(
            &bundle,
            &restored,
            Tolerance {
                atol: 0.0,
                rtol: 0.0
            }
        )
        .unwrap()
        .matches()
    );
    let cli_validation = Command::new(env!("CARGO_BIN_EXE_simd-paddle-ocr-rs"))
        .arg("validate")
        .arg(&restored)
        .output()
        .unwrap();
    assert_success(&cli_validation);

    let archive_hash = hash(&first_zip);
    let sidecar = fs::read_to_string(archives.join("fixture-a.zip.sha256")).unwrap();
    assert_eq!(sidecar.trim(), format!("{archive_hash}  fixture-a.zip"));
    let original_catalog = fs::read(&catalog).unwrap();
    let rows: Vec<Value> = serde_json::from_slice(&original_catalog).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["id"], "fixture-a");
    assert_eq!(rows[0]["archive_sha256"], archive_hash);
    assert_eq!(rows[0]["archive_bytes"], first_zip.len() as u64);
    assert_eq!(
        rows[0]["manifest_sha256"],
        hash(&fs::read(bundle.join("manifest.json")).unwrap())
    );
    assert_eq!(rows[0]["source_tree_sha256"], hash(b"fixture source tree"));
    assert_eq!(rows[0]["assets"][0]["sha256"], hash(b"tiny frozen model"));

    let duplicate = archive(shell, &bundle, "fixture-a", &archives, &catalog);
    assert!(!duplicate.status.success());
    assert_eq!(fs::read(archives.join("fixture-a.zip")).unwrap(), first_zip);
    assert_eq!(fs::read(&catalog).unwrap(), original_catalog);
    // A catalog ID is immutable even if the archive output directory changes.
    let other_archives = temp.path("duplicate archive destination");
    assert!(
        !archive(shell, &bundle, "fixture-a", &other_archives, &catalog)
            .status
            .success()
    );
    assert!(!other_archives.exists());
    assert!(!other_archives.join("fixture-a.zip").exists());
    assert!(!other_archives.join("verification/fixture-a").exists());
    assert_eq!(fs::read(&catalog).unwrap(), original_catalog);
}

#[test]
fn source_output_overlap_and_malformed_import_references_create_no_archive() {
    let Some(shell) = powershell() else {
        return;
    };
    let temp = TempRoot::new();
    let bundle = fixture(&temp);
    let catalog = temp.path("rejected-catalog.json");
    let same_directory = archive(shell, &bundle, "same-directory", &bundle, &catalog);
    assert!(!same_directory.status.success());
    assert!(String::from_utf8_lossy(&same_directory.stderr).contains("outside the source bundle"));
    assert!(!bundle.join("same-directory.zip").exists());
    assert!(!bundle.join("verification").exists());
    let nested = bundle.join("nested archives");
    let nested_directory = archive(shell, &bundle, "nested-directory", &nested, &catalog);
    assert!(!nested_directory.status.success());
    assert!(
        String::from_utf8_lossy(&nested_directory.stderr).contains("outside the source bundle")
    );
    assert!(!nested.exists());
    assert!(!catalog.exists());

    let malformed = temp.path("malformed import");
    fs::create_dir(&malformed).unwrap();
    let invalid_manifest = json!({
        "schema_version":1, "kind":"model_import_contracts", "models":[{
            "id":"cls", "path":"../source model.onnx", "bytes":17, "sha256":hash(b"tiny frozen model")
        }], "dictionaries":[], "contracts":[], "auxiliary_files":[]
    });
    fs::write(
        malformed.join("manifest.json"),
        serde_json::to_vec_pretty(&invalid_manifest).unwrap(),
    )
    .unwrap();
    let validation = Command::new(env!("CARGO_BIN_EXE_simd-paddle-ocr-rs"))
        .arg("validate-import")
        .arg(&malformed)
        .output()
        .unwrap();
    assert_eq!(validation.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&validation.stderr).contains("unsafe bundle path"));
    let archives = temp.path("invalid archives");
    let invalid_archive = archive(shell, &malformed, "invalid-import", &archives, &catalog);
    assert!(!invalid_archive.status.success());
    assert!(String::from_utf8_lossy(&invalid_archive.stderr).contains("unsafe bundle path"));
    assert!(!archives.exists());
    assert!(!catalog.exists());
}
