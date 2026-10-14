use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{Tolerance, compare_bundles};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct Fixtures {
    root: PathBuf,
    reference: PathBuf,
    candidate: PathBuf,
}

impl Fixtures {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "simd-paddle-ocr-parity-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let reference = root.join("reference");
        let candidate = root.join("candidate");
        for directory in [&reference, &candidate] {
            fs::create_dir(directory).unwrap();
            let model = b"frozen model";
            fs::write(directory.join("model.onnx"), model).unwrap();
            let input = b"\x01\x02\x03";
            let tensor: Vec<u8> = [0.0f32, 1.0, 2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
            let result = br#"{"lines":[{"text":"hello","score":0.9}],"count":1}"#;
            let artifacts = [
                artifact(
                    directory,
                    "input",
                    "input.u8",
                    "u8",
                    json!([3]),
                    "HWC",
                    input,
                ),
                artifact(
                    directory,
                    "node/conv",
                    "conv.f32",
                    "f32",
                    json!([1, 3]),
                    "NC",
                    &tensor,
                ),
                artifact(
                    directory,
                    "results",
                    "result.json",
                    "json",
                    json!([]),
                    "json",
                    result,
                ),
            ];
            let manifest = json!({
                "schema_version":1, "contract":{"threads":1, "confidence":"compact-direct"},
                "models":[{"id":"rec", "path":"model.onnx", "sha256":hash(model), "bytes":model.len()}],
                "cases":[{"id":"line", "kind":"recognition", "artifacts":artifacts}],
                "provenance":{"commit":"reference"}
            });
            write_manifest(directory, &manifest);
        }
        Self {
            root,
            reference,
            candidate,
        }
    }

    fn edit(&self, edit: impl FnOnce(&mut Value)) {
        let mut manifest = manifest(&self.candidate);
        edit(&mut manifest);
        write_manifest(&self.candidate, &manifest);
    }

    fn replace_artifact(&self, index: usize, bytes: &[u8]) {
        self.edit(|manifest| {
            let artifact = &mut manifest["cases"][0]["artifacts"][index];
            fs::write(
                self.candidate.join(artifact["path"].as_str().unwrap()),
                bytes,
            )
            .unwrap();
            artifact["sha256"] = json!(hash(bytes));
            artifact["bytes"] = json!(bytes.len());
        });
    }

    fn compare(
        &self,
    ) -> Result<simd_paddle_ocr_rs::ComparisonReport, simd_paddle_ocr_rs::BundleError> {
        compare_bundles(&self.reference, &self.candidate, Tolerance::default())
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_simd-paddle-ocr-rs"));
        command
            .arg("compare")
            .arg(&self.reference)
            .arg(&self.candidate);
        command
    }
}

impl Drop for Fixtures {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}

fn artifact(
    directory: &Path,
    id: &str,
    path: &str,
    dtype: &str,
    shape: Value,
    layout: &str,
    bytes: &[u8],
) -> Value {
    fs::write(directory.join(path), bytes).unwrap();
    json!({"id":id, "path":path, "dtype":dtype, "shape":shape, "layout":layout, "sha256":hash(bytes), "bytes":bytes.len()})
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn manifest(directory: &Path) -> Value {
    serde_json::from_slice(&fs::read(directory.join("manifest.json")).unwrap()).unwrap()
}
fn write_manifest(directory: &Path, value: &Value) {
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(value).unwrap(),
    )
    .unwrap();
}

#[test]
fn identical_bundles_match_and_ignore_provenance() {
    let f = Fixtures::new();
    f.edit(|m| {
        m["provenance"] = json!({"commit":"rust"});
        m["performance"] = json!({"latency":999});
    });
    let report = f.compare().unwrap();
    assert!(report.matches());
    assert_eq!((report.checked_cases, report.checked_artifacts), (1, 3));
    assert_eq!(f.command().output().unwrap().status.code(), Some(0));
}

#[test]
fn float_comparison_reports_first_stage_and_all_mismatch_metrics() {
    let f = Fixtures::new();
    let bytes: Vec<u8> = [0.0f32, 1.01, 3.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    f.replace_artifact(1, &bytes);
    let difference = f.compare().unwrap().first_difference.unwrap();
    assert!(
        difference.contains("case line, artifact node/conv"),
        "{difference}"
    );
    assert!(
        difference.contains("count=2/3, first index=1"),
        "{difference}"
    );
    assert!(difference.contains("max abs=") && difference.contains("max rel="));
    assert_eq!(f.command().output().unwrap().status.code(), Some(1));
}

#[test]
fn little_endian_floats_use_reference_relative_tolerance() {
    let f = Fixtures::new();
    let bytes: Vec<u8> = [0.000001f32, 1.00005, 2.0001]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    f.replace_artifact(1, &bytes);
    assert!(f.compare().unwrap().matches());
    let strict = compare_bundles(
        &f.reference,
        &f.candidate,
        Tolerance {
            atol: 0.0,
            rtol: 0.0,
        },
    )
    .unwrap();
    assert!(!strict.matches());
    assert_eq!(
        f.command()
            .args(["--atol", "0", "--rtol", "0"])
            .output()
            .unwrap()
            .status
            .code(),
        Some(1)
    );
}

#[test]
fn bytes_compare_exactly_even_with_large_numeric_tolerance() {
    let f = Fixtures::new();
    f.replace_artifact(0, b"\x01\x09\x03");
    let report = compare_bundles(
        &f.reference,
        &f.candidate,
        Tolerance {
            atol: 100.0,
            rtol: 100.0,
        },
    )
    .unwrap();
    let difference = report.first_difference.unwrap();
    assert!(difference.contains("artifact input: u8 mismatch count=1/3, first index=1"));
}

#[test]
fn json_floats_tolerate_but_text_and_integer_structure_are_exact() {
    let f = Fixtures::new();
    f.replace_artifact(
        2,
        br#"{"count":1,"lines":[{"score":0.900001,"text":"hello"}]}"#,
    );
    assert!(f.compare().unwrap().matches());
    f.replace_artifact(2, br#"{"count":2,"lines":[{"score":0.9,"text":"hello"}]}"#);
    let difference = compare_bundles(
        &f.reference,
        &f.candidate,
        Tolerance {
            atol: 10.0,
            rtol: 10.0,
        },
    )
    .unwrap()
    .first_difference
    .unwrap();
    assert!(difference.contains("integer differs"));
    f.replace_artifact(
        2,
        br#"{"count":1,"lines":[{"score":0.9,"text":"changed"}]}"#,
    );
    assert!(
        f.compare()
            .unwrap()
            .first_difference
            .unwrap()
            .contains("$[\"lines\"][0][\"text\"]")
    );
}

#[test]
fn all_files_are_validated_before_a_parity_difference_is_returned() {
    let f = Fixtures::new();
    f.replace_artifact(0, b"\x09\x02\x03");
    f.replace_artifact(2, b"invalid JSON");
    assert!(
        f.compare()
            .unwrap_err()
            .to_string()
            .contains("invalid JSON")
    );
    assert_eq!(f.command().output().unwrap().status.code(), Some(2));
}

#[test]
fn corrupted_artifact_or_model_and_nonfinite_tensor_are_invalid() {
    for path in ["input.u8", "conv.f32", "result.json", "model.onnx"] {
        let f = Fixtures::new();
        let mut bytes = fs::read(f.candidate.join(path)).unwrap();
        bytes[0] ^= 1;
        fs::write(f.candidate.join(path), bytes).unwrap();
        assert!(
            f.compare()
                .unwrap_err()
                .to_string()
                .contains("sha256 mismatch")
        );
    }
    let f = Fixtures::new();
    let bytes: Vec<u8> = [0.0f32, f32::NAN, 2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    f.replace_artifact(1, &bytes);
    assert!(
        f.compare()
            .unwrap_err()
            .to_string()
            .contains("nonfinite f32 at index 1")
    );
}

#[test]
fn malformed_lengths_shapes_and_json_metadata_are_invalid() {
    let f = Fixtures::new();
    fs::write(f.candidate.join("input.u8"), [1, 2]).unwrap();
    assert!(f.compare().unwrap_err().to_string().contains("byte length"));
    let f = Fixtures::new();
    f.edit(|m| m["cases"][0]["artifacts"][1]["shape"] = json!([1, 4]));
    assert!(
        f.compare()
            .unwrap_err()
            .to_string()
            .contains("shape implies")
    );
    let f = Fixtures::new();
    f.edit(|m| m["cases"][0]["artifacts"][1]["shape"] = json!([usize::MAX, 4]));
    assert!(f.compare().unwrap_err().to_string().contains("overflow"));
    let f = Fixtures::new();
    f.edit(|m| m["cases"][0]["artifacts"][2]["shape"] = json!([1]));
    assert!(
        f.compare()
            .unwrap_err()
            .to_string()
            .contains("JSON artifacts require")
    );
}

#[test]
fn duplicates_and_unsupported_schema_are_invalid() {
    for collection in ["models", "cases", "artifacts"] {
        let f = Fixtures::new();
        f.edit(|m| {
            let entries = if collection == "artifacts" {
                &mut m["cases"][0]["artifacts"]
            } else {
                &mut m[collection]
            };
            let duplicate = entries[0].clone();
            entries.as_array_mut().unwrap().push(duplicate);
        });
        assert!(f.compare().unwrap_err().to_string().contains("duplicate"));
    }
    let f = Fixtures::new();
    f.edit(|m| m["schema_version"] = json!(2));
    assert!(
        f.compare()
            .unwrap_err()
            .to_string()
            .contains("unsupported schema")
    );
}

#[test]
fn contract_model_hashes_artifact_metadata_and_order_are_parity_differences() {
    let f = Fixtures::new();
    f.edit(|m| m["contract"]["threads"] = json!(2));
    assert!(
        f.compare()
            .unwrap()
            .first_difference
            .unwrap()
            .contains("contract")
    );
    let f = Fixtures::new();
    fs::write(f.candidate.join("model.onnx"), b"other model").unwrap();
    f.edit(|m| {
        m["models"][0]["sha256"] = json!(hash(b"other model"));
        m["models"][0]["bytes"] = json!(11);
    });
    assert!(
        f.compare()
            .unwrap()
            .first_difference
            .unwrap()
            .contains("model rec sha256")
    );
    let f = Fixtures::new();
    f.edit(|m| m["cases"][0]["artifacts"][1]["shape"] = json!([3, 1]));
    assert!(
        f.compare()
            .unwrap()
            .first_difference
            .unwrap()
            .contains("metadata differs")
    );
    let f = Fixtures::new();
    f.edit(|m| {
        m["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .swap(0, 1)
    });
    assert!(
        f.compare()
            .unwrap()
            .first_difference
            .unwrap()
            .contains("stage order/ID differs")
    );
}

#[test]
fn missing_and_extra_cases_and_artifacts_are_parity_differences() {
    let f = Fixtures::new();
    f.edit(|m| {
        m["cases"].as_array_mut().unwrap().clear();
    });
    assert!(
        f.compare()
            .unwrap()
            .first_difference
            .unwrap()
            .contains("case line: missing")
    );
    let f = Fixtures::new();
    f.edit(|m| {
        let mut extra = m["cases"][0].clone();
        extra["id"] = json!("extra");
        m["cases"].as_array_mut().unwrap().push(extra);
    });
    assert!(
        f.compare()
            .unwrap()
            .first_difference
            .unwrap()
            .contains("case extra: extra")
    );
    let f = Fixtures::new();
    f.edit(|m| {
        m["cases"][0]["artifacts"].as_array_mut().unwrap().pop();
    });
    assert!(
        f.compare()
            .unwrap()
            .first_difference
            .unwrap()
            .contains("artifact results: missing")
    );
    let f = Fixtures::new();
    f.edit(|m| {
        let mut extra = m["cases"][0]["artifacts"][0].clone();
        extra["id"] = json!("extra");
        m["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .push(extra);
    });
    assert!(
        f.compare()
            .unwrap()
            .first_difference
            .unwrap()
            .contains("artifact extra: extra")
    );
}

#[test]
fn absolute_paths_traversal_and_windows_drive_paths_are_rejected() {
    for path in [
        "../input.u8",
        "..\\input.u8",
        "/tmp/input.u8",
        "C:\\input.u8",
        "\\\\host\\share",
        "./input.u8",
        "a//input.u8",
    ] {
        let f = Fixtures::new();
        f.edit(|m| m["cases"][0]["artifacts"][0]["path"] = json!(path));
        assert!(
            f.compare()
                .unwrap_err()
                .to_string()
                .contains("unsafe bundle path"),
            "{path}"
        );
    }
}

#[test]
fn nonportable_windows_names_are_explicitly_rejected_before_file_lookup() {
    let f = Fixtures::new();
    for path in [
        "CON",
        "prn.u8",
        "Aux/input.u8",
        "nul.any.extension",
        "COM1.u8",
        "com9.data",
        "LPT1.u8",
        "lpt9/data.u8",
        "trailing./input.u8",
        "input.u8 ",
        "input<u8",
        "input>u8",
        "input:u8",
        "input\"u8",
        "input|u8",
        "input?u8",
        "input*u8",
        "input\u{0}u8",
        "input\u{1}u8",
        "input\u{1f}u8",
    ] {
        f.edit(|m| m["cases"][0]["artifacts"][0]["path"] = json!(path));
        assert!(
            f.compare()
                .unwrap_err()
                .to_string()
                .contains("unsafe bundle path"),
            "{path:?}"
        );
    }
}

#[test]
fn similar_nonreserved_portable_names_remain_valid() {
    let f = Fixtures::new();
    for path in [
        "conifer.u8",
        "COM0.u8",
        "COM10.u8",
        "LPT0.u8",
        "nulled.u8",
        ".hidden.u8",
    ] {
        fs::write(f.candidate.join(path), [1, 2, 3]).unwrap();
        f.edit(|m| m["cases"][0]["artifacts"][0]["path"] = json!(path));
        assert!(f.compare().unwrap().matches(), "{path:?}");
    }
}

#[test]
fn symlinks_outside_bundle_are_rejected() {
    let f = Fixtures::new();
    let outside = f.root.join("outside.u8");
    fs::write(&outside, [1, 2, 3]).unwrap();
    fs::remove_file(f.candidate.join("input.u8")).unwrap();
    #[cfg(unix)]
    let result = std::os::unix::fs::symlink(&outside, f.candidate.join("input.u8"));
    #[cfg(windows)]
    let result = std::os::windows::fs::symlink_file(&outside, f.candidate.join("input.u8"));
    match result {
        Ok(()) => assert!(
            f.compare()
                .unwrap_err()
                .to_string()
                .contains("outside its bundle")
        ),
        Err(error)
            if error.kind() == std::io::ErrorKind::PermissionDenied
                || error.raw_os_error() == Some(1314) =>
        {
            eprintln!("Symlink test skipped: symlink privilege unavailable")
        }
        Err(error) => panic!("creating test symlink: {error}"),
    }
}

#[cfg(windows)]
#[test]
fn windows_junctions_outside_bundle_are_rejected_without_symlink_privilege() {
    let f = Fixtures::new();
    let outside = f.root.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("input.u8"), [1, 2, 3]).unwrap();
    let status = Command::new("powershell")
        .args([
            "-NoProfile", "-NonInteractive", "-Command",
            "New-Item -ItemType Junction -Path $env:PARITY_JUNCTION -Target $env:PARITY_TARGET -ErrorAction Stop | Out-Null",
        ])
        .env("PARITY_JUNCTION", f.candidate.join("external"))
        .env("PARITY_TARGET", &outside)
        .status()
        .unwrap();
    assert!(status.success(), "creating test directory junction");
    f.edit(|m| m["cases"][0]["artifacts"][0]["path"] = json!("external/input.u8"));
    assert!(
        f.compare()
            .unwrap_err()
            .to_string()
            .contains("outside its bundle")
    );
}

#[test]
fn cli_help_and_invalid_arguments_have_correct_exit_statuses() {
    let binary = env!("CARGO_BIN_EXE_simd-paddle-ocr-rs");
    let help = Command::new(binary).arg("--help").output().unwrap();
    assert_eq!(help.status.code(), Some(0));
    assert!(
        String::from_utf8(help.stdout)
            .unwrap()
            .contains("Exit status: 0 match")
    );
    let f = Fixtures::new();
    for args in [
        ["--atol", "NaN"],
        ["--rtol", "-1"],
        ["--atol", "inf"],
        ["--unknown", "1"],
    ] {
        assert_eq!(
            f.command().args(args).output().unwrap().status.code(),
            Some(2)
        );
    }
    assert_eq!(
        Command::new(binary)
            .arg("compare")
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
}
