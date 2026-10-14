use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{
    ComparisonScope, MAX_JSON_BYTES, STREAM_BLOCK_BYTES, Tolerance,
    bundle::{BundleWriter, Dtype},
    compare_bundles, compare_stage, validate_bundle,
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct TempRoot(PathBuf);
impl TempRoot {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "simd-ocr-harness-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for TempRoot {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn contract() -> Value {
    json!({"profile":"canonical", "nodes":"checkpoints"})
}
fn manifest(directory: &Path) -> Value {
    serde_json::from_slice(&fs::read(directory.join("manifest.json")).unwrap()).unwrap()
}
fn edit_manifest(directory: &Path, edit: impl FnOnce(&mut Value)) {
    let mut value = manifest(directory);
    edit(&mut value);
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&value).unwrap(),
    )
    .unwrap();
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_fixture(temp: &TempRoot, name: &str, only_stage: bool) -> PathBuf {
    let source = temp.path("frozen.onnx");
    fs::write(&source, b"frozen fixture model").unwrap();
    let path = temp.path(name);
    let mut writer = BundleWriter::new(&path, contract()).unwrap();
    writer
        .copy_model("cls", &source, "models/cls.onnx")
        .unwrap();
    writer
        .write_auxiliary("graph-inventory.json", b"[]\n", "graph_inventory")
        .unwrap();
    let selected = writer
        .write_floats(
            "cls.node.0000",
            "cases/sample/conv.f32",
            &[0.0, 1.0, 2.0],
            &[1, 3],
            "row_major",
        )
        .unwrap();
    let mut artifacts = Vec::new();
    if !only_stage {
        artifacts.push(
            writer
                .write_bytes("input", "cases/sample/input.u8", &[1, 2, 3], &[3], "bgr24")
                .unwrap(),
        );
    }
    artifacts.push(selected);
    if !only_stage {
        artifacts.push(
            writer
                .write_json(
                    "result",
                    "cases/sample/result.json",
                    &json!({"text":"hello", "count":1, "score":0.9}),
                )
                .unwrap(),
        );
    }
    writer.add_case("sample", "classifier", artifacts).unwrap();
    writer
        .finish(json!({"provenance":{"fixture":true}}))
        .unwrap()
}

#[test]
fn writer_round_trips_scalar_empty_float_pixel_json_and_auxiliary_files() {
    let temp = TempRoot::new();
    let model = temp.path("source.onnx");
    fs::write(&model, b"model bytes").unwrap();
    let directory = temp.path("round-trip");
    let mut writer = BundleWriter::new(&directory, contract()).unwrap();
    writer.copy_model("cls", &model, "models/cls.onnx").unwrap();
    let scalar = writer
        .write_floats("scalar", "scalar.f32", &[1.25], &[], "row_major")
        .unwrap();
    let empty = writer
        .write_floats("empty", "empty.f32", &[], &[1, 0, 3], "nchw")
        .unwrap();
    let values = writer
        .write_floats("values", "values.f32", &[-1.0, 0.0, 1.0], &[3], "row_major")
        .unwrap();
    let pixels = writer
        .write_bytes("pixels", "pixels.u8", &[1, 2, 3], &[1, 1, 3], "bgr24")
        .unwrap();
    let public = writer
        .write_json(
            "public",
            "public.json",
            &json!({"count":1, "score":0.99, "text":"你好"}),
        )
        .unwrap();
    assert_eq!(scalar.dtype, Dtype::F32);
    assert_eq!(
        fs::read(directory.join("values.f32")).unwrap(),
        [-1.0f32, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    writer
        .write_auxiliary(
            "provenance/snapshot.txt",
            b"captured source",
            "source_snapshot",
        )
        .unwrap();
    writer
        .add_case("sample", "ocr", vec![scalar, empty, values, pixels, public])
        .unwrap();
    assert!(!directory.join("manifest.json").exists());
    writer.finish(json!({})).unwrap();
    assert!(!directory.join(".manifest.json.tmp").exists());
    let report = validate_bundle(&directory).unwrap();
    assert_eq!(
        (
            report.checked_models,
            report.checked_cases,
            report.checked_artifacts,
            report.checked_auxiliary_files
        ),
        (1, 1, 5, 1)
    );
    let recorded = manifest(&directory);
    let bytes: u64 = recorded["models"]
        .as_array()
        .unwrap()
        .iter()
        .chain(recorded["cases"][0]["artifacts"].as_array().unwrap().iter())
        .chain(recorded["auxiliary_files"].as_array().unwrap().iter())
        .map(|descriptor| descriptor["bytes"].as_u64().unwrap())
        .sum();
    assert_eq!(report.checked_bytes, bytes);
    assert!(report.elapsed_ms.unwrap() >= 0.0);
    assert!(
        compare_bundles(
            &directory,
            &directory,
            Tolerance {
                atol: 0.0,
                rtol: 0.0
            }
        )
        .unwrap()
        .matches()
    );
}

#[test]
fn selected_stage_accepts_partial_candidate_and_reports_complete_validation() {
    let temp = TempRoot::new();
    let reference = write_fixture(&temp, "reference", false);
    let candidate = write_fixture(&temp, "candidate", true);
    let report = compare_stage(
        &reference,
        &candidate,
        "sample",
        "cls.node.0000",
        Tolerance::default(),
    )
    .unwrap();
    assert!(report.matches());
    assert_eq!((report.checked_cases, report.checked_artifacts), (1, 1));
    assert_eq!(report.reference_validation.checked_artifacts, 3);
    assert_eq!(report.candidate_validation.checked_artifacts, 1);
    assert_eq!(report.reference_validation.checked_auxiliary_files, 1);
    assert!(report.reference_validation.elapsed_ms.is_none());
    assert!(matches!(report.scope, ComparisonScope::Stage { .. }));
    assert!(report.to_string().starts_with("PARTIAL MATCH"));
    let structured = serde_json::to_value(&report).unwrap();
    assert_eq!(structured["scope"]["kind"], "stage");
    assert!(structured["elapsed_ms"].as_f64().unwrap() >= 0.0);
    assert!(
        !compare_bundles(&reference, &candidate, Tolerance::default())
            .unwrap()
            .matches()
    );
}

#[test]
fn stage_selection_uses_ids_and_preserves_contract_model_and_kind_identity() {
    let temp = TempRoot::new();
    let reference = write_fixture(&temp, "reference", false);
    let candidate = write_fixture(&temp, "candidate", false);
    edit_manifest(&candidate, |manifest| {
        manifest["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .reverse()
    });
    assert!(
        compare_stage(
            &reference,
            &candidate,
            "sample",
            "cls.node.0000",
            Tolerance::default()
        )
        .unwrap()
        .matches()
    );
    edit_manifest(&candidate, |manifest| {
        manifest["contract"]["profile"] = json!("cpu")
    });
    assert!(
        compare_stage(
            &reference,
            &candidate,
            "sample",
            "cls.node.0000",
            Tolerance::default()
        )
        .unwrap()
        .first_difference
        .unwrap()
        .contains("contract")
    );
    edit_manifest(&candidate, |manifest| {
        manifest["contract"] = contract();
        manifest["cases"][0]["kind"] = json!("recognizer");
    });
    assert!(
        compare_stage(
            &reference,
            &candidate,
            "sample",
            "cls.node.0000",
            Tolerance::default()
        )
        .unwrap()
        .first_difference
        .unwrap()
        .contains("kind differs")
    );
    edit_manifest(&candidate, |manifest| {
        manifest["cases"][0]["kind"] = json!("classifier")
    });
    fs::write(candidate.join("models/cls.onnx"), b"other fixture model").unwrap();
    edit_manifest(&candidate, |manifest| {
        manifest["models"][0]["sha256"] = json!(hash(b"other fixture model"));
        manifest["models"][0]["bytes"] = json!(19);
    });
    assert!(
        compare_stage(
            &reference,
            &candidate,
            "sample",
            "cls.node.0000",
            Tolerance::default()
        )
        .unwrap()
        .first_difference
        .unwrap()
        .contains("model cls sha256")
    );
}

#[test]
fn missing_reference_selection_is_invalid_and_missing_candidate_stage_differs() {
    let temp = TempRoot::new();
    let reference = write_fixture(&temp, "reference", false);
    let candidate = write_fixture(&temp, "candidate", true);
    for (case, stage) in [
        ("missing", "cls.node.0000"),
        ("sample", "missing"),
        ("", "cls.node.0000"),
    ] {
        assert!(compare_stage(&reference, &candidate, case, stage, Tolerance::default()).is_err());
    }
    edit_manifest(&candidate, |manifest| {
        manifest["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .clear()
    });
    let report = compare_stage(
        &reference,
        &candidate,
        "sample",
        "cls.node.0000",
        Tolerance::default(),
    )
    .unwrap();
    assert!(!report.matches());
    assert!(report.to_string().starts_with("PARTIAL DIFFERENCE"));
    assert!(
        report
            .first_difference
            .unwrap()
            .contains("missing from candidate")
    );
    assert_eq!(report.reference_validation.checked_artifacts, 3);
}

#[test]
fn early_difference_never_hides_corruption_in_unselected_or_extra_artifacts() {
    let temp = TempRoot::new();
    let reference = write_fixture(&temp, "reference", false);
    let candidate = write_fixture(&temp, "candidate", false);
    fs::write(
        candidate.join("cases/sample/conv.f32"),
        [0.0f32, 1.0, 9.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    edit_manifest(&candidate, |manifest| {
        manifest["cases"][0]["artifacts"][1]["sha256"] = json!(hash(
            &fs::read(candidate.join("cases/sample/conv.f32")).unwrap()
        ))
    });
    fs::write(candidate.join("cases/sample/result.json"), b"corrupt").unwrap();
    assert!(
        compare_stage(
            &reference,
            &candidate,
            "sample",
            "cls.node.0000",
            Tolerance::default()
        )
        .is_err()
    );
    assert!(compare_bundles(&reference, &candidate, Tolerance::default()).is_err());
    edit_manifest(&candidate, |manifest| {
        manifest["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .pop();
        manifest["cases"].as_array_mut().unwrap().push(json!({"id":"extra", "kind":"ocr", "artifacts":[{
            "id":"bad", "path":"cases/sample/result.json", "dtype":"json", "shape":[], "layout":"json", "bytes":7, "sha256":hash(b"corrupt")
        }]}));
    });
    assert!(
        compare_stage(
            &reference,
            &candidate,
            "sample",
            "cls.node.0000",
            Tolerance::default()
        )
        .unwrap_err()
        .to_string()
        .contains("invalid JSON")
    );
    assert!(
        compare_bundles(&reference, &candidate, Tolerance::default())
            .unwrap_err()
            .to_string()
            .contains("invalid JSON")
    );
}

#[test]
fn nonfinite_values_in_unselected_float_artifacts_are_invalid() {
    let temp = TempRoot::new();
    let reference = write_fixture(&temp, "reference", false);
    let candidate = write_fixture(&temp, "candidate", false);
    let values = [0.0f32, f32::INFINITY, 2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    fs::write(candidate.join("cases/sample/conv.f32"), &values).unwrap();
    edit_manifest(&candidate, |manifest| {
        manifest["cases"][0]["artifacts"][1]["sha256"] = json!(hash(&values))
    });
    assert!(
        compare_stage(
            &reference,
            &candidate,
            "sample",
            "input",
            Tolerance::default()
        )
        .unwrap_err()
        .to_string()
        .contains("nonfinite f32 at index 1")
    );
    assert!(validate_bundle(&candidate).is_err());
}

#[test]
fn auxiliary_integrity_is_checked_but_provenance_content_is_not_parity() {
    let temp = TempRoot::new();
    let reference = write_fixture(&temp, "reference", false);
    let candidate = write_fixture(&temp, "candidate", false);
    fs::write(candidate.join("graph-inventory.json"), b"[1]\n").unwrap();
    edit_manifest(&candidate, |manifest| {
        manifest["auxiliary_files"][0]["sha256"] = json!(hash(b"[1]\n"));
        manifest["auxiliary_files"][0]["bytes"] = json!(4);
    });
    assert!(
        compare_bundles(&reference, &candidate, Tolerance::default())
            .unwrap()
            .matches()
    );
    fs::write(candidate.join("graph-inventory.json"), b"[2]\n").unwrap();
    assert!(
        validate_bundle(&candidate)
            .unwrap_err()
            .to_string()
            .contains("sha256 mismatch")
    );
    assert!(
        compare_stage(
            &reference,
            &candidate,
            "sample",
            "input",
            Tolerance::default()
        )
        .is_err()
    );
    edit_manifest(&candidate, |manifest| {
        manifest
            .as_object_mut()
            .unwrap()
            .remove("auxiliary_files")
            .map(|_| ())
            .unwrap()
    });
    let report = validate_bundle(&candidate).unwrap();
    assert_eq!(report.checked_auxiliary_files, 0);
    assert!(
        compare_bundles(&reference, &candidate, Tolerance::default())
            .unwrap()
            .matches()
    );
}

#[test]
fn chunk_boundary_mismatches_have_global_indices_and_complete_metrics() {
    let temp = TempRoot::new();
    let count = STREAM_BLOCK_BYTES / 4 + 7;
    for (name, candidate) in [("reference", false), ("candidate", true)] {
        let mut writer = BundleWriter::new(&temp.path(name), contract()).unwrap();
        let mut values = vec![1.0f32; count];
        if candidate {
            values[STREAM_BLOCK_BYTES / 4] = 3.0;
            values[count - 1] = 4.0;
        }
        let artifact = writer
            .write_floats("stage", "stage.f32", &values, &[count], "row_major")
            .unwrap();
        writer.add_case("sample", "tensor", vec![artifact]).unwrap();
        writer.finish(json!({})).unwrap();
    }
    let report = compare_bundles(
        &temp.path("reference"),
        &temp.path("candidate"),
        Tolerance::default(),
    )
    .unwrap();
    let difference = report.first_difference.unwrap();
    assert!(
        difference.contains(&format!(
            "count=2/{count}, first index={}",
            STREAM_BLOCK_BYTES / 4
        )),
        "{difference}"
    );
    assert!(difference.contains("max abs=3.000000000e0"), "{difference}");
    assert_eq!(
        report.reference_validation.checked_bytes,
        (count * 4) as u64
    );
    assert_eq!(
        report.candidate_validation.checked_bytes,
        (count * 4) as u64
    );
}

#[test]
fn byte_comparison_keeps_exact_mismatch_counts_across_chunks() {
    let temp = TempRoot::new();
    let count = STREAM_BLOCK_BYTES + 3;
    for (name, candidate) in [("reference", false), ("candidate", true)] {
        let mut writer = BundleWriter::new(&temp.path(name), contract()).unwrap();
        let mut values = vec![1; count];
        if candidate {
            values[STREAM_BLOCK_BYTES - 1] = 2;
            values[STREAM_BLOCK_BYTES] = 3;
            values[count - 1] = 4;
        }
        let artifact = writer
            .write_bytes("stage", "stage.u8", &values, &[count], "row_major")
            .unwrap();
        writer.add_case("sample", "tensor", vec![artifact]).unwrap();
        writer.finish(json!({})).unwrap();
    }
    let report = compare_bundles(
        &temp.path("reference"),
        &temp.path("candidate"),
        Tolerance {
            atol: 100.0,
            rtol: 100.0,
        },
    )
    .unwrap();
    let difference = report.first_difference.unwrap();
    assert!(
        difference.contains(&format!(
            "count=3/{count}, first index={}",
            STREAM_BLOCK_BYTES - 1
        )),
        "{difference}"
    );
    assert_eq!(report.reference_validation.checked_bytes, count as u64);
}

#[test]
fn json_size_limit_and_auxiliary_path_shape_are_enforced() {
    let temp = TempRoot::new();
    let candidate = write_fixture(&temp, "candidate", false);
    edit_manifest(&candidate, |manifest| {
        manifest["cases"][0]["artifacts"][2]["bytes"] = json!(MAX_JSON_BYTES + 1)
    });
    assert!(
        validate_bundle(&candidate)
            .unwrap_err()
            .to_string()
            .contains("JSON exceeds")
    );
    edit_manifest(&candidate, |manifest| {
        manifest["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .pop();
        manifest["auxiliary_files"][0]["path"] = json!("../outside");
    });
    assert!(
        validate_bundle(&candidate)
            .unwrap_err()
            .to_string()
            .contains("unsafe bundle path")
    );
}

#[test]
fn writer_rejects_overwrite_invalid_outputs_and_unattached_or_forged_artifacts() {
    let temp = TempRoot::new();
    let directory = temp.path("fresh");
    let mut writer = BundleWriter::new(&directory, contract()).unwrap();
    assert!(BundleWriter::new(&directory, contract()).is_err());
    assert!(
        writer
            .write_floats("bad", "bad.f32", &[f32::NAN], &[1], "row_major")
            .is_err()
    );
    assert!(
        writer
            .write_bytes("bad", "bad.u8", &[1], &[2], "row_major")
            .is_err()
    );
    assert!(
        writer
            .write_bytes("bad", "../outside.u8", &[1], &[1], "row_major")
            .is_err()
    );
    assert!(
        writer
            .write_auxiliary("manifest.json", b"{}", "metadata")
            .is_err()
    );
    assert!(
        writer
            .write_auxiliary("MANIFEST.JSON", b"{}", "metadata")
            .is_err()
    );
    assert!(!temp.path("outside.u8").exists());
    assert!(!directory.join("manifest.json").exists());
    let artifact = writer
        .write_bytes("stage", "stage.u8", &[1], &[1], "row_major")
        .unwrap();
    assert!(
        writer
            .write_bytes("stage", "stage.u8", &[1], &[1], "row_major")
            .is_err()
    );
    let mut forged = artifact.clone();
    forged.sha256 = "0".repeat(64);
    assert!(writer.add_case("sample", "tensor", vec![forged]).is_err());
    assert!(
        writer
            .finish(json!({}))
            .unwrap_err()
            .to_string()
            .contains("attached")
    );
    assert!(!directory.join("manifest.json").exists());
    let writer = BundleWriter::new(&temp.path("reserved"), contract()).unwrap();
    assert!(writer.finish(json!({"schema_version":2})).is_err());
    assert!(!temp.path("reserved/manifest.json").exists());
}

#[test]
fn auxiliary_roles_duplicate_paths_and_manifest_self_reference_are_invalid() {
    let temp = TempRoot::new();
    let candidate = write_fixture(&temp, "candidate", false);
    edit_manifest(&candidate, |manifest| {
        manifest["auxiliary_files"][0]["role"] = json!("")
    });
    assert!(
        validate_bundle(&candidate)
            .unwrap_err()
            .to_string()
            .contains("role is empty")
    );
    edit_manifest(&candidate, |manifest| {
        manifest["auxiliary_files"][0]["role"] = json!("inventory");
        let duplicate = manifest["auxiliary_files"][0].clone();
        manifest["auxiliary_files"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
    });
    assert!(
        validate_bundle(&candidate)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    edit_manifest(&candidate, |manifest| {
        manifest["auxiliary_files"].as_array_mut().unwrap().pop();
        manifest["auxiliary_files"][0]["path"] = json!("manifest.json");
    });
    assert!(
        validate_bundle(&candidate)
            .unwrap_err()
            .to_string()
            .contains("self-referential")
    );
}

#[test]
fn writer_rejects_linked_output_parents_before_creating_outside_files() {
    let temp = TempRoot::new();
    let outside = temp.path("outside");
    fs::create_dir(&outside).unwrap();
    let directory = temp.path("bundle");
    let mut writer = BundleWriter::new(&directory, contract()).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(&outside, directory.join("linked")).unwrap();
    #[cfg(windows)]
    {
        let status = std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", "New-Item -ItemType Junction -Path $env:HARNESS_LINK -Target $env:HARNESS_TARGET -ErrorAction Stop | Out-Null"])
            .env("HARNESS_LINK", directory.join("linked"))
            .env("HARNESS_TARGET", &outside)
            .status().unwrap();
        assert!(status.success());
    }
    assert!(
        writer
            .write_bytes("pixels", "linked/child/pixels.u8", &[1], &[1], "row_major")
            .unwrap_err()
            .to_string()
            .contains("outside its bundle")
    );
    assert!(!outside.join("child").exists());
    assert!(!directory.join("manifest.json").exists());
}
