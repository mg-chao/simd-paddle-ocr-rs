use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{
    Tolerance,
    full_ocr::{compare_ocr, run_ocr},
};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "simd-ocr-full-harness-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let root = self.0.canonicalize().unwrap();
        assert!(root.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        assert!(
            root.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("simd-ocr-full-harness-")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
fn reference() -> PathBuf {
    std::env::var_os("SIMDOCR_OCR_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("reference-bundles/tiny-ocr-allnodes-v1")
        })
}
fn copy_bundle(source: &Path, destination: &Path) {
    fs::create_dir(destination).unwrap();
    let mut stack = vec![(source.to_path_buf(), destination.to_path_buf())];
    while let Some((source, destination)) = stack.pop() {
        for entry in fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            let target = destination.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                fs::create_dir(&target).unwrap();
                stack.push((entry.path(), target));
            } else {
                fs::copy(entry.path(), target).unwrap();
            }
        }
    }
}
#[test]
#[ignore = "requires a supplied local semantic tiny OCR all-node reference"]
fn complete_tiny_ocr_matches_every_native_stage_and_public_result() {
    let temp = Temp::new();
    let reference = reference();
    let candidate = temp.path("candidate");
    let run = run_ocr(&reference, &candidate).unwrap();
    assert!(run.processed_cases > 0);
    assert!(run.processed_lines > 0);
    let report = compare_ocr(&reference, &candidate, Tolerance::default()).unwrap();
    assert!(report.matches(), "{report}");
    assert!(run_ocr(&reference, &candidate).is_err());
}

fn verify_frozen_family_ocr(family: &str, variable: &str) {
    let reference = std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("reference-bundles/{family}-ocr-ci-v1"))
        });
    let manifest: Value =
        serde_json::from_slice(&fs::read(reference.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["contract"]["model"], family);
    let temp = Temp::new();
    let candidate = temp.path("candidate");
    let run = run_ocr(&reference, &candidate).unwrap();
    assert!(run.processed_cases > 0);
    assert!(run.processed_lines > 0);
    let report = compare_ocr(&reference, &candidate, Tolerance::default()).unwrap();
    assert!(report.matches(), "{report}");
    assert!(run_ocr(&reference, &candidate).is_err());
}

#[test]
#[ignore = "requires a supplied local all-node small OCR reference"]
fn complete_small_ocr_matches_every_native_stage_and_public_result() {
    verify_frozen_family_ocr("small", "SIMDOCR_SMALL_OCR_REFERENCE");
}

#[test]
#[ignore = "requires a supplied local all-node medium OCR reference"]
fn complete_medium_ocr_matches_every_native_stage_and_public_result() {
    verify_frozen_family_ocr("medium", "SIMDOCR_MEDIUM_OCR_REFERENCE");
}
#[test]
#[ignore = "requires fresh semantic tiny OCR reference for completeness corruption tests"]
fn complete_ocr_rejects_rehashed_omissions_and_out_of_range_lines() {
    let temp = Temp::new();
    let original = reference();
    let modified = temp.path("missing");
    copy_bundle(&original, &modified);
    let file = modified.join("manifest.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    manifest["cases"][0]["artifacts"]
        .as_array_mut()
        .unwrap()
        .retain(|artifact| artifact["id"] != "line.000.rec.public");
    fs::write(&file, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let output = temp.path("omission-output");
    assert!(
        run_ocr(&modified, &output)
            .unwrap_err()
            .to_string()
            .contains("missing")
    );
    assert!(!output.exists());
    let modified = temp.path("extra");
    copy_bundle(&original, &modified);
    let file = modified.join("manifest.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    let pixels = manifest["cases"][0]["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["id"] == "line.000.rec.pixels")
        .unwrap()
        .clone();
    let mut extra = pixels;
    extra["id"] = json!("line.999.rec.pixels");
    manifest["cases"][0]["artifacts"]
        .as_array_mut()
        .unwrap()
        .push(extra);
    fs::write(file, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(
        run_ocr(&modified, &temp.path("extra-output"))
            .unwrap_err()
            .to_string()
            .contains("outside public count")
    );
}
#[test]
#[ignore = "requires fresh semantic tiny OCR reference for malformed public structure gate"]
fn complete_ocr_rejects_floating_integer_fields_even_after_rehash() {
    let temp = Temp::new();
    let modified = temp.path("floating");
    copy_bundle(&reference(), &modified);
    let file = modified.join("manifest.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    let artifact = manifest["cases"][0]["artifacts"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|artifact| artifact["id"] == "result.public")
        .unwrap();
    let path = modified.join(artifact["path"].as_str().unwrap());
    let mut result: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    result["detected_count"] = json!(result["detected_count"].as_u64().unwrap() as f64);
    let bytes = serde_json::to_vec(&result).unwrap();
    fs::write(path, &bytes).unwrap();
    artifact["bytes"] = json!(bytes.len());
    artifact["sha256"] = json!(format!("{:x}", Sha256::digest(bytes)));
    fs::write(file, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(
        run_ocr(&modified, &temp.path("floating-output"))
            .unwrap_err()
            .to_string()
            .contains("exact nonnegative integer")
    );
}
