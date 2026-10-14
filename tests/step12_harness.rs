use serde_json::{Value, json};
use simd_paddle_ocr_rs::{
    ComparisonScope, Tolerance, bundle::BundleWriter, compare_det, compare_step12, det::run_det,
    step12::run_step12, validate_bundle,
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "simd-step12-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let resolved = self.0.canonicalize().unwrap();
        assert!(resolved.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        assert!(
            resolved
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("simd-step12-")
        );
        fs::remove_dir_all(&resolved).unwrap();
    }
}
fn varint(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    while value >= 128 {
        bytes.push(value as u8 | 128);
        value >>= 7;
    }
    bytes.push(value as u8);
    bytes
}
fn number(field: u64, value: u64) -> Vec<u8> {
    [varint(field << 3), varint(value)].concat()
}
fn bytes(field: u64, value: &[u8]) -> Vec<u8> {
    [
        varint(field << 3 | 2),
        varint(value.len() as u64),
        value.to_vec(),
    ]
    .concat()
}
fn info(name: &str, shape: &[usize]) -> Vec<u8> {
    let dimensions: Vec<u8> = shape
        .iter()
        .flat_map(|d| bytes(1, &number(1, *d as u64)))
        .collect();
    let tensor = [number(1, 1), bytes(2, &dimensions)].concat();
    [bytes(1, name.as_bytes()), bytes(2, &bytes(1, &tensor))].concat()
}
fn model() -> Vec<u8> {
    let axes = [bytes(1, b"axes"), number(8, 1), number(20, 7)].concat();
    let node = [
        bytes(1, b"x"),
        bytes(2, b"mid"),
        bytes(3, b"mean"),
        bytes(4, b"ReduceMean"),
        bytes(5, &axes),
    ]
    .concat();
    let second = [
        bytes(1, b"mid"),
        bytes(2, b"y"),
        bytes(3, b"identity"),
        bytes(4, b"ReduceMean"),
        bytes(5, &axes),
    ]
    .concat();
    let graph = [
        bytes(1, &node),
        bytes(1, &second),
        bytes(11, &info("x", &[1, 3, 2, 2])),
        bytes(12, &info("y", &[1, 1, 2, 2])),
    ]
    .concat();
    [number(1, 8), bytes(8, &number(2, 13)), bytes(7, &graph)].concat()
}
fn fixture(temp: &Temp, name: &str) -> PathBuf {
    let source = temp.path("model.onnx");
    fs::write(&source, model()).unwrap();
    let root = temp.path(name);
    let mut writer=BundleWriter::new(&root,json!({"profile":"canonical","backend":"cpu","logical_input_layout":"nchw","nodes":"all","capture_scope":"step12","max_traced_lines":0,"input_origin":"frozen_bgr_pixels"})).unwrap();
    writer
        .copy_model("det", &source, "models/det.onnx")
        .unwrap();
    writer.write_auxiliary("graph-inventory.json",&serde_json::to_vec(&json!([{"id":"det","inputs":["original:x"],"outputs":["original:y"],"nodes":[{"index":0,"name":"mean","op_type":"ReduceMean","domain":"","opset":13,"inputs":["original:x"],"outputs":["original:mid"]},{"index":1,"name":"identity","op_type":"ReduceMean","domain":"","opset":13,"inputs":["original:mid"],"outputs":["original:y"]}]}])).unwrap(),"graph_inventory").unwrap();
    let mut artifacts = Vec::new();
    artifacts.push(writer.write_json("input.metadata","case/input.json",&json!({"width":2,"height":2,"stride":6,"pixel_format":"bgr24","origin":"independent fixture"})).unwrap());
    artifacts.push(
        writer
            .write_bytes(
                "input.pixels",
                "case/pixels.bin",
                &[0; 12],
                &[2, 6],
                "bgr24_rows",
            )
            .unwrap(),
    );
    artifacts.push(
        writer
            .write_floats(
                "det.input",
                "case/input.f32",
                &[1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
                &[1, 3, 2, 2],
                "nchw",
            )
            .unwrap(),
    );
    for id in ["det.node.0000", "det.node.0001", "det.probabilities"] {
        artifacts.push(
            writer
                .write_floats(
                    id,
                    &format!("case/{id}.f32"),
                    &[2.; 4],
                    &[1, 1, 2, 2],
                    "nchw",
                )
                .unwrap(),
        );
    }
    artifacts.push(writer.write_json("det.boxes","case/boxes.json",&json!({"boxes":[],"resized_width":2,"resized_height":2,"width_ratio":1.,"height_ratio":1.})).unwrap());
    artifacts.push(
        writer
            .write_json(
                "step12.public",
                "case/result.json",
                &json!({"detected_count":0,"detector_resized_width":2,"detector_resized_height":2,"width_ratio":1.,"height_ratio":1.,"lines":[]}),
            )
            .unwrap(),
    );
    artifacts.push(
        writer
            .write_bytes(
                "unselected.extra",
                "case/extra.bin",
                &[7],
                &[1],
                "row_major",
            )
            .unwrap(),
    );
    writer.add_case("sample", "step12", artifacts).unwrap();
    writer
        .finish(json!({"provenance":{"synthetic":true}}))
        .unwrap();
    root
}
fn edit(path: &Path, mut change: impl FnMut(&mut Value)) {
    let file = path.join("manifest.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    change(&mut value);
    fs::write(file, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
}

#[test]
fn detector_harness_executes_and_cannot_hide_missing_extra_or_invalid_payloads() {
    let temp = Temp::new();
    let reference = fixture(&temp, "reference");
    let candidate = temp.path("candidate");
    let run = run_det(&reference, &candidate).unwrap();
    assert_eq!(run.processed_cases, 1);
    assert_eq!(run.written_artifacts, 4);
    let report = compare_det(&reference, &candidate, Tolerance::default()).unwrap();
    assert!(report.matches(), "{report}");
    assert_eq!(
        report.scope,
        ComparisonScope::Det {
            required_cases: 1,
            required_artifacts: 4
        }
    );
    assert!(report.to_string().contains("no pixel preprocessing"));
    assert!(
        !compare_step12(&reference, &candidate, Tolerance::default())
            .unwrap()
            .matches()
    );
    edit(&candidate, |v| {
        v["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .retain(|a| a["id"] != "det.node.0000")
    });
    assert!(
        compare_det(&reference, &candidate, Tolerance::default())
            .unwrap()
            .first_difference
            .unwrap()
            .contains("missing from candidate")
    );
    fs::write(reference.join("case/extra.bin"), [8]).unwrap();
    assert!(compare_det(&reference, &candidate, Tolerance::default()).is_err());
    let other = fixture(&temp, "other");
    let extra = temp.path("extra");
    run_det(&other, &extra).unwrap();
    edit(&extra, |v| {
        let mut artifact = v["cases"][0]["artifacts"][1].clone();
        artifact["id"] = json!("det.node.0002");
        v["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .push(artifact);
    });
    assert!(
        compare_det(&other, &extra, Tolerance::default())
            .unwrap()
            .first_difference
            .unwrap()
            .contains("extra in-scope")
    );
}
#[test]
fn complete_pixel_scope_rejects_incomplete_references_before_creation() {
    let temp = Temp::new();
    let reference = fixture(&temp, "reference");
    assert!(
        compare_step12(&reference, &reference, Tolerance::default())
            .unwrap()
            .matches()
    );
    edit(&reference, |v| v["contract"]["max_traced_lines"] = json!(3));
    let output = temp.path("must-not-exist");
    assert!(run_step12(&reference, &output).is_err());
    assert!(!output.exists());
    edit(&reference, |v| v["contract"]["max_traced_lines"] = json!(0));
    edit(&reference, |v| {
        v["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .retain(|a| a["id"] != "step12.public")
    });
    assert!(compare_step12(&reference, &reference, Tolerance::default()).is_err());
}
fn edit_payload(bundle: &Path, id: &str, mut change: impl FnMut(&mut Value)) {
    let manifest: Value =
        serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    let descriptor = manifest["cases"][0]["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["id"] == id)
        .unwrap();
    let file = bundle.join(descriptor["path"].as_str().unwrap());
    let mut value: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    change(&mut value);
    let bytes = serde_json::to_vec(&value).unwrap();
    fs::write(file, &bytes).unwrap();
    use sha2::Digest;
    let hash = format!("{:x}", sha2::Sha256::digest(&bytes));
    edit(bundle, |manifest| {
        let artifact = manifest["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|artifact| artifact["id"] == id)
            .unwrap();
        artifact["bytes"] = json!(bytes.len());
        artifact["sha256"] = json!(hash);
    });
}
#[test]
fn reference_cannot_self_certify_removed_nodes_or_numeric_structure_type_changes() {
    let temp = Temp::new();
    let removed = fixture(&temp, "removed");
    edit(&removed, |v| {
        v["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .retain(|artifact| artifact["id"] != "det.node.0001")
    });
    assert!(
        compare_det(&removed, &removed, Tolerance::default())
            .unwrap_err()
            .to_string()
            .contains("missing captures")
    );
    assert!(compare_step12(&removed, &removed, Tolerance::default()).is_err());
    let wrong_shape = fixture(&temp, "wrong-shape");
    edit(&wrong_shape, |value| {
        let node = value["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|node| node["id"] == "det.node.0000")
            .unwrap();
        node["shape"] = json!([1, 1, 1, 4]);
    });
    assert!(
        compare_det(&wrong_shape, &wrong_shape, Tolerance::default())
            .unwrap_err()
            .to_string()
            .contains("bound model")
    );
    assert!(compare_step12(&wrong_shape, &wrong_shape, Tolerance::default()).is_err());
    let alias = fixture(&temp, "alias");
    edit(&alias, |v| {
        let artifact = v["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|artifact| artifact["id"] == "det.node.0001")
            .unwrap();
        artifact["id"] = json!("det.node.0");
    });
    assert!(
        compare_det(&alias, &alias, Tolerance::default())
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    let reference = fixture(&temp, "valid");
    let candidate = fixture(&temp, "float-candidate");
    edit_payload(&candidate, "step12.public", |v| {
        v["detector_resized_width"] = json!(2.0)
    });
    assert!(
        compare_step12(&reference, &candidate, Tolerance::default())
            .unwrap_err()
            .to_string()
            .contains("exact nonnegative JSON integer")
    );
    edit_payload(&candidate, "step12.public", |v| {
        v["detector_resized_width"] = json!(2);
        v["detected_count"] = json!(0.000001);
    });
    assert!(compare_step12(&reference, &candidate, Tolerance::default()).is_err());
}
#[test]
fn pixel_scope_rejects_extra_empty_cases() {
    let temp = Temp::new();
    let reference = fixture(&temp, "reference");
    let candidate = fixture(&temp, "candidate");
    edit(&candidate, |value| {
        value["cases"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":"extra-empty","kind":"step12","artifacts":[]}))
    });
    let report = compare_step12(&reference, &candidate, Tolerance::default()).unwrap();
    assert!(!report.matches());
    assert!(
        report
            .first_difference
            .unwrap()
            .contains("extra Step12 case")
    );
}
#[test]
fn classifier_capture_ids_and_public_integer_types_are_checked() {
    let temp = Temp::new();
    let reference = fixture(&temp, "reference");
    let malformed = fixture(&temp, "malformed");
    edit(&malformed, |v| {
        let mut node = v["cases"][0]["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["id"] == "det.node.0000")
            .unwrap()
            .clone();
        node["id"] = json!("line.000.cls.node.foo");
        v["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .push(node);
    });
    assert!(
        compare_step12(&malformed, &malformed, Tolerance::default())
            .unwrap_err()
            .to_string()
            .contains("malformed classifier node")
    );
    let candidate = fixture(&temp, "candidate");
    let bytes=serde_json::to_vec(&json!({"label":1.0,"score":0.5,"orientation_degrees":180,"resized_width":160,"dense_content_width":160})).unwrap();
    fs::write(candidate.join("case/cls-public.json"), &bytes).unwrap();
    use sha2::Digest;
    let hash = format!("{:x}", sha2::Sha256::digest(&bytes));
    edit(&candidate, |v| {
        v["cases"][0]["artifacts"].as_array_mut().unwrap().push(json!({"id":"line.000.cls.public","path":"case/cls-public.json","dtype":"json","shape":[],"layout":"json","bytes":bytes.len(),"sha256":hash}))
    });
    assert!(
        compare_step12(&reference, &candidate, Tolerance::default())
            .unwrap_err()
            .to_string()
            .contains("label must be an exact")
    );
}
#[test]
fn detector_cli_reports_scoped_parity_and_existing_outputs_are_immutable() {
    let temp = Temp::new();
    let reference = fixture(&temp, "reference");
    let candidate = temp.path("candidate");
    let binary = env!("CARGO_BIN_EXE_simd-paddle-ocr-rs");
    let run = Command::new(binary)
        .arg("run-det")
        .arg(&reference)
        .arg("--out")
        .arg(&candidate)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let report = Command::new(binary)
        .arg("compare-det")
        .arg(&reference)
        .arg(&candidate)
        .arg("--json")
        .output()
        .unwrap();
    assert!(report.status.success());
    let json: Value = serde_json::from_slice(&report.stdout).unwrap();
    assert_eq!(json["scope"]["kind"], "det");
    assert!(run_det(&reference, &candidate).is_err());
    validate_bundle(&candidate).unwrap();
}

#[test]
#[ignore = "requires tools/prepare-step12-fixtures.py; CI explicitly runs this gate"]
fn compact_real_models_match_all_detector_and_pixel_stages() {
    let root = std::env::var_os("SIMDOCR_STEP12_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/step12-fixtures"));
    let temp = Temp::new();
    for variant in ["tiny", "small", "medium"] {
        let reference = root.join(variant);
        assert!(
            reference.join("manifest.json").is_file(),
            "prepare fixtures first: {}",
            reference.display()
        );
        let det = temp.path(&format!("{variant}-det"));
        run_det(&reference, &det).unwrap();
        let report = compare_det(&reference, &det, Tolerance::default()).unwrap();
        assert!(report.matches(), "{variant}: {report}");
        let pixel = temp.path(&format!("{variant}-pixel"));
        run_step12(&reference, &pixel).unwrap();
        let report = compare_step12(&reference, &pixel, Tolerance::default()).unwrap();
        assert!(report.matches(), "{variant}: {report}");
    }
    let cls_reference = root.join("cls");
    let cls = temp.path("cls");
    simd_paddle_ocr_rs::cls::run_cls(&cls_reference, &cls).unwrap();
    let report =
        simd_paddle_ocr_rs::compare_cls(&cls_reference, &cls, Tolerance::default()).unwrap();
    assert!(report.matches(), "{report}");
}
