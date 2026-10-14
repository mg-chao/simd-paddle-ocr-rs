use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{
    ComparisonScope, Tolerance,
    bundle::BundleWriter,
    cls::{load_cls_inputs, run_cls},
    compare_cls, validate_bundle,
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

struct TempRoot(PathBuf);
impl TempRoot {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "simd-ocr-cls-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed),
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
        let resolved = self.0.canonicalize().unwrap();
        assert!(resolved.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        assert!(
            self.0
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("simd-ocr-cls-")
        );
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn varint(mut value: u64) -> Vec<u8> {
    let mut result = Vec::new();
    while value >= 128 {
        result.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    result.push(value as u8);
    result
}
fn number(field: u64, value: u64) -> Vec<u8> {
    [varint(field << 3), varint(value)].concat()
}
fn bytes(field: u64, value: &[u8]) -> Vec<u8> {
    [
        varint((field << 3) | 2),
        varint(value.len() as u64),
        value.to_vec(),
    ]
    .concat()
}
fn info(name: &str, shape: &[usize]) -> Vec<u8> {
    let dims: Vec<u8> = shape
        .iter()
        .flat_map(|dimension| bytes(1, &number(1, *dimension as u64)))
        .collect();
    let tensor_type = [number(1, 1), bytes(2, &dims)].concat();
    [bytes(1, name.as_bytes()), bytes(2, &bytes(1, &tensor_type))].concat()
}
fn int_attribute(name: &str, value: u64) -> Vec<u8> {
    [bytes(1, name.as_bytes()), number(3, value), number(20, 2)].concat()
}
fn node(name: &str, op: &str, inputs: &[&str], output: &str, attributes: &[Vec<u8>]) -> Vec<u8> {
    let mut records: Vec<u8> = inputs
        .iter()
        .flat_map(|input| bytes(1, input.as_bytes()))
        .collect();
    records.extend(bytes(2, output.as_bytes()));
    records.extend(bytes(3, name.as_bytes()));
    records.extend(bytes(4, op.as_bytes()));
    for attribute in attributes {
        records.extend(bytes(5, attribute));
    }
    bytes(1, &records)
}
fn classifier_model() -> Vec<u8> {
    let axes = [bytes(1, b"axes"), number(8, 2), number(8, 3), number(20, 7)].concat();
    let weights: Vec<u8> = [1f32, 0., 0., 1., 0., 0.]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let tensor = [
        number(1, 3),
        number(1, 2),
        number(2, 1),
        bytes(8, b"w"),
        bytes(9, &weights),
    ]
    .concat();
    let graph = [
        node(
            "mean",
            "ReduceMean",
            &["x"],
            "mean",
            &[axes, int_attribute("keepdims", 0)],
        ),
        node("head", "MatMul", &["mean", "w"], "logits", &[]),
        node(
            "probabilities",
            "Softmax",
            &["logits"],
            "y",
            &[int_attribute("axis", 1)],
        ),
        bytes(5, &tensor),
        bytes(11, &info("x", &[1, 3, 80, 160])),
        bytes(12, &info("y", &[1, 2])),
    ]
    .concat();
    [number(1, 8), bytes(8, &number(2, 13)), bytes(7, &graph)].concat()
}
fn fixture(temp: &TempRoot, name: &str) -> PathBuf {
    let model = temp.path("model.onnx");
    fs::write(&model, classifier_model()).unwrap();
    let dictionary = temp.path("keys.txt");
    fs::write(&dictionary, b"fixture\n").unwrap();
    let root = temp.path(name);
    let mut writer = BundleWriter::new(&root, json!({
        "profile":"canonical", "backend":"cpu", "logical_input_layout":"nchw", "nodes":"all", "max_traced_lines":0,
    })).unwrap();
    for id in ["det", "cls", "rec"] {
        writer
            .copy_model(id, &model, &format!("models/{id}.onnx"))
            .unwrap();
    }
    writer
        .copy_model("dictionary", &dictionary, "models/keys.txt")
        .unwrap();
    let inventory = json!([{"id":"cls", "nodes":[
        {"index":0,"name":"mean","op_type":"ReduceMean","domain":"","opset":13},
        {"index":1,"name":"head","op_type":"MatMul","domain":"","opset":13},
        {"index":2,"name":"probabilities","op_type":"Softmax","domain":"","opset":13},
    ]}]);
    writer
        .write_auxiliary(
            "graph-inventory.json",
            &serde_json::to_vec(&inventory).unwrap(),
            "graph_inventory",
        )
        .unwrap();
    let mut artifacts = Vec::new();
    for line in [3, 0] {
        let prefix = format!("line.{line:03}.cls");
        let directory = format!("cases/sample/{prefix}");
        let mut input = Vec::new();
        for channel in [1f32, 2., 3.] {
            input.extend(std::iter::repeat_n(channel, 80 * 160));
        }
        let p = (-1f32).exp();
        let probabilities = [p / (p + 1.), 1. / (p + 1.)];
        artifacts.push(
            writer
                .write_floats(
                    &format!("{prefix}.input"),
                    &format!("{directory}/input.f32"),
                    &input,
                    &[1, 3, 80, 160],
                    "nchw",
                )
                .unwrap(),
        );
        for (ordinal, shape, values) in [
            (0, &[1, 3][..], &[1., 2., 3.][..]),
            (1, &[1, 2][..], &[1., 2.][..]),
            (2, &[1, 2][..], &probabilities[..]),
        ] {
            artifacts.push(
                writer
                    .write_floats(
                        &format!("{prefix}.node.{ordinal:04}"),
                        &format!("{directory}/node-{ordinal:04}.f32"),
                        values,
                        shape,
                        "row_major",
                    )
                    .unwrap(),
            );
        }
        artifacts.push(
            writer
                .write_floats(
                    &format!("{prefix}.probabilities"),
                    &format!("{directory}/probabilities.f32"),
                    &probabilities,
                    &[1, 2],
                    "row_major",
                )
                .unwrap(),
        );
        artifacts.push(
            writer
                .write_json(
                    &format!("{prefix}.public"),
                    &format!("{directory}/public.json"),
                    &json!({"label":1,"score":probabilities[1],"resized_width":160}),
                )
                .unwrap(),
        );
    }
    artifacts.push(
        writer
            .write_bytes(
                "unimplemented.det",
                "cases/sample/unimplemented.u8",
                &[1, 2, 3],
                &[3],
                "row_major",
            )
            .unwrap(),
    );
    writer.add_case("sample", "ocr", artifacts).unwrap();
    writer.add_case("blank", "ocr", Vec::new()).unwrap();
    writer
        .finish(json!({"provenance":{"fixture":true}}))
        .unwrap();
    root
}
fn manifest(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path.join("manifest.json")).unwrap()).unwrap()
}
fn edit_manifest(path: &Path, edit: impl FnOnce(&mut Value)) {
    let mut value = manifest(path);
    edit(&mut value);
    fs::write(
        path.join("manifest.json"),
        serde_json::to_vec_pretty(&value).unwrap(),
    )
    .unwrap();
}

#[test]
fn frozen_reader_and_candidate_cover_every_line_in_manifest_order() {
    let temp = TempRoot::new();
    let reference = fixture(&temp, "reference");
    let inputs = load_cls_inputs(&reference).unwrap();
    assert_eq!(inputs.inputs.len(), 2);
    assert_eq!(inputs.inputs[0].artifact_id, "line.003.cls.input");
    assert_eq!(inputs.inputs[1].artifact_id, "line.000.cls.input");
    assert_eq!(inputs.inputs[0].values.len(), 3 * 80 * 160);
    let candidate = temp.path("candidate");
    let report = run_cls(&reference, &candidate).unwrap();
    assert_eq!(report.processed_lines, 2);
    assert_eq!(report.written_artifacts, 10);
    let value = manifest(&candidate);
    assert_eq!(value["models"].as_array().unwrap().len(), 4);
    assert_eq!(value["cases"].as_array().unwrap().len(), 2);
    assert!(
        value["cases"][1]["artifacts"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(
        value["cases"][0]["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .all(|artifact| !artifact["id"].as_str().unwrap().ends_with(".public"))
    );
    validate_bundle(&candidate).unwrap();
    let comparison = compare_cls(&reference, &candidate, Tolerance::default()).unwrap();
    assert!(comparison.matches(), "{comparison}");
    assert_eq!(comparison.checked_artifacts, 10);
    assert_eq!(comparison.reference_validation.checked_artifacts, 13);
    assert_eq!(comparison.candidate_validation.checked_artifacts, 10);
    assert_eq!(comparison.reference_validation.checked_cases, 2);
    assert_eq!(comparison.candidate_validation.checked_cases, 2);
    assert_eq!(
        comparison.scope,
        ComparisonScope::Cls {
            required_lines: 2,
            required_artifacts: 10
        }
    );
    assert!(comparison.to_string().starts_with("PARTIAL MATCH"));
    assert!(run_cls(&reference, &candidate).is_err());
}

#[test]
fn missing_or_extra_cls_artifacts_fail_and_unselected_corruption_takes_precedence() {
    let temp = TempRoot::new();
    let reference = fixture(&temp, "reference");
    let candidate = temp.path("candidate");
    run_cls(&reference, &candidate).unwrap();
    edit_manifest(&candidate, |value| {
        value["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .retain(|artifact| artifact["id"] != "line.003.cls.node.0001");
    });
    let missing = compare_cls(&reference, &candidate, Tolerance::default()).unwrap();
    assert!(!missing.matches());
    assert!(
        missing
            .first_difference
            .unwrap()
            .contains("missing from candidate")
    );
    fs::write(reference.join("cases/sample/unimplemented.u8"), [9, 8, 7]).unwrap();
    assert!(compare_cls(&reference, &candidate, Tolerance::default()).is_err());

    let other_reference = fixture(&temp, "other-reference");
    let other_candidate = temp.path("other-candidate");
    run_cls(&other_reference, &other_candidate).unwrap();
    edit_manifest(&other_candidate, |value| {
        let mut extra = value["cases"][0]["artifacts"][0].clone();
        extra["id"] = json!("line.999.cls.input");
        value["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .push(extra);
    });
    let extra = compare_cls(&other_reference, &other_candidate, Tolerance::default()).unwrap();
    assert!(!extra.matches());
    assert!(
        extra
            .first_difference
            .unwrap()
            .contains("extra classifier artifact")
    );
}

#[test]
fn zero_capture_noncanonical_or_invalid_inventory_is_rejected_before_candidate_creation() {
    let temp = TempRoot::new();
    let reference = fixture(&temp, "reference");
    edit_manifest(&reference, |value| {
        value["contract"]["profile"] = json!("cpu")
    });
    let output = temp.path("must-not-exist");
    assert!(run_cls(&reference, &output).is_err());
    assert!(!output.exists());
    edit_manifest(&reference, |value| {
        value["contract"]["profile"] = json!("canonical");
        for case in value["cases"].as_array_mut().unwrap() {
            case["artifacts"].as_array_mut().unwrap().clear();
        }
    });
    assert!(
        compare_cls(&reference, &reference, Tolerance::default())
            .unwrap_err()
            .to_string()
            .contains("no captured classifier")
    );
    assert!(load_cls_inputs(&reference).is_err());

    let other = fixture(&temp, "other");
    let inventory_path = other.join("graph-inventory.json");
    let mut inventory: Value = serde_json::from_slice(&fs::read(&inventory_path).unwrap()).unwrap();
    inventory[0]["nodes"][0]["index"] = json!(1);
    let bytes = serde_json::to_vec(&inventory).unwrap();
    fs::write(&inventory_path, &bytes).unwrap();
    edit_manifest(&other, |value| {
        value["auxiliary_files"][0]["sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
        value["auxiliary_files"][0]["bytes"] = json!(bytes.len());
    });
    assert!(
        run_cls(&other, &output)
            .unwrap_err()
            .to_string()
            .contains("inventory differs")
    );
    assert!(!output.exists());
}

#[test]
fn cls_cli_writes_candidate_and_reports_partial_scope() {
    let temp = TempRoot::new();
    let reference = fixture(&temp, "reference");
    let candidate = temp.path("candidate");
    let executable = env!("CARGO_BIN_EXE_simd-paddle-ocr-rs");
    let output = Command::new(executable)
        .arg("run-cls")
        .arg(&reference)
        .arg("--out")
        .arg(&candidate)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(executable)
        .arg("compare-cls")
        .arg(&reference)
        .arg(&candidate)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["scope"]["kind"], "cls");
    assert_eq!(report["scope"]["required_lines"], 2);
}

#[test]
fn classifier_selection_uses_ids_and_never_hides_nonfinite_extra_payloads() {
    let temp = TempRoot::new();
    let reference = fixture(&temp, "reference");
    let candidate = temp.path("candidate");
    run_cls(&reference, &candidate).unwrap();
    edit_manifest(&candidate, |value| {
        value["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .reverse();
        value["cases"].as_array_mut().unwrap().reverse();
    });
    assert!(
        compare_cls(&reference, &candidate, Tolerance::default())
            .unwrap()
            .matches()
    );
    edit_manifest(&candidate, |value| {
        let artifacts = value["cases"][1]["artifacts"].as_array_mut().unwrap();
        artifacts
            .iter_mut()
            .find(|artifact| artifact["id"] == "line.003.cls.probabilities")
            .unwrap()["shape"] = json!([2, 1]);
    });
    let mismatch = compare_cls(&reference, &candidate, Tolerance::default()).unwrap();
    assert!(!mismatch.matches());
    assert!(
        mismatch
            .first_difference
            .unwrap()
            .contains("metadata differs")
    );
    let bytes = f32::NAN.to_le_bytes();
    fs::write(candidate.join("extra.f32"), bytes).unwrap();
    edit_manifest(&candidate, |value| {
        value["cases"][1]["artifacts"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "id":"extra.unselected", "path":"extra.f32", "dtype":"f32", "shape":[1],
                "layout":"row_major", "bytes":4, "sha256":format!("{:x}", Sha256::digest(bytes)),
            }));
    });
    assert!(
        compare_cls(&reference, &candidate, Tolerance::default())
            .unwrap_err()
            .to_string()
            .contains("nonfinite")
    );
}

#[test]
#[ignore = "requires SIMDOCR_CLS_REFERENCE pointing at a canonical all-node local archive"]
fn all_frozen_cls_lines_match_authoritative_canonical_reference() {
    let reference =
        PathBuf::from(std::env::var("SIMDOCR_CLS_REFERENCE").expect("set SIMDOCR_CLS_REFERENCE"));
    let value = manifest(&reference);
    assert_eq!(value["contract"]["nodes"], "all");
    assert_eq!(value["contract"]["max_traced_lines"], 0);
    let temp = TempRoot::new();
    let candidate = temp.path("candidate");
    let run = run_cls(&reference, &candidate).unwrap();
    let comparison = compare_cls(&reference, &candidate, Tolerance::default()).unwrap();
    assert!(comparison.matches(), "{comparison}");
    assert_eq!(
        run.processed_lines,
        match &comparison.scope {
            ComparisonScope::Cls { required_lines, .. } => *required_lines,
            _ => unreachable!(),
        }
    );
    let candidate_manifest = manifest(&candidate);
    let mut labels = 0;
    for case in value["cases"].as_array().unwrap() {
        let candidate_case = candidate_manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|other| other["id"] == case["id"])
            .unwrap();
        for artifact in case["artifacts"].as_array().unwrap() {
            let id = artifact["id"].as_str().unwrap();
            if let Some(prefix) = id.strip_suffix(".cls.public") {
                let public: Value = serde_json::from_slice(
                    &fs::read(reference.join(artifact["path"].as_str().unwrap())).unwrap(),
                )
                .unwrap();
                let probabilities_id = format!("{prefix}.cls.probabilities");
                let output = candidate_case["artifacts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|other| other["id"] == probabilities_id)
                    .unwrap();
                let bytes = fs::read(candidate.join(output["path"].as_str().unwrap())).unwrap();
                assert_eq!(bytes.len(), 8);
                let first = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
                let second = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
                assert_eq!(
                    u64::from(second > first),
                    public["label"].as_u64().unwrap(),
                    "case {}, line {prefix}",
                    case["id"]
                );
                labels += 1;
            }
        }
    }
    assert_eq!(labels, run.processed_lines);
}
