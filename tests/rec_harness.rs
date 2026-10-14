use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{
    Tolerance,
    bundle::BundleWriter,
    model::Model,
    rec::{compare_rec, run_rec},
    runtime::{CompiledModel, ExecutionLimits, Session},
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "simd-ocr-rec-harness-{}-{}",
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
                .starts_with("simd-ocr-rec-harness-")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
fn varint(mut value: u64) -> Vec<u8> {
    let mut output = Vec::new();
    while value >= 128 {
        output.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    output.push(value as u8);
    output
}
fn number(field: u64, value: i64) -> Vec<u8> {
    [varint(field << 3), varint(value as u64)].concat()
}
fn bytes(field: u64, value: Vec<u8>) -> Vec<u8> {
    [varint(field << 3 | 2), varint(value.len() as u64), value].concat()
}
fn string(field: u64, value: &str) -> Vec<u8> {
    bytes(field, value.as_bytes().to_vec())
}
fn info(name: &str, shape: &[i64]) -> Vec<u8> {
    [
        string(1, name),
        bytes(
            2,
            bytes(
                1,
                [
                    number(1, 1),
                    bytes(
                        2,
                        shape
                            .iter()
                            .flat_map(|dimension| bytes(1, number(1, *dimension)))
                            .collect(),
                    ),
                ]
                .concat(),
            ),
        ),
    ]
    .concat()
}
fn node(name: &str, op: &str, inputs: &[&str], output: &str, attributes: Vec<Vec<u8>>) -> Vec<u8> {
    bytes(
        1,
        [
            inputs.iter().flat_map(|input| string(1, input)).collect(),
            string(2, output),
            string(3, name),
            string(4, op),
            attributes
                .into_iter()
                .flat_map(|attribute| bytes(5, attribute))
                .collect(),
        ]
        .concat(),
    )
}
fn tensor(name: &str, shape: &[i64], values: &[f32]) -> Vec<u8> {
    bytes(
        5,
        [
            shape
                .iter()
                .flat_map(|dimension| number(1, *dimension))
                .collect(),
            number(2, 1),
            string(8, name),
            bytes(
                9,
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
        ]
        .concat(),
    )
}
fn recognizer_model(cancellation: bool, tensor_width: usize, classes: usize) -> Vec<u8> {
    let inner = 3 * tensor_width;
    let reshape = bytes(
        5,
        [
            number(1, 3),
            number(2, 7),
            string(8, "shape"),
            bytes(
                9,
                [1i64, 48, inner as i64]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
        ]
        .concat(),
    );
    let axis = [string(1, "axis"), number(3, 2), number(20, 2)].concat();
    let mut weights = vec![0.; inner * classes];
    let bias = if cancellation {
        weights[0] = 1e8;
        weights[classes] = -1e8;
        let mut bias = vec![0.5; classes];
        bias[0] = 1.;
        bias
    } else {
        let mut bias = vec![0.; classes];
        bias[1] = 1.;
        bias
    };
    let graph = [
        node("reshape", "Reshape", &["x", "shape"], "feature", vec![]),
        node("head", "MatMul", &["feature", "weights"], "logits", vec![]),
        node("bias", "Add", &["logits", "bias"], "biased", vec![]),
        node("softmax", "Softmax", &["biased"], "y", vec![axis]),
        tensor("weights", &[inner as i64, classes as i64], &weights),
        tensor("bias", &[classes as i64], &bias),
        reshape,
        bytes(11, info("x", &[1, 3, 48, tensor_width as i64])),
        bytes(12, info("y", &[1, 48, classes as i64])),
    ]
    .concat();
    [number(1, 8), bytes(8, number(2, 13)), bytes(7, graph)].concat()
}
fn fixture(temp: &Temp, name: &str) -> PathBuf {
    fixture_with_cancellation(temp, name, false)
}
fn fixture_with_cancellation(temp: &Temp, name: &str, cancellation: bool) -> PathBuf {
    fixture_with_options(temp, name, cancellation, 32, true)
}
fn fixture_with_options(
    temp: &Temp,
    name: &str,
    cancellation: bool,
    tensor_width: usize,
    adaptive: bool,
) -> PathBuf {
    fixture_with_dictionary(temp, name, cancellation, tensor_width, adaptive, false)
}
fn fixture_with_dictionary(
    temp: &Temp,
    name: &str,
    cancellation: bool,
    tensor_width: usize,
    adaptive: bool,
    empty_token: bool,
) -> PathBuf {
    let model_path = temp.path(&format!("{name}.onnx"));
    let classes = if empty_token { 4 } else { 3 };
    fs::write(
        &model_path,
        recognizer_model(cancellation, tensor_width, classes),
    )
    .unwrap();
    let dictionary = temp.path(&format!("{name}.txt"));
    let labels: &[u8] = if empty_token { b"\na\n" } else { b"a\n" };
    fs::write(&dictionary, labels).unwrap();
    let first_token = if empty_token { "" } else { "a" };
    let model = Arc::new(Model::load_path(&model_path).unwrap());
    let compiled = Arc::new(CompiledModel::new(model.clone()).unwrap());
    let root = temp.path(name);
    let mut writer=BundleWriter::new(&root,json!({"model":"tiny","profile":"canonical","backend":"cpu","logical_input_layout":"nchw","nodes":"all","max_traced_lines":0,"return_ctc_alignment":true,"inventory_semantics":"lineage_keys_v1","options":{"recognizer":{"adaptive_width":adaptive,"target_width":if adaptive { 320 } else { tensor_width }}}})).unwrap();
    writer
        .copy_model("rec", &model_path, "models/rec.onnx")
        .unwrap();
    writer
        .copy_model("dictionary", &dictionary, "models/keys.txt")
        .unwrap();
    let graph = model.graph();
    let keys = |ids: &[Option<simd_paddle_ocr_rs::model::TensorId>]| {
        json!(
            ids.iter()
                .map(|id| id.map(|id| graph.tensors[id.0].key.as_str()))
                .collect::<Vec<_>>()
        )
    };
    let inventory = json!([{"id":"rec","inputs":graph.inputs.iter().map(|id|graph.tensors[id.0].key.as_str()).collect::<Vec<_>>(),"outputs":graph.outputs.iter().map(|id|graph.tensors[id.0].key.as_str()).collect::<Vec<_>>(),"nodes":graph.nodes.iter().enumerate().map(|(index,node)|json!({"index":index,"name":node.name,"op_type":node.op_type,"domain":node.domain,"opset":node.opset,"inputs":keys(&node.inputs),"outputs":keys(&node.outputs)})).collect::<Vec<_>>()}]);
    writer
        .write_auxiliary(
            "graph-inventory.json",
            &serde_json::to_vec(&inventory).unwrap(),
            "graph_inventory",
        )
        .unwrap();
    let mut artifacts = Vec::new();
    let values = vec![if cancellation { 1. } else { 0. }; 3 * 48 * tensor_width];
    artifacts.push(
        writer
            .write_bytes(
                "line.000.rec.pixels",
                "case/pixels.bin",
                &vec![if cancellation { 255 } else { 0 }; 3 * 48 * tensor_width],
                &[48, tensor_width, 3],
                "bgr24",
            )
            .unwrap(),
    );
    artifacts.push(
        writer
            .write_floats(
                "line.000.rec.input",
                "case/input.bin",
                &values,
                &[1, 3, 48, tensor_width],
                "nchw",
            )
            .unwrap(),
    );
    let mut session = Session::new(
        compiled,
        &[1, 3, 48, tensor_width],
        ExecutionLimits::default(),
    )
    .unwrap();
    let output = session
        .run_with_trace(&values, |trace| {
            artifacts.push(
                writer
                    .write_floats(
                        &format!("line.000.rec.node.{:04}", trace.index),
                        &format!("case/node-{:04}.bin", trace.index),
                        trace.values,
                        trace.shape.dimensions(),
                        trace.layout,
                    )
                    .unwrap(),
            );
            Ok(())
        })
        .unwrap();
    artifacts.push(
        writer
            .write_floats(
                "line.000.rec.probabilities",
                "case/probabilities.bin",
                output.data(),
                output.shape().dimensions(),
                "row_major",
            )
            .unwrap(),
    );
    for suffix in ["decoded_dense", "public"] {
        // The cancellation oracle is literal: zero-first MatMul then Add
        // chooses blank; bias-first accumulation chooses token a at 0.5.
        let empty = cancellation && suffix == "decoded_dense";
        let score = if empty {
            0.
        } else if suffix == "decoded_dense" {
            output.data()[1]
        } else if cancellation {
            0.5
        } else {
            1.
        };
        let spans = if empty {
            json!([])
        } else {
            json!([{"text":first_token,"score":score,"start_column":0,"end_column":48}])
        };
        artifacts.push(writer.write_json(&format!("line.000.rec.{suffix}"),&format!("case/{suffix}.json"),&json!({"text":if empty { "" } else { first_token },"score":score,"emitted_count":if empty { 0 } else { 1 },"resized_width":tensor_width,"tensor_width":tensor_width,"time_steps":48,"ctc_spans":spans})).unwrap());
    }
    writer.add_case("sample", "ocr", artifacts).unwrap();
    writer.finish(json!({})).unwrap();
    root
}
fn edit_manifest(path: &Path, edit: impl FnOnce(&mut Value)) {
    let file = path.join("manifest.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    edit(&mut value);
    fs::write(file, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
}
fn rewrite_json_artifact(path: &Path, id: &str, edit: impl FnOnce(&mut Value)) {
    let manifest: Value =
        serde_json::from_slice(&fs::read(path.join("manifest.json")).unwrap()).unwrap();
    let artifact = manifest["cases"][0]["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["id"] == id)
        .unwrap();
    let relative = artifact["path"].as_str().unwrap();
    let mut value: Value = serde_json::from_slice(&fs::read(path.join(relative)).unwrap()).unwrap();
    edit(&mut value);
    let bytes = serde_json::to_vec(&value).unwrap();
    fs::write(path.join(relative), &bytes).unwrap();
    edit_manifest(path, |manifest| {
        let artifact = manifest["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|artifact| artifact["id"] == id)
            .unwrap();
        artifact["bytes"] = json!(bytes.len());
        artifact["sha256"] = json!(format!("{:x}", Sha256::digest(bytes)));
    });
}
fn rewrite_inventory(path: &Path, edit: impl FnOnce(&mut Value)) {
    let file = path.join("graph-inventory.json");
    let mut inventory: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    edit(&mut inventory);
    let bytes = serde_json::to_vec(&inventory).unwrap();
    fs::write(file, &bytes).unwrap();
    edit_manifest(path, |manifest| {
        let descriptor = manifest["auxiliary_files"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|descriptor| descriptor["path"] == "graph-inventory.json")
            .unwrap();
        descriptor["bytes"] = json!(bytes.len());
        descriptor["sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
    });
}
fn corrupt_unselected_payload(path: &Path) {
    let bytes = [1u8, 2, 3, 4];
    let relative = "case/unselected.bin";
    fs::write(path.join(relative), bytes).unwrap();
    edit_manifest(path, |manifest| {
        manifest["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":"debug.unselected","path":relative,"dtype":"u8","shape":[4],"layout":"row_major","bytes":4,"sha256":format!("{:x}",Sha256::digest(bytes))}));
    });
    // Same length and an unchanged descriptor make this a hash-integrity
    // failure, not an earlier metadata/length failure.
    fs::write(path.join(relative), [0u8, 2, 3, 4]).unwrap();
}
fn compare_cli(reference: &Path, candidate: &Path, expected_code: i32) {
    let output = Command::new(env!("CARGO_BIN_EXE_simd-paddle-ocr-rs"))
        .arg("compare-rec")
        .arg(reference)
        .arg(candidate)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(expected_code),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
#[test]
fn frozen_rec_harness_preserves_dense_and_public_score_contracts() {
    let temp = Temp::new();
    let reference = fixture(&temp, "reference");
    let candidate = temp.path("candidate");
    let run = run_rec(&reference, &candidate).unwrap();
    assert_eq!(run.processed_lines, 1);
    assert!(
        compare_rec(&reference, &candidate, Tolerance::default())
            .unwrap()
            .matches()
    );
    assert!(run_rec(&reference, &candidate).is_err());
}
#[test]
fn all_node_references_cannot_self_certify_missing_nodes_or_float_structure() {
    let temp = Temp::new();
    let reference = fixture(&temp, "missing");
    edit_manifest(&reference, |value| {
        value["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .retain(|artifact| artifact["id"] != "line.000.rec.node.0000")
    });
    let output = temp.path("missing-output");
    assert!(
        run_rec(&reference, &output)
            .unwrap_err()
            .to_string()
            .contains("missing captures")
    );
    assert!(!output.exists());
    let reference = fixture(&temp, "floating");
    rewrite_json_artifact(&reference, "line.000.rec.public", |value| {
        value["time_steps"] = json!(48.0)
    });
    assert!(
        run_rec(&reference, &temp.path("float-output"))
            .unwrap_err()
            .to_string()
            .contains("exact nonnegative integer")
    );
}
#[test]
fn semantic_inventory_edges_and_unselected_integrity_are_not_optional() {
    let temp = Temp::new();
    let reference = fixture(&temp, "inventory");
    let file = reference.join("graph-inventory.json");
    let mut inventory: Value = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
    inventory[0]["nodes"][0]["inputs"] = json!(["original:wrong"]);
    let bytes = serde_json::to_vec(&inventory).unwrap();
    fs::write(file, &bytes).unwrap();
    edit_manifest(&reference, |manifest| {
        manifest["auxiliary_files"][0]["bytes"] = json!(bytes.len());
        manifest["auxiliary_files"][0]["sha256"] = json!(format!("{:x}", Sha256::digest(bytes)));
    });
    assert!(
        run_rec(&reference, &temp.path("inventory-output"))
            .unwrap_err()
            .to_string()
            .contains("inventory")
    );
    let reference = fixture(&temp, "integrity");
    let candidate = temp.path("integrity-candidate");
    run_rec(&reference, &candidate).unwrap();
    rewrite_json_artifact(&candidate, "line.000.rec.public", |value| {
        value["score"] = json!(2.);
        value["ctc_spans"][0]["score"] = json!(2.);
    });
    fs::write(candidate.join("graph-inventory.json"), b"corrupted").unwrap();
    assert!(
        compare_rec(&reference, &candidate, Tolerance::default())
            .unwrap_err()
            .to_string()
            .contains("byte length")
    );
}
#[test]
fn candidate_rec_inventory_cannot_self_certify_rehashed_semantic_drift() {
    let temp = Temp::new();
    let reference = fixture(&temp, "reference");
    for field in [
        "graph_input",
        "graph_output",
        "node_input",
        "node_output",
        "name",
        "op_type",
        "domain",
        "opset",
        "index",
    ] {
        let candidate = temp.path(field);
        run_rec(&reference, &candidate).unwrap();
        rewrite_inventory(&candidate, |inventory| match field {
            "graph_input" => inventory[0]["inputs"] = json!(["original:wrong"]),
            "graph_output" => inventory[0]["outputs"] = json!(["original:wrong"]),
            "node_input" => inventory[0]["nodes"][0]["inputs"] = json!(["original:wrong"]),
            "node_output" => inventory[0]["nodes"][0]["outputs"] = json!(["original:wrong"]),
            "opset" => inventory[0]["nodes"][0][field] = json!(999),
            "index" => inventory[0]["nodes"][0][field] = json!(999),
            _ => inventory[0]["nodes"][0][field] = json!("wrong"),
        });
        let error = compare_rec(&reference, &candidate, Tolerance::default())
            .expect_err("unchanged tensors cannot validate a corrupted candidate inventory");
        assert!(
            error.to_string().contains("inventory")
                || error.to_string().contains("graph input/output identity"),
            "{field}: {error}"
        );
        compare_cli(&reference, &candidate, 2);
    }
    let candidate = temp.path("unindexed-inventory");
    run_rec(&reference, &candidate).unwrap();
    edit_manifest(&candidate, |manifest| {
        manifest["auxiliary_files"] = json!([])
    });
    assert!(
        compare_rec(&reference, &candidate, Tolerance::default())
            .unwrap_err()
            .to_string()
            .contains("indexed auxiliary")
    );
}
#[test]
fn both_sides_complete_integrity_validation_precedes_rec_semantics() {
    let temp = Temp::new();
    let reference = fixture(&temp, "missing-reference-node");
    let candidate = temp.path("corrupt-candidate-payload");
    run_rec(&reference, &candidate).unwrap();
    edit_manifest(&reference, |manifest| {
        manifest["cases"][0]["artifacts"]
            .as_array_mut()
            .unwrap()
            .retain(|artifact| artifact["id"] != "line.000.rec.node.0000")
    });
    corrupt_unselected_payload(&candidate);
    let error = compare_rec(&reference, &candidate, Tolerance::default()).unwrap_err();
    assert!(error.to_string().contains("unselected.bin"), "{error}");
    assert!(error.to_string().contains("sha256 mismatch"), "{error}");
    compare_cli(&reference, &candidate, 2);

    let reference = fixture(&temp, "corrupt-reference-payload");
    let candidate = temp.path("floating-candidate-structure");
    run_rec(&reference, &candidate).unwrap();
    rewrite_json_artifact(&candidate, "line.000.rec.public", |result| {
        result["time_steps"] = json!(48.0)
    });
    corrupt_unselected_payload(&reference);
    let error = compare_rec(&reference, &candidate, Tolerance::default()).unwrap_err();
    assert!(error.to_string().contains("unselected.bin"), "{error}");
    assert!(error.to_string().contains("sha256 mismatch"), "{error}");
}
#[test]
fn rec_match_missing_extra_score_and_identity_differences_preserve_cli_codes() {
    let temp = Temp::new();
    let reference = fixture(&temp, "reference");
    for kind in ["match", "missing", "extra", "score", "identity"] {
        let candidate = temp.path(kind);
        run_rec(&reference, &candidate).unwrap();
        match kind {
            "missing" => edit_manifest(&candidate, |manifest| {
                manifest["cases"][0]["artifacts"]
                    .as_array_mut()
                    .unwrap()
                    .retain(|artifact| artifact["id"] != "line.000.rec.node.0000")
            }),
            "extra" => edit_manifest(&candidate, |manifest| {
                let artifacts = manifest["cases"][0]["artifacts"].as_array_mut().unwrap();
                let mut extra = artifacts
                    .iter()
                    .find(|artifact| artifact["id"] == "line.000.rec.node.0000")
                    .unwrap()
                    .clone();
                extra["id"] = json!("line.000.rec.node.9999");
                artifacts.push(extra);
            }),
            "score" => rewrite_json_artifact(&candidate, "line.000.rec.public", |result| {
                result["score"] = json!(2.0);
                result["ctc_spans"][0]["score"] = json!(2.0);
            }),
            "identity" => edit_manifest(&candidate, |manifest| {
                manifest["contract"]["profile"] = json!("cpu")
            }),
            _ => {}
        }
        let report = compare_rec(&reference, &candidate, Tolerance::default()).unwrap();
        assert_eq!(report.matches(), kind == "match", "{kind}: {report}");
        compare_cli(&reference, &candidate, if kind == "match" { 0 } else { 1 });
    }
}
#[test]
fn bias_cancellation_preserves_independently_decoded_dense_and_public_forms() {
    let temp = Temp::new();
    let reference = fixture_with_cancellation(&temp, "cancellation-reference", true);
    let candidate = temp.path("candidate");
    let run = run_rec(&reference, &candidate).unwrap();
    assert_eq!(run.processed_lines, 1);
    let dense: Value =
        serde_json::from_slice(&fs::read(candidate.join("case/decoded_dense.json")).unwrap())
            .unwrap();
    let public: Value =
        serde_json::from_slice(&fs::read(candidate.join("case/public.json")).unwrap()).unwrap();
    assert_eq!(dense["text"], "");
    assert_eq!(dense["emitted_count"], 0);
    assert_eq!(dense["ctc_spans"], json!([]));
    assert_eq!(public["text"], "a");
    assert_eq!(public["score"], 0.5);
    assert_eq!(public["emitted_count"], 1);
    assert_eq!(public["ctc_spans"][0]["start_column"], 0);
    assert_eq!(public["ctc_spans"][0]["end_column"], 48);
    assert!(
        compare_rec(&reference, &candidate, Tolerance::default())
            .unwrap()
            .matches()
    );
}
#[test]
fn emitted_empty_dictionary_tokens_keep_count_scores_and_spans() {
    let temp = Temp::new();
    // A nonempty second label keeps the dictionary valid while class one
    // deliberately maps to an interior empty entry.
    let reference = fixture_with_dictionary(&temp, "empty-token-reference", false, 32, true, true);
    let candidate = temp.path("candidate");
    run_rec(&reference, &candidate).unwrap();
    let public: Value =
        serde_json::from_slice(&fs::read(candidate.join("case/public.json")).unwrap()).unwrap();
    assert_eq!(public["text"], "");
    assert_eq!(public["emitted_count"], 1);
    assert_eq!(public["score"], 1.0);
    assert_eq!(public["ctc_spans"][0]["text"], "");
    assert!(
        compare_rec(&reference, &candidate, Tolerance::default())
            .unwrap()
            .matches()
    );
}
#[test]
fn fixed_rec_width_is_independent_of_adaptive_buckets() {
    let temp = Temp::new();
    for width in [1, 33] {
        let reference = fixture_with_options(&temp, &format!("fixed-{width}"), false, width, false);
        let candidate = temp.path(&format!("candidate-{width}"));
        run_rec(&reference, &candidate).unwrap();
        let public: Value =
            serde_json::from_slice(&fs::read(candidate.join("case/public.json")).unwrap()).unwrap();
        assert_eq!(public["resized_width"], width);
        assert_eq!(public["tensor_width"], width);
        assert!(
            compare_rec(&reference, &candidate, Tolerance::default())
                .unwrap()
                .matches()
        );
    }
}
#[test]
#[ignore = "requires SIMDOCR_REC_REFERENCE or the archived tiny canonical all-node reference; explicit milestone gate"]
fn all_frozen_tiny_rec_lines_match_authoritative_native_reference() {
    let reference = std::env::var_os("SIMDOCR_REC_REFERENCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("reference-bundles/tiny-canonical-allnodes-v1")
        });
    let temp = Temp::new();
    let candidate = temp.path("native-candidate");
    let run = run_rec(&reference, &candidate).unwrap();
    assert!(run.processed_lines > 0);
    assert!(
        compare_rec(&reference, &candidate, Tolerance::default())
            .unwrap()
            .matches()
    );
}
