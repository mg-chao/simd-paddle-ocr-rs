use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::model::{LoadOptions, Model, compare_import, validate_import_reference};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

#[test]
#[ignore = "requires SIMDOCR_IMPORT_REFERENCE pointing at an authoritative local archive"]
fn all_shipped_models_match_authoritative_imports() {
    let reference =
        std::env::var("SIMDOCR_IMPORT_REFERENCE").expect("set SIMDOCR_IMPORT_REFERENCE");
    for id in [
        "cls",
        "det",
        "rec",
        "small_det",
        "small_rec",
        "medium_det",
        "medium_rec",
    ] {
        let report = compare_import(&reference, id, &LoadOptions::default()).unwrap();
        assert!(report.matches(), "{report}");
    }
}

fn varint(mut value: u64) -> Vec<u8> {
    let mut bytes = vec![];
    loop {
        let low = (value & 127) as u8;
        value >>= 7;
        bytes.push(low | if value == 0 { 0 } else { 128 });
        if value == 0 {
            break;
        }
    }
    bytes
}
fn number(field: u64, value: i64) -> Vec<u8> {
    [varint(field << 3), varint(value as u64)].concat()
}
fn bytes(field: u64, value: Vec<u8>) -> Vec<u8> {
    [varint((field << 3) | 2), varint(value.len() as u64), value].concat()
}
fn string(field: u64, value: &str) -> Vec<u8> {
    bytes(field, value.as_bytes().to_vec())
}
fn info(name: &str, dims: &[i64]) -> Vec<u8> {
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
                        dims.iter().flat_map(|&d| bytes(1, number(1, d))).collect(),
                    ),
                ]
                .concat(),
            ),
        ),
    ]
    .concat()
}
fn tensor(name: &str, dims: &[i64], values: &[f32]) -> Vec<u8> {
    bytes(
        5,
        [
            dims.iter().flat_map(|&d| number(1, d)).collect(),
            number(2, 1),
            string(8, name),
            bytes(
                9,
                values
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect(),
            ),
        ]
        .concat(),
    )
}
fn node(name: &str, op: &str, inputs: &[&str], output: &str, attrs: Vec<Vec<u8>>) -> Vec<u8> {
    bytes(
        1,
        [
            inputs.iter().flat_map(|v| string(1, v)).collect(),
            string(2, output),
            string(3, name),
            string(4, op),
            attrs.into_iter().flat_map(|v| bytes(5, v)).collect(),
        ]
        .concat(),
    )
}
fn ints(name: &str, values: &[i64]) -> Vec<u8> {
    [
        string(1, name),
        number(20, 7),
        values.iter().flat_map(|&v| number(8, v)).collect(),
    ]
    .concat()
}
fn model(records: Vec<Vec<u8>>) -> Vec<u8> {
    model_with_opset(records, 13)
}
fn model_with_opset(records: Vec<Vec<u8>>, opset: i64) -> Vec<u8> {
    model_with_shapes(records, opset, &[1, 1, 1, 1], &[1, 1, 1, 1])
}
fn model_with_shapes(
    records: Vec<Vec<u8>>,
    opset: i64,
    input_shape: &[i64],
    output_shape: &[i64],
) -> Vec<u8> {
    [
        number(1, 8),
        bytes(8, number(2, opset)),
        bytes(
            7,
            [
                records.concat(),
                bytes(11, info("x", input_shape)),
                bytes(12, info("y", output_shape)),
            ]
            .concat(),
        ),
    ]
    .concat()
}

#[test]
fn missing_softmax_axis_uses_imported_opset_defaults() {
    use simd_paddle_ocr_rs::model::EffectiveAttributes;
    for (opset, expected) in [(7, 1), (12, 1), (13, -1), (14, -1)] {
        let bytes = model_with_opset(
            vec![node("probability", "Softmax", &["x"], "y", vec![])],
            opset,
        );
        let loaded = Model::load_bytes(&bytes).unwrap();
        assert_eq!(loaded.graph().nodes[0].opset, opset);
        let EffectiveAttributes::Axis(axis) = loaded.graph().nodes[0].attributes else {
            panic!("Softmax axis expected")
        };
        assert_eq!(axis, expected, "opset {opset}");
    }
}

#[test]
fn identity_resolution_drops_dead_data_and_preserves_source_identity() {
    let bytes = model(vec![
        node("alias", "Identity", &["x"], "alias", vec![]),
        node("plus", "Add", &["alias", "one"], "y", vec![]),
        tensor("one", &[], &[1.0]),
        tensor("unused", &[0], &[]),
    ]);
    let options = LoadOptions {
        capture_raw_snapshot: true,
        ..Default::default()
    };
    let loaded = Model::load_bytes_with_options(&bytes, &options).unwrap();
    assert_eq!(loaded.graph().nodes.len(), 1);
    assert_eq!(loaded.graph().nodes[0].key, "node:1");
    assert_eq!(loaded.graph().tensors.len(), 3);
    let raw = loaded.raw_snapshot().unwrap();
    assert!(
        raw.tensors
            .iter()
            .any(|t| t.name == "unused" && t.shape.dims.len() == 1 && t.data_bytes == 0)
    );
    let scalar = loaded
        .graph()
        .tensors
        .iter()
        .find(|t| t.name == "one")
        .unwrap();
    assert!(scalar.constant.as_ref().unwrap().dimensions().is_empty());
}

#[test]
fn cyclic_identity_and_missing_inputs_fail_early() {
    let cycle = model(vec![
        node("a", "Identity", &["b"], "a", vec![]),
        node("b", "Identity", &["a"], "b", vec![]),
        node("out", "Relu", &["a"], "y", vec![]),
    ]);
    assert!(
        Model::load_bytes(&cycle)
            .unwrap_err()
            .to_string()
            .contains("cycle")
    );
    let missing = model(vec![node("out", "Relu", &["missing"], "y", vec![])]);
    assert!(
        Model::load_bytes(&missing)
            .unwrap_err()
            .to_string()
            .contains("unresolved")
    );
}

#[test]
fn centered_parallel_convolution_merges_constants_and_lineage() {
    let one = ints("kernel_shape", &[1, 1]);
    let bytes = model(vec![
        node("first", "Conv", &["x", "a"], "first", vec![one.clone()]),
        node("second", "Conv", &["x", "b"], "second", vec![one]),
        node("sum", "Add", &["first", "second"], "y", vec![]),
        tensor("a", &[1, 1, 1, 1], &[2.0]),
        tensor("b", &[1, 1, 1, 1], &[3.0]),
    ]);
    let loaded = Model::load_bytes(&bytes).unwrap();
    assert_eq!(loaded.graph().nodes.len(), 1);
    assert_eq!(loaded.graph().nodes[0].source_indices, [0, 1, 2]);
    let weight = loaded
        .graph()
        .tensors
        .iter()
        .find(|t| t.key == "generated:branch_weight:2")
        .unwrap();
    assert_eq!(weight.constant.as_ref().unwrap().as_f32().unwrap(), &[5.0]);
}

fn float_attr(name: &str, value: f32) -> Vec<u8> {
    [
        string(1, name),
        number(20, 1),
        varint((2 << 3) | 5),
        value.to_bits().to_le_bytes().to_vec(),
    ]
    .concat()
}
fn int_tensor(name: &str, dims: &[i64], values: &[i64]) -> Vec<u8> {
    bytes(
        5,
        [
            dims.iter().flat_map(|&d| number(1, d)).collect(),
            number(2, 7),
            string(8, name),
            bytes(9, values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ]
        .concat(),
    )
}

#[test]
fn batch_normalization_folds_exact_float_operations() {
    let bytes = model(vec![
        node(
            "conv",
            "Conv",
            &["x", "w", "bias"],
            "conv_out",
            vec![ints("kernel_shape", &[1, 1])],
        ),
        node(
            "bn",
            "BatchNormalization",
            &["conv_out", "gamma", "beta", "mean", "var"],
            "y",
            vec![float_attr("epsilon", 1.0)],
        ),
        tensor("w", &[1, 1, 1, 1], &[2.0]),
        tensor("bias", &[1], &[3.0]),
        tensor("gamma", &[1], &[2.0]),
        tensor("beta", &[1], &[0.5]),
        tensor("mean", &[1], &[1.0]),
        tensor("var", &[1], &[3.0]),
    ]);
    let loaded = Model::load_bytes(&bytes).unwrap();
    assert_eq!(loaded.graph().nodes.len(), 1);
    assert_eq!(loaded.graph().nodes[0].source_indices, [0, 1]);
    let values = |key: &str| {
        loaded
            .graph()
            .tensors
            .iter()
            .find(|t| t.key == key)
            .unwrap()
            .constant
            .as_ref()
            .unwrap()
            .as_f32()
            .unwrap()
    };
    assert_eq!(values("generated:bn_weight:0:1"), &[2.0]);
    assert_eq!(values("generated:bn_bias:0:1"), &[2.5]);
}

#[test]
fn dynamic_shape_program_retains_integer_operands_without_raw_diagnostics() {
    let bytes = model(vec![
        node("shape", "Shape", &["x"], "shape", vec![]),
        node(
            "slice",
            "Slice",
            &["shape", "starts", "ends"],
            "target",
            vec![],
        ),
        node("alias", "Identity", &["target"], "target_alias", vec![]),
        node("reshape", "Reshape", &["x", "target_alias"], "y", vec![]),
        int_tensor("starts", &[1], &[0]),
        int_tensor("ends", &[1], &[4]),
    ]);
    let loaded = Model::load_bytes(&bytes).unwrap();
    assert!(loaded.raw_snapshot().is_none());
    assert_eq!(loaded.graph().nodes.len(), 1);
    assert_eq!(loaded.shape_recipes()[0].source_indices, [0, 1]);
    assert_eq!(loaded.shape_metadata().nodes.len(), 2);
    assert_eq!(loaded.shape_metadata().constants.len(), 2);
    let start = loaded
        .shape_metadata()
        .constants
        .iter()
        .find(|t| t.name == "starts")
        .unwrap();
    assert_eq!(start.data.as_i64().unwrap(), &[0]);
    assert!(loaded.graph().tensors.iter().all(|t| t.name != "starts"));
}

#[test]
fn absent_optional_input_retains_its_position() {
    let bytes = model(vec![
        node(
            "conv",
            "Conv",
            &["x", "w", ""],
            "y",
            vec![ints("kernel_shape", &[1, 1])],
        ),
        tensor("w", &[1, 1, 1, 1], &[1.0]),
    ]);
    let loaded = Model::load_bytes(&bytes).unwrap();
    assert_eq!(loaded.graph().nodes[0].inputs.len(), 3);
    assert!(loaded.graph().nodes[0].inputs[2].is_none());
}

#[test]
fn same_upper_even_kernel_padding_is_normalized_and_overflow_is_rejected() {
    let auto = [
        string(1, "auto_pad"),
        number(20, 3),
        string(4, "SAME_UPPER"),
    ]
    .concat();
    let bytes = model(vec![node(
        "pool",
        "MaxPool",
        &["x"],
        "y",
        vec![ints("kernel_shape", &[2, 2]), auto.clone()],
    )]);
    let loaded = Model::load_bytes(&bytes).unwrap();
    let simd_paddle_ocr_rs::model::EffectiveAttributes::Pool(attrs) =
        &loaded.graph().nodes[0].attributes
    else {
        panic!("pool expected")
    };
    assert_eq!(attrs.pads, [0, 0, 1, 1]);
    let overflow = model(vec![
        node(
            "bad",
            "Conv",
            &["x", "w"],
            "y",
            vec![ints("kernel_shape", &[i64::MIN, 1]), auto],
        ),
        tensor("w", &[1, 1, 1, 1], &[1.0]),
    ]);
    assert!(
        Model::load_bytes(&overflow)
            .unwrap_err()
            .to_string()
            .contains("overflow")
    );
}

#[test]
fn tensor_valued_constant_becomes_owned_convolution_weight() {
    let tensor_payload = [
        number(1, 1),
        number(1, 1),
        number(1, 1),
        number(1, 1),
        number(2, 1),
        string(8, "w"),
        bytes(9, 2.0f32.to_bits().to_le_bytes().to_vec()),
    ]
    .concat();
    let value = [string(1, "value"), number(20, 4), bytes(5, tensor_payload)].concat();
    let bytes = model(vec![
        node("weights", "Constant", &[], "w", vec![value]),
        node(
            "conv",
            "Conv",
            &["x", "w"],
            "y",
            vec![ints("kernel_shape", &[1, 1])],
        ),
    ]);
    let loaded = Model::load_bytes(&bytes).unwrap();
    assert_eq!(loaded.graph().nodes.len(), 1);
    let w = loaded
        .graph()
        .tensors
        .iter()
        .find(|t| t.name == "w")
        .unwrap();
    assert_eq!(w.constant.as_ref().unwrap().as_f32().unwrap(), &[2.0]);
}

#[test]
fn scalar_and_integer_list_constant_forms_preserve_payloads() {
    let scalar = float_attr("value_float", -0.0);
    let bytes = model(vec![
        node("constant", "Constant", &[], "one", vec![scalar]),
        node("add", "Add", &["x", "one"], "y", vec![]),
    ]);
    let loaded = Model::load_bytes(&bytes).unwrap();
    let value = loaded
        .graph()
        .tensors
        .iter()
        .find(|t| t.name == "one")
        .unwrap()
        .constant
        .as_ref()
        .unwrap();
    assert!(value.dimensions().is_empty());
    assert_eq!(value.as_f32().unwrap()[0].to_bits(), (-0.0f32).to_bits());
    let list = model(vec![
        node(
            "shape",
            "Constant",
            &[],
            "target",
            vec![ints("value_ints", &[-1])],
        ),
        node("reshape", "Reshape", &["x", "target"], "y", vec![]),
    ]);
    let loaded = Model::load_bytes(&list).unwrap();
    let target = loaded
        .graph()
        .tensors
        .iter()
        .find(|t| t.name == "target")
        .unwrap()
        .constant
        .as_ref()
        .unwrap();
    assert_eq!(target.as_i64().unwrap(), &[-1]);
    assert_eq!(loaded.graph().nodes[0].inputs.len(), 2);
}

#[test]
fn batch_normalization_preserves_a_convolution_with_another_consumer() {
    let bytes = model(vec![
        node(
            "conv",
            "Conv",
            &["x", "w"],
            "conv_out",
            vec![ints("kernel_shape", &[1, 1])],
        ),
        node(
            "bn",
            "BatchNormalization",
            &["conv_out", "gamma", "beta", "mean", "var"],
            "normalized",
            vec![],
        ),
        node("residual", "Add", &["conv_out", "normalized"], "y", vec![]),
        tensor("w", &[1, 1, 1, 1], &[2.0]),
        tensor("gamma", &[1], &[2.0]),
        tensor("beta", &[1], &[0.5]),
        tensor("mean", &[1], &[1.0]),
        tensor("var", &[1], &[3.0]),
    ]);
    let imported = Model::load_bytes(&bytes).unwrap();
    assert_eq!(
        imported
            .graph()
            .nodes
            .iter()
            .map(|n| n.op_type.as_str())
            .collect::<Vec<_>>(),
        ["Conv", "BatchNormalization", "Add"]
    );
    assert!(
        imported
            .graph()
            .tensors
            .iter()
            .all(|t| !t.key.starts_with("generated:bn_"))
    );
    let weight = imported
        .graph()
        .tensors
        .iter()
        .find(|t| t.name == "w")
        .unwrap();
    assert_eq!(weight.constant.as_ref().unwrap().as_f32().unwrap(), &[2.0]);
}

#[test]
fn a_noncentered_branch_with_broadcast_output_keeps_its_original_convolutions() {
    // On [1,1,3,3], the valid 3x3 produces [1,1,1,1], while 1x1 produces
    // [1,1,3,3]. Their Add is valid by broadcasting, but no centered sum applies.
    let bytes = model_with_shapes(
        vec![
            node(
                "valid",
                "Conv",
                &["x", "large"],
                "valid_out",
                vec![ints("kernel_shape", &[3, 3]), ints("pads", &[0, 0, 0, 0])],
            ),
            node(
                "pointwise",
                "Conv",
                &["x", "small"],
                "pointwise_out",
                vec![ints("kernel_shape", &[1, 1])],
            ),
            node("sum", "Add", &["valid_out", "pointwise_out"], "y", vec![]),
            tensor("large", &[1, 1, 3, 3], &[1.0; 9]),
            tensor("small", &[1, 1, 1, 1], &[2.0]),
        ],
        13,
        &[1, 1, 3, 3],
        &[1, 1, 3, 3],
    );
    let imported = Model::load_bytes(&bytes).unwrap();
    assert_eq!(imported.graph().nodes.len(), 3);
    assert!(
        imported
            .graph()
            .tensors
            .iter()
            .all(|t| !t.key.starts_with("generated:branch_"))
    );
    assert!(imported.graph().tensors.iter().any(|t|t.name=="large"&&t.constant.as_ref().unwrap().dimensions()==[1,1,3,3]));
}

#[test]
fn duplicate_producers_are_rejected_before_normalization() {
    let bytes = model(vec![
        node("first", "Relu", &["x"], "shared", vec![]),
        node("second", "Sigmoid", &["x"], "shared", vec![]),
        node("out", "Add", &["shared", "x"], "y", vec![]),
    ]);
    assert!(
        Model::load_bytes(&bytes)
            .unwrap_err()
            .to_string()
            .contains("duplicate tensor producer")
    );
}

struct ImportFixture {
    root: PathBuf,
}
impl ImportFixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let parent = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/import-fixtures");
        fs::create_dir_all(&parent).unwrap();
        let root = parent.join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        fs::create_dir(root.join("models")).unwrap();
        fs::create_dir(root.join("contracts")).unwrap();
        fs::create_dir(root.join("weights")).unwrap();
        let source = model(vec![
            node("add", "Add", &["x", "one"], "data", vec![]),
            node("shape", "Shape", &["data"], "shape", vec![]),
            node(
                "slice",
                "Slice",
                &["shape", "starts", "ends"],
                "target",
                vec![],
            ),
            node("alias", "Identity", &["target"], "target_alias", vec![]),
            node("reshape", "Reshape", &["data", "target_alias"], "y", vec![]),
            tensor("one", &[], &[1.0]),
            int_tensor("starts", &[1], &[0]),
            int_tensor("ends", &[1], &[4]),
        ]);
        fs::write(root.join("models/fixture.onnx"), &source).unwrap();
        let model = Model::load_bytes_with_options(
            &source,
            &LoadOptions {
                capture_raw_snapshot: true,
                ..Default::default()
            },
        )
        .unwrap();
        let raw = serde_json::to_value(model.raw_snapshot().unwrap()).unwrap();
        let mut normalized = serde_json::to_value(model.normalized_snapshot().unwrap()).unwrap();
        for t in normalized["tensors"].as_array_mut().unwrap() {
            if t["constant"] == true {
                t["data_path"] = json!("weights/one.bin");
            }
        }
        fs::write(root.join("weights/one.bin"), 1.0f32.to_bits().to_le_bytes()).unwrap();
        fs::write(
            root.join("contracts/raw.json"),
            serde_json::to_vec_pretty(&raw).unwrap(),
        )
        .unwrap();
        fs::write(
            root.join("contracts/normalized.json"),
            serde_json::to_vec_pretty(&normalized).unwrap(),
        )
        .unwrap();
        let record = |id: &str, path: &str| file_record(&root, id, path);
        let manifest = json!({"schema_version":1,"kind":"model_import_contracts","models":[record("fixture","models/fixture.onnx")],"dictionaries":[],"contracts":[{"model_id":"fixture","raw":record("raw:fixture","contracts/raw.json"),"normalized":record("normalized:fixture","contracts/normalized.json")}],"auxiliary_files":[record("weight:fixture:one","weights/one.bin")]});
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        Self { root }
    }
    fn mutate(&self, stage: &str, edit: impl FnOnce(&mut Value)) {
        let path = format!("contracts/{stage}.json");
        let mut snapshot: Value =
            serde_json::from_slice(&fs::read(self.root.join(&path)).unwrap()).unwrap();
        edit(&mut snapshot);
        fs::write(
            self.root.join(&path),
            serde_json::to_vec_pretty(&snapshot).unwrap(),
        )
        .unwrap();
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(self.root.join("manifest.json")).unwrap()).unwrap();
        let id = manifest["contracts"][0][stage]["id"]
            .as_str()
            .unwrap()
            .to_string();
        manifest["contracts"][0][stage] = file_record(&self.root, &id, &path);
        fs::write(
            self.root.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }
}
impl Drop for ImportFixture {
    fn drop(&mut self) {
        let parent = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/import-fixtures")
            .canonicalize()
            .unwrap();
        let resolved = self.root.canonicalize().unwrap();
        assert!(resolved.starts_with(parent));
        let _ = fs::remove_dir_all(resolved);
    }
}
fn file_record(root: &Path, id: &str, path: &str) -> Value {
    let bytes = fs::read(root.join(path)).unwrap();
    json!({"id":id,"path":path,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))})
}

#[test]
fn rehashed_contracts_must_keep_ir_versions_and_opsets_consistent() {
    let valid = ImportFixture::new();
    assert!(validate_import_reference(&valid.root).is_ok());
    for field in ["ir_version", "opset"] {
        let fixture = ImportFixture::new();
        fixture.mutate("normalized", |s| {
            if field == "ir_version" {
                s["ir_version"] = json!(99);
            } else {
                s["opsets"][0]["version"] = json!(7);
            }
        });
        assert!(
            validate_import_reference(&fixture.root)
                .unwrap_err()
                .to_string()
                .contains("IR version or opset")
        );
    }
}

#[test]
fn rehashed_recipe_bindings_and_templates_must_match_graph_nodes() {
    for field in ["data_input", "shape_input", "template"] {
        let fixture = ImportFixture::new();
        fixture.mutate("normalized", |s| {
            if field == "template" {
                s["recipes"][0]["template"]["dims"][0]["value"] = json!(2);
            } else {
                s["recipes"][0][field] = json!("original:missing");
            }
        });
        let message = validate_import_reference(&fixture.root)
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("binding") || message.contains("template"),
            "{message}"
        );
    }
}

#[test]
fn rehashed_shape_metadata_operands_must_match_raw_integer_bits() {
    let fixture = ImportFixture::new();
    fixture.mutate("normalized", |s| {
        s["shape_metadata"]["constants"][0]["values"][0] = json!(99)
    });
    assert!(
        validate_import_reference(&fixture.root)
            .unwrap_err()
            .to_string()
            .contains("differs from raw initializer")
    );
}
