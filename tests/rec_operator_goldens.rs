//! Independent native C# operator captures for the tiny REC runtime subset.
use serde_json::Value;
use simd_paddle_ocr_rs::{
    model::Model,
    runtime::{CompiledModel, ExecutionLimits, Session},
    validate_bundle,
};
use std::{fs, path::Path, sync::Arc};

fn artifact<'a>(case: &'a Value, id: &str) -> &'a Value {
    case["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|artifact| artifact["id"] == id)
        .unwrap()
}
fn floats(root: &Path, artifact: &Value) -> Vec<f32> {
    let bytes = fs::read(root.join(artifact["path"].as_str().unwrap())).unwrap();
    assert_eq!(bytes.len() as u64, artifact["bytes"].as_u64().unwrap());
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}
fn shape(artifact: &Value) -> Vec<usize> {
    artifact["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect()
}
fn compare(actual: &[f32], expected: &[f32], exact: bool, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{context}: {index}"
        );
        if exact {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{context}: {index}");
        } else {
            assert!(
                (f64::from(actual) - f64::from(expected)).abs()
                    <= 1e-5 + 1e-4 * f64::from(expected).abs(),
                "{context}: {index}, native={expected}, rust={actual}"
            );
        }
    }
}

#[test]
fn tiny_rec_subset_matches_independent_native_nodes_and_outputs() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recognition-v1/operators");
    let validation =
        validate_bundle(&root).expect("committed independent native operator captures");
    assert_eq!(validation.checked_cases, 8);
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let mut checked_nodes = 0;
    for case in manifest["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let metadata: Value = serde_json::from_slice(
            &fs::read(root.join(artifact(case, "fixture")["path"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let exact = metadata["exact"].as_bool().unwrap();
        let model = Arc::new(
            Model::load_path(root.join(artifact(case, "model")["path"].as_str().unwrap())).unwrap(),
        );
        let compiled = Arc::new(CompiledModel::new(model).unwrap());
        let input = artifact(case, "input");
        let values = floats(&root, input);
        let mut session =
            Session::new(compiled.clone(), &shape(input), ExecutionLimits::default()).unwrap();
        let mut observed = 0;
        let output = session
            .run_with_trace(&values, |node| {
                let saved = artifact(case, &format!("node.{:04}", node.index));
                assert_eq!(node.shape.dimensions(), shape(saved), "{id}: node shape");
                assert_eq!(
                    node.layout,
                    saved["layout"].as_str().unwrap(),
                    "{id}: node layout"
                );
                compare(node.values, &floats(&root, saved), exact, id);
                observed += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(
            observed,
            compiled.node_count(),
            "{id}: complete node coverage"
        );
        let saved = artifact(case, "output");
        assert_eq!(
            output.shape().dimensions(),
            shape(saved),
            "{id}: output shape"
        );
        compare(output.data(), &floats(&root, saved), exact, id);
        checked_nodes += observed;
    }
    assert_eq!(checked_nodes, 8);
}
