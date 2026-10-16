//! Compact cross-language import contracts; no inference or source patching.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{
    model::Model,
    onnx::{self, AttributeValue, LoadLimits, ValueInfo},
    tensor::{Dim, ShapeSpec, TensorData},
};
use std::{collections::BTreeMap, fs, io::BufReader, path::Path};

fn shape(shape: &ShapeSpec, normalized: bool) -> Vec<i64> {
    match shape {
        ShapeSpec::Ranked(dims) => dims
            .iter()
            .map(|dim| match dim {
                Dim::Known(value) => *value as i64,
                _ => -1,
            })
            .collect(),
        ShapeSpec::UnknownRank => {
            if normalized {
                vec![-1]
            } else {
                vec![]
            }
        }
    }
}

fn tensor(data: &TensorData) -> Value {
    let mut hash = Sha256::new();
    match data {
        TensorData::F32(t) => {
            for value in t.data() {
                hash.update(value.to_le_bytes());
            }
        }
        TensorData::I32(t) => {
            for value in t.data() {
                hash.update(value.to_le_bytes());
            }
        }
        TensorData::I64(t) => {
            for value in t.data() {
                hash.update(value.to_le_bytes());
            }
        }
        TensorData::U8(t) => hash.update(t.data()),
    }
    json!({"dtype":data.dtype(),"shape":data.dimensions(),
        "data_bytes":data.len()*data.dtype().element_bytes(),"data_sha256":format!("{:x}",hash.finalize())})
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn attribute(value: &AttributeValue) -> Value {
    match value {
        AttributeValue::Float(value) => json!({"kind":"float","bits":value.to_bits()}),
        AttributeValue::Int(value) => json!({"kind":"int","value":value}),
        AttributeValue::String(value) => json!({"kind":"bytes","hex":hex(value)}),
        AttributeValue::Tensor(value) => json!({"kind":"tensor","value":tensor(&value.data)}),
        AttributeValue::Floats(values) => {
            json!({"kind":"floats","bits":values.iter().map(|value|value.to_bits()).collect::<Vec<_>>()})
        }
        AttributeValue::Ints(values) => json!({"kind":"ints","values":values}),
        AttributeValue::Strings(values) => {
            json!({"kind":"strings","hex":values.iter().map(|value|hex(value)).collect::<Vec<_>>()})
        }
    }
}

fn value(info: &ValueInfo) -> Value {
    json!({"name":info.name,"dtype":info.dtype,"shape":shape(&info.shape,false)})
}

pub fn write(path: &Path, output: &Path, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = onnx::parse(
        BufReader::new(fs::File::open(path)?),
        &LoadLimits::default(),
    )?;
    let mut initializers: Vec<_> = parsed.graph.initializers.iter().collect();
    initializers.sort_by_key(|value| &value.name);
    let mut value_info: Vec<_> = parsed.graph.value_info.iter().collect();
    value_info.sort_by_key(|value| &value.name);
    let raw = json!({"ir_version":parsed.ir_version,
        "opsets":parsed.opsets.iter().map(|(domain,version)|json!({"domain":domain,"version":version})).collect::<Vec<_>>(),
        "inputs":parsed.graph.inputs.iter().map(value).collect::<Vec<_>>(),
        "outputs":parsed.graph.outputs.iter().map(value).collect::<Vec<_>>(),
        "value_info":value_info.into_iter().map(value).collect::<Vec<_>>(),
        "initializers":initializers.iter().map(|value|json!({"name":value.name,"tensor":tensor(&value.data)})).collect::<Vec<_>>(),
        "nodes":parsed.graph.nodes.iter().map(|node|json!({"name":node.name,"op_type":node.op_type,"domain":node.domain,
            "inputs":node.inputs,"outputs":node.outputs,"attributes":node.attributes.iter().map(|value|(value.name.clone(),attribute(&value.value))).collect::<BTreeMap<_,_>>()})).collect::<Vec<_>>()});
    drop(parsed);
    let model = Model::load_path(path)?;
    let snapshot = model.normalized_snapshot()?;
    let graph = model.graph();
    let mut producers = BTreeMap::new();
    for (index, node) in graph.nodes.iter().enumerate() {
        for (slot, output) in node.outputs.iter().enumerate() {
            if let Some(output) = output {
                producers.insert(output.0, (index, slot));
            }
        }
    }
    let edge = |id: Option<simd_paddle_ocr_rs::model::TensorId>| -> Value {
        let Some(id) = id else {
            return Value::Null;
        };
        let descriptor = &graph.tensors[id.0];
        let origin = if let Some(data) = &descriptor.constant {
            let data = tensor(data);
            json!({"kind":"constant","data_bytes":data["data_bytes"],"data_sha256":data["data_sha256"]})
        } else if let Some(&(index, slot)) = producers.get(&id.0) {
            json!({"kind":"node","index":index,"slot":slot})
        } else {
            let index = graph
                .inputs
                .iter()
                .position(|input| *input == id)
                .expect("bound graph input");
            json!({"kind":"input","index":index})
        };
        json!({"dtype":descriptor.dtype,"shape":shape(&descriptor.shape,true),"origin":origin})
    };
    let normalized = json!({"inputs":graph.inputs.iter().map(|&id|edge(Some(id))).collect::<Vec<_>>(),
        "outputs":graph.outputs.iter().map(|&id|edge(Some(id))).collect::<Vec<_>>(),
        "nodes":graph.nodes.iter().zip(&snapshot.nodes).map(|(node,inspection)| {
            let mut attributes = serde_json::to_value(&inspection.attributes).expect("serializable attributes");
            // The pinned C# parameter blocks retain only scales for Resize;
            // Reshape and Slice have no parameter block. Raw contracts retain
            // their source attributes and execution parity checks behavior.
            if node.op_type == "Resize" { attributes.as_object_mut().unwrap().retain(|key,_| key == "scales"); }
            if node.op_type == "Reshape" || node.op_type == "Slice" { attributes = json!({}); }
            json!({"name":node.name,"op_type":node.op_type,"domain":node.domain,"opset":node.opset,"attributes":attributes,
            "inputs":node.inputs.iter().map(|&id|edge(id)).collect::<Vec<_>>(),
            "outputs":node.outputs.iter().map(|&id|edge(id)).collect::<Vec<_>>()}) }).collect::<Vec<_>>()});
    fs::create_dir_all(output.join("imports"))?;
    fs::write(
        output.join("imports").join(format!("{name}.json")),
        serde_json::to_vec_pretty(&json!({"schema_version":1,"raw":raw,"normalized":normalized}))?,
    )?;
    println!("import {name}: raw and normalized");
    Ok(())
}
