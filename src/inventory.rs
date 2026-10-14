//! Shared semantic identity checks for canonical frozen graph inventories.
use crate::{Bundle, Result, det, invalid, model::Model, read_bounded, safe_file};
use serde::Deserialize;
use serde_json::{Value, json};
use std::fs::File;

#[derive(Deserialize)]
struct Inventory {
    id: String,
    #[serde(default)]
    inputs: Option<Value>,
    #[serde(default)]
    outputs: Option<Value>,
    nodes: Vec<Node>,
}
#[derive(Deserialize)]
struct Node {
    #[serde(default)]
    inputs: Option<Value>,
    #[serde(default)]
    outputs: Option<Value>,
}

pub(crate) fn verify(
    bundle: &Bundle,
    model: &Model,
    id: &str,
    nodes: &[(usize, usize)],
    semantic: bool,
) -> Result<()> {
    if !bundle
        .manifest
        .auxiliary_files
        .iter()
        .any(|file| file.path == "graph-inventory.json")
    {
        return Err(invalid("graph inventory must be an indexed auxiliary file"));
    }
    // This also checks names, domains, opsets, ordinals, all-node coverage and
    // any available semantic edges in older inventories.
    det::verify_inventory_bundle(bundle, model, id, nodes)?;
    if !semantic {
        return Ok(());
    }
    let bytes = read_bounded(
        File::open(safe_file(&bundle.root, "graph-inventory.json")?)
            .map_err(|error| invalid(error.to_string()))?,
        "graph-inventory.json",
    )?;
    let inventories: Vec<Inventory> = serde_json::from_slice(&bytes)
        .map_err(|error| invalid(format!("graph inventory: {error}")))?;
    let selected = inventories
        .iter()
        .filter(|inventory| inventory.id == id)
        .collect::<Vec<_>>();
    if selected.len() != 1 {
        return Err(invalid(format!("{id} inventory identity must be unique")));
    }
    let inventory = selected[0];
    let graph = model.graph();
    let graph_keys = |ids: &[crate::model::TensorId]| {
        json!(
            ids.iter()
                .map(|id| graph.tensors[id.0].key.as_str())
                .collect::<Vec<_>>()
        )
    };
    if inventory.inputs.as_ref() != Some(&graph_keys(&graph.inputs))
        || inventory.outputs.as_ref() != Some(&graph_keys(&graph.outputs))
    {
        return Err(invalid(format!(
            "{id} inventory requires matching semantic graph edges"
        )));
    }
    let node_keys = |ids: &[Option<crate::model::TensorId>]| {
        json!(
            ids.iter()
                .map(|id| id.map(|id| graph.tensors[id.0].key.as_str()))
                .collect::<Vec<_>>()
        )
    };
    for (ordinal, (saved, node)) in inventory.nodes.iter().zip(&graph.nodes).enumerate() {
        if saved.inputs.as_ref() != Some(&node_keys(&node.inputs))
            || saved.outputs.as_ref() != Some(&node_keys(&node.outputs))
        {
            return Err(invalid(format!(
                "{id} inventory requires matching semantic edges at node {ordinal}"
            )));
        }
    }
    Ok(())
}
