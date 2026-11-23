//! The FP32 schedule shared by the supported B-Scale exports.
use crate::{
    error::{ErrorKind, ImportError, RuntimeError},
    onnx::{AttributeValue, RawGraph},
    runtime::RuntimeResult,
    tensor::TensorData,
};
use std::collections::HashMap;

mod cpu;
pub(super) use cpu::{CpuScorer, CpuWeights};

pub(crate) const NODE_INPUT: usize = 3;
pub(crate) const EDGE_INPUT: usize = 26;
pub(crate) const NODE_WIDTH: usize = 96;
pub(crate) const EDGE_WIDTH: usize = 144;
pub(crate) const RELATION_WIDTH: usize = 2 * NODE_WIDTH + EDGE_WIDTH;
pub(crate) const EPSILON: f32 = 1e-5;

pub(crate) struct Input<'a> {
    pub(crate) nodes: &'a [f32],
    pub(crate) edges: &'a [f32],
    pub(crate) pairs: &'a [[usize; 2]],
}

impl Input<'_> {
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len() / NODE_INPUT
    }

    pub(super) fn validate(&self) -> RuntimeResult<()> {
        let nodes = self.node_count();
        if !self.nodes.len().is_multiple_of(NODE_INPUT)
            || self.pairs.len().checked_mul(2 * EDGE_INPUT) != Some(self.edges.len())
            || self
                .pairs
                .iter()
                .any(|&[a, b]| a == b || a >= nodes || b >= nodes)
            || self.nodes.iter().chain(self.edges).any(|v| !v.is_finite())
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "invalid B-Scale scorer inputs",
            ));
        }
        Ok(())
    }
}

pub(crate) struct Linear {
    pub(crate) input: usize,
    pub(crate) output: usize,
    /// Output-major rows as stored by the ONNX Gemm coefficients.
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}
pub(crate) struct Norm {
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}
pub(crate) struct Layer {
    pub(crate) edge_update: [Linear; 2],
    pub(crate) edge_norm: Norm,
    pub(crate) attention_mixed: Linear,
    pub(crate) attention_final: Linear,
    pub(crate) message: [Linear; 2],
    pub(crate) update: [Linear; 2],
    pub(crate) node_norm: Norm,
}
pub(crate) struct Weights {
    pub(crate) node: [Linear; 2],
    pub(crate) edge: [Linear; 2],
    pub(crate) layers: [Layer; 4],
    pub(crate) classifier: [Linear; 2],
}

impl Weights {
    pub(super) fn load(graph: RawGraph) -> Result<Self, ImportError> {
        if graph.initializers.len() != 88 {
            return Err(ImportError::new("unexpected B-Scale scorer graph"));
        }
        let mut tensors = HashMap::with_capacity(graph.initializers.len());
        for tensor in graph.initializers {
            if tensors.insert(tensor.name, tensor.data).is_some() {
                return Err(ImportError::new("duplicate B-Scale coefficient"));
            }
        }
        let mut loader = Loader {
            tensors,
            operators: graph
                .nodes
                .iter()
                .map(|op| (op.name.as_str(), op))
                .collect(),
        };
        let weights = Self {
            node: loader.mlp("node", NODE_INPUT, NODE_WIDTH, NODE_WIDTH)?,
            edge: loader.mlp("edge_encoder", EDGE_INPUT, EDGE_WIDTH, EDGE_WIDTH)?,
            layers: [
                loader.layer(0)?,
                loader.layer(1)?,
                loader.layer(2)?,
                loader.layer(3)?,
            ],
            classifier: loader.mlp("classifier", RELATION_WIDTH, NODE_WIDTH, 1)?,
        };
        if !loader.tensors.is_empty() {
            return Err(ImportError::new("unused B-Scale coefficients"));
        }
        Ok(weights)
    }
}

struct Loader<'a> {
    tensors: HashMap<String, TensorData>,
    operators: HashMap<&'a str, &'a crate::onnx::RawNode>,
}

impl Loader<'_> {
    fn coefficients(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, ImportError> {
        match self.tensors.remove(name) {
            Some(TensorData::F32(tensor))
                if tensor.shape().dimensions() == shape
                    && tensor.data().iter().all(|v| v.is_finite()) =>
            {
                Ok(tensor.into_vec())
            }
            _ => Err(ImportError::new(format!(
                "invalid B-Scale coefficient {name}"
            ))),
        }
    }

    fn linear(&mut self, name: &str, input: usize, output: usize) -> Result<Linear, ImportError> {
        let weight = format!("{name}.weight");
        let bias = format!("{name}.bias");
        if !self.operators.values().any(|op| {
            op.op_type == "Gemm"
                && op.domain.is_empty()
                && op.inputs.get(1).map(String::as_str) == Some(weight.as_str())
                && op.inputs.get(2).map(String::as_str) == Some(bias.as_str())
                && op
                    .attributes
                    .iter()
                    .any(|a| a.name == "transB" && matches!(a.value, AttributeValue::Int(1)))
        }) {
            return Err(ImportError::new(format!(
                "missing expected B-Scale Gemm for {name}"
            )));
        }
        Ok(Linear {
            input,
            output,
            weight: self.coefficients(&weight, &[output, input])?,
            bias: self.coefficients(&bias, &[output])?,
        })
    }

    fn mlp(
        &mut self,
        name: &str,
        input: usize,
        hidden: usize,
        output: usize,
    ) -> Result<[Linear; 2], ImportError> {
        Ok([
            self.linear(&format!("{name}.0"), input, hidden)?,
            self.linear(&format!("{name}.2"), hidden, output)?,
        ])
    }

    fn norm(&mut self, layer: &str, component: &str, width: usize) -> Result<Norm, ImportError> {
        let weight = format!("{layer}.{component}.weight");
        let bias = format!("{layer}.{component}.bias");
        let name = format!("/{layer}/{component}/LayerNormalization");
        let op = self
            .operators
            .get(name.as_str())
            .ok_or_else(|| ImportError::new(format!("missing B-Scale normalization at {name}")))?;
        if op.op_type != "LayerNormalization"
            || !op.domain.is_empty()
            || op.inputs.get(1).map(String::as_str) != Some(weight.as_str())
            || op.inputs.get(2).map(String::as_str) != Some(bias.as_str())
            || !op
                .attributes
                .iter()
                .any(|a| a.name == "axis" && matches!(a.value, AttributeValue::Int(-1)))
            || !op.attributes.iter().any(|a| {
                a.name == "epsilon" && matches!(a.value, AttributeValue::Float(v) if v == EPSILON)
            })
        {
            return Err(ImportError::new(format!(
                "unexpected B-Scale normalization at {name}"
            )));
        }
        Ok(Norm {
            weight: self.coefficients(&weight, &[width])?,
            bias: self.coefficients(&bias, &[width])?,
        })
    }

    fn layer(&mut self, index: usize) -> Result<Layer, ImportError> {
        let layer = format!("layers.{index}");
        let path = format!("/{layer}/attention_final/MatMul");
        let name = self
            .operators
            .get(path.as_str())
            .filter(|op| op.op_type == "MatMul" && op.domain.is_empty() && op.inputs.len() == 2)
            .ok_or_else(|| {
                ImportError::new(format!("missing B-Scale attention projection at {path}"))
            })?
            .inputs[1]
            .clone();
        Ok(Layer {
            edge_update: self.mlp(
                &format!("{layer}.edge_update"),
                RELATION_WIDTH,
                EDGE_WIDTH,
                EDGE_WIDTH,
            )?,
            edge_norm: self.norm(&layer, "edge_norm", EDGE_WIDTH)?,
            attention_mixed: self.linear(
                &format!("{layer}.attention_mixed"),
                RELATION_WIDTH,
                NODE_WIDTH,
            )?,
            attention_final: Linear {
                input: NODE_WIDTH,
                output: 1,
                weight: self.coefficients(&name, &[NODE_WIDTH, 1])?,
                bias: Vec::new(),
            },
            message: self.mlp(
                &format!("{layer}.message"),
                RELATION_WIDTH,
                NODE_WIDTH,
                NODE_WIDTH,
            )?,
            update: self.mlp(
                &format!("{layer}.update"),
                2 * NODE_WIDTH,
                NODE_WIDTH,
                NODE_WIDTH,
            )?,
            node_norm: self.norm(&layer, "node_norm", NODE_WIDTH)?,
        })
    }
}
