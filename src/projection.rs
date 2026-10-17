//! Shared validation of the public classifier and recognizer projection heads.
use crate::{
    image::{ImageError, ImageResult},
    model::{Graph, Operator, TensorId},
    tensor::TensorData,
};

pub(crate) struct ProjectionHead {
    pub(crate) feature: TensorId,
    pub(crate) feature_node: usize,
    pub(crate) weights: TensorId,
    pub(crate) bias: Option<TensorId>,
    pub(crate) inner: usize,
    pub(crate) classes: usize,
}
impl ProjectionHead {
    pub(crate) fn compile(
        graph: &Graph,
        classes: usize,
        rank: usize,
        stage: &str,
    ) -> ImageResult<Self> {
        let fail = || {
            ImageError::new(format!(
                "{stage} requires a MatMul/optional bias/last-axis Softmax head with {classes} classes"
            ))
        };
        let softmax = graph.nodes.last().ok_or_else(fail)?;
        if softmax.inputs.len() != 1
            || softmax.outputs.as_slice() != [graph.outputs.first().copied()]
            || !matches!(softmax.operator, Operator::Softmax(axis) if axis == -1 || axis == rank as i64 - 1)
        {
            return Err(fail());
        }
        let mut index = graph.nodes.len().checked_sub(2).ok_or_else(fail)?;
        let mut target = softmax.inputs[0].ok_or_else(fail)?;
        let mut bias = None;
        if matches!(graph.nodes[index].operator, Operator::Add) {
            let add = &graph.nodes[index];
            if add.inputs.len() != 2 || add.outputs.as_slice() != [Some(target)] {
                return Err(fail());
            }
            index = index.checked_sub(1).ok_or_else(fail)?;
            let output = graph.nodes[index]
                .outputs
                .first()
                .copied()
                .flatten()
                .ok_or_else(fail)?;
            let id = if add.inputs[0] == Some(output) {
                add.inputs[1]
            } else if add.inputs[1] == Some(output) {
                add.inputs[0]
            } else {
                return Err(fail());
            }
            .ok_or_else(fail)?;
            match &graph.tensors[id.0].constant {
                Some(TensorData::F32(t)) if t.shape().dimensions() == [classes] => {}
                _ => return Err(fail()),
            }
            bias = Some(id);
            target = output;
        }
        let matmul = &graph.nodes[index];
        if !matches!(matmul.operator, Operator::MatMul)
            || matmul.inputs.len() != 2
            || matmul.outputs.as_slice() != [Some(target)]
        {
            return Err(fail());
        }
        let feature = matmul.inputs[0].ok_or_else(fail)?;
        let weights = matmul.inputs[1].ok_or_else(fail)?;
        let inner = match &graph.tensors[weights.0].constant {
            Some(TensorData::F32(t))
                if t.shape().rank() == 2 && t.shape().dimensions()[1] == classes =>
            {
                t.shape().dimensions()[0]
            }
            _ => return Err(fail()),
        };
        if inner == 0 {
            return Err(fail());
        }
        let feature_node = graph.nodes[..index]
            .iter()
            .position(|n| n.outputs.contains(&Some(feature)))
            .ok_or_else(fail)?;
        Ok(Self {
            feature,
            feature_node,
            weights,
            bias,
            inner,
            classes,
        })
    }
}
