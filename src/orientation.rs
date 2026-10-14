//! Native public orientation classification, distinct from dense graph output.
use crate::{
    image::{ImageError, ImageResult, ImageView},
    math::exp_f32,
    model::{EffectiveAttributes, Graph, TensorId},
    preprocess::{self, ResizeWorkspace},
    runtime::{CompiledModel, ExecutionLimits, NodeTrace, RuntimeError, Session},
    tensor::TensorData,
};
use serde::Serialize;
use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct ClassificationResult {
    pub label: u32,
    pub score: f32,
    pub resized_width: usize,
}
impl ClassificationResult {
    pub fn orientation_degrees(self) -> u16 {
        if self.label == 0 { 0 } else { 180 }
    }
}

pub(crate) enum ClassifierTrace<'a> {
    Input {
        values: &'a [f32],
        resized_width: usize,
    },
    Node(NodeTrace<'a>),
    Probabilities(&'a [f32]),
    Public(ClassificationResult),
}

struct PublicHead {
    feature: TensorId,
    feature_node: usize,
    weights: TensorId,
    bias: Option<TensorId>,
    inner: usize,
}
impl PublicHead {
    fn compile(graph: &Graph) -> ImageResult<Self> {
        let fail = || {
            ImageError::new(
                "classifier requires the shipped two-class MatMul/optional bias/Softmax head",
            )
        };
        let softmax = graph.nodes.last().ok_or_else(fail)?;
        if softmax.op_type != "Softmax"
            || softmax.inputs.len() != 1
            || softmax.outputs.as_slice() != [graph.outputs.first().copied()]
        {
            return Err(fail());
        }
        if !matches!(softmax.attributes, EffectiveAttributes::Axis(-1 | 1)) {
            return Err(fail());
        }
        let mut head_index = graph.nodes.len().checked_sub(2).ok_or_else(fail)?;
        let mut target = softmax.inputs[0].ok_or_else(fail)?;
        let mut bias = None;
        if graph.nodes[head_index].op_type == "Add" {
            let add = &graph.nodes[head_index];
            if add.inputs.len() != 2 || add.outputs.as_slice() != [Some(target)] {
                return Err(fail());
            }
            head_index = head_index.checked_sub(1).ok_or_else(fail)?;
            let output = graph.nodes[head_index]
                .outputs
                .first()
                .copied()
                .flatten()
                .ok_or_else(fail)?;
            let other = if add.inputs[0] == Some(output) {
                add.inputs[1]
            } else if add.inputs[1] == Some(output) {
                add.inputs[0]
            } else {
                return Err(fail());
            };
            let id = other.ok_or_else(fail)?;
            match &graph.tensors[id.0].constant {
                Some(TensorData::F32(t)) if t.shape().dimensions() == [2] => {}
                _ => return Err(fail()),
            }
            bias = Some(id);
            target = output;
        }
        let matmul = &graph.nodes[head_index];
        if matmul.op_type != "MatMul"
            || matmul.inputs.len() != 2
            || matmul.outputs.as_slice() != [Some(target)]
        {
            return Err(fail());
        }
        let feature = matmul.inputs[0].ok_or_else(fail)?;
        let weights = matmul.inputs[1].ok_or_else(fail)?;
        let inner = match &graph.tensors[weights.0].constant {
            Some(TensorData::F32(t)) if t.shape().rank() == 2 && t.shape().dimensions()[1] == 2 => {
                t.shape().dimensions()[0]
            }
            _ => return Err(fail()),
        };
        let feature_node = graph.nodes[..head_index]
            .iter()
            .position(|node| node.outputs.contains(&Some(feature)))
            .ok_or_else(fail)?;
        Ok(Self {
            feature,
            feature_node,
            weights,
            bias,
            inner,
        })
    }
    fn evaluate(
        &self,
        graph: &Graph,
        feature: &[f32],
        resized_width: usize,
    ) -> ImageResult<ClassificationResult> {
        if feature.len() != self.inner {
            return Err(ImageError::new("classifier feature must contain one row"));
        }
        let weights = match &graph.tensors[self.weights.0].constant {
            Some(TensorData::F32(t)) => t.data(),
            _ => unreachable!(),
        };
        let bias = self.bias.map(|id| match &graph.tensors[id.0].constant {
            Some(TensorData::F32(t)) => t.data(),
            _ => unreachable!(),
        });
        public_head(feature, weights, bias, resized_width)
    }
}

fn public_head(
    feature: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    resized_width: usize,
) -> ImageResult<ClassificationResult> {
    if weights.len()
        != feature
            .len()
            .checked_mul(2)
            .ok_or_else(|| ImageError::new("classifier head size overflow"))?
        || bias.is_some_and(|b| b.len() != 2)
    {
        return Err(ImageError::new("invalid classifier head dimensions"));
    }
    let mut logits = [0f32; 2];
    for column in 0..2 {
        let mut sum = bias.map_or(0., |b| b[column]);
        for (index, &value) in feature.iter().enumerate() {
            sum += value * weights[index * 2 + column];
        }
        logits[column] = sum;
    }
    if logits.iter().any(|v| !v.is_finite()) {
        return Err(ImageError::new("nonfinite public classifier logits"));
    }
    let label = u32::from(logits[1] > logits[0]);
    let maximum = logits[0].max(logits[1]);
    let e0 = exp_f32(logits[0] - maximum);
    let e1 = exp_f32(logits[1] - maximum);
    let mut sum = 0f32;
    sum += e0;
    sum += e1;
    let score = if label == 0 { e0 } else { e1 } / sum;
    Ok(ClassificationResult {
        label,
        score,
        resized_width,
    })
}

pub(crate) struct Classifier {
    compiled: Arc<CompiledModel>,
    session: Session,
    head: PublicHead,
    resize: ResizeWorkspace,
}
impl Classifier {
    pub(crate) fn new(compiled: Arc<CompiledModel>, limits: ExecutionLimits) -> ImageResult<Self> {
        let head = PublicHead::compile(compiled.model().graph())?;
        let session = Session::new(compiled.clone(), &[1, 3, 80, 160], limits)?;
        if session.output_shape().dimensions() != [1, 2] {
            return Err(ImageError::new("classifier output must be [1,2]"));
        }
        Ok(Self {
            compiled,
            session,
            head,
            resize: ResizeWorkspace::default(),
        })
    }
    pub(crate) fn classify(&mut self, image: ImageView<'_>) -> ImageResult<ClassificationResult> {
        let mut width = 0;
        let resize = &mut self.resize;
        let feature = self
            .session
            .run_prefix_with_input(self.head.feature_node, |input| {
                width = preprocess::cls(image, input, resize)
                    .map_err(|e| RuntimeError::new(e.to_string()))?;
                Ok(())
            })?;
        self.head
            .evaluate(self.compiled.model().graph(), feature.data(), width)
    }
    pub(crate) fn classify_with_trace<F>(
        &mut self,
        image: ImageView<'_>,
        observer: F,
    ) -> ImageResult<ClassificationResult>
    where
        F: for<'a> FnMut(ClassifierTrace<'a>) -> ImageResult<()>,
    {
        let observer = RefCell::new(observer);
        let width = Cell::new(0);
        let public = Cell::new(None);
        let resize = &mut self.resize;
        let head = &self.head;
        let graph = self.compiled.model().graph();
        let probabilities = self.session.run_with_input_and_trace(
            |input| {
                let actual = preprocess::cls(image, input, resize)
                    .map_err(|e| RuntimeError::new(e.to_string()))?;
                width.set(actual);
                observer.borrow_mut()(ClassifierTrace::Input {
                    values: input,
                    resized_width: actual,
                })
                .map_err(|e| RuntimeError::new(e.to_string()))?;
                Ok(())
            },
            |node| {
                if node.output_id == head.feature {
                    public.set(Some(
                        head.evaluate(graph, node.values, width.get())
                            .map_err(|e| RuntimeError::new(e.to_string()))?,
                    ));
                }
                observer.borrow_mut()(ClassifierTrace::Node(node))
                    .map_err(|e| RuntimeError::new(e.to_string()))
            },
        )?;
        observer.borrow_mut()(ClassifierTrace::Probabilities(probabilities.data()))?;
        let result = public
            .get()
            .ok_or_else(|| ImageError::new("classifier feature was not observed"))?;
        observer.borrow_mut()(ClassifierTrace::Public(result))?;
        Ok(result)
    }
    pub(crate) fn activation_bytes(&self) -> usize {
        self.session.activation_bytes()
    }
    pub(crate) fn retained_activation_bytes(&self) -> usize {
        self.session.retained_activation_bytes()
    }
    pub(crate) fn retained_workspace_bytes(&self) -> usize {
        self.resize.retained_bytes()
    }
}

pub fn should_rotate(result: ClassificationResult, threshold: f32) -> bool {
    result.label == 1 && result.score > threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_head_preserves_bias_order_and_logit_winner() {
        let r = public_head(&[1., 1.], &[1e8, 0., -1e8, 0.], Some(&[1., 0.5]), 160).unwrap();
        assert_eq!(r.label, 1);
        let r = public_head(&[1.], &[0., 1e-8], None, 80).unwrap();
        assert_eq!(r.label, 1);
        assert_eq!(r.score, 0.5);
        assert!(!should_rotate(r, 0.5));
        assert!(should_rotate(r, f32::from_bits(0.5f32.to_bits() - 1)));
        assert_eq!(public_head(&[1.], &[1., 1.], None, 80).unwrap().label, 0);
    }
    #[test]
    fn malformed_or_nonfinite_heads_fail() {
        assert!(public_head(&[1.], &[1.], None, 160).is_err());
        assert!(public_head(&[1.], &[1., 1.], Some(&[]), 160).is_err());
        assert!(public_head(&[f32::INFINITY], &[1., 1.], None, 160).is_err());
    }
}
