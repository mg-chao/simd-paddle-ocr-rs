//! Reusable scalar recognition with distinct dense and compatible public scores.
use crate::{
    image::{ImageError, ImageResult, ImageView, resize_vec},
    model::{EffectiveAttributes, Graph, TensorId},
    preprocess::{self, ResizeWorkspace},
    runtime::{CompiledModel, ExecutionLimits, NodeTrace, RuntimeError, Session},
    tensor::{Shape, TensorData},
};
use serde::Serialize;
use std::{
    cell::{Cell, RefCell},
    mem::size_of,
    sync::Arc,
};

#[derive(Debug)]
pub struct Dictionary {
    text: String,
    labels: Vec<(usize, usize)>,
}
impl Dictionary {
    /// Parse the native UTF-8 label file without trimming label content or
    /// collapsing empty/duplicate tokens. Blank and space are implicit classes.
    pub fn from_utf8(bytes: &[u8]) -> ImageResult<Self> {
        let raw = std::str::from_utf8(bytes)
            .map_err(|_| ImageError::new("dictionary is not valid UTF-8"))?;
        let raw = raw.strip_prefix('\u{feff}').unwrap_or(raw);
        let mut text = String::new();
        text.try_reserve_exact(raw.len()).map_err(allocation)?;
        text.push_str(raw);
        let count = text.split('\n').count() - usize::from(text.ends_with('\n'));
        let mut labels = Vec::new();
        labels.try_reserve_exact(count).map_err(allocation)?;
        let mut offset = 0;
        for token in text.split('\n').take(count) {
            let label = token.strip_suffix('\r').unwrap_or(token);
            labels.push((offset, offset + label.len()));
            offset += token.len() + 1;
        }
        if labels.is_empty() || labels.iter().all(|&(a, b)| a == b) {
            return Err(ImageError::new("dictionary is empty"));
        }
        labels
            .len()
            .checked_add(2)
            .ok_or_else(|| ImageError::new("dictionary class count overflow"))?;
        Ok(Self { text, labels })
    }
    pub fn class_count(&self) -> usize {
        self.labels.len() + 2
    }
    pub fn retained_bytes(&self) -> usize {
        self.text.capacity() + self.labels.capacity() * size_of::<(usize, usize)>()
    }
    /// Blank (zero) and out-of-range classes return None; the final class is space.
    pub fn token(&self, class: usize) -> Option<&str> {
        if class == self.labels.len() + 1 {
            return Some(" ");
        }
        let &(a, b) = self.labels.get(class.checked_sub(1)?)?;
        Some(&self.text[a..b])
    }
}
fn allocation(error: impl std::fmt::Display) -> ImageError {
    ImageError::new(format!("recognition allocation failed: {error}"))
}

#[derive(Clone, Copy, Debug)]
pub struct RecognizerOptions {
    /// Tensor width in fixed mode; adaptive mode selects its own width.
    pub target_width: usize,
    /// Round the natural height-48 content width up to a multiple of 32,
    /// without a 320-pixel upper cap.
    pub adaptive_width: bool,
    pub max_image_pixels: usize,
    /// Active UTF-8 and alignment records; excludes shared dictionary and activations.
    pub max_result_bytes: usize,
}
impl Default for RecognizerOptions {
    fn default() -> Self {
        Self {
            target_width: 320,
            adaptive_width: true,
            max_image_pixels: 40_000_000,
            max_result_bytes: i32::MAX as usize,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecognitionSize {
    pub resized_width: usize,
    pub tensor_width: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CtcRun {
    pub text_start: usize,
    pub text_end: usize,
    pub score: f32,
    pub start_column: usize,
    pub end_column: usize,
}
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct CtcSpanView<'a> {
    pub text: &'a str,
    pub score: f32,
    pub start_column: usize,
    pub end_column: usize,
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OwnedCtcSpan {
    pub text: String,
    pub score: f32,
    pub start_column: usize,
    pub end_column: usize,
}
#[derive(Clone, Copy, Debug)]
enum SpanSource<'a> {
    Runs(&'a str, &'a [CtcRun]),
    Owned(&'a [OwnedCtcSpan]),
}
#[derive(Clone, Copy, Debug)]
pub struct CtcSpans<'a> {
    source: SpanSource<'a>,
}
impl<'a> CtcSpans<'a> {
    pub(crate) fn from_owned(spans: &'a [OwnedCtcSpan]) -> Self {
        Self {
            source: SpanSource::Owned(spans),
        }
    }
    pub(crate) fn from_runs(text: &'a str, runs: &'a [CtcRun]) -> Self {
        Self {
            source: SpanSource::Runs(text, runs),
        }
    }
    pub fn len(self) -> usize {
        match self.source {
            SpanSource::Runs(_, x) => x.len(),
            SpanSource::Owned(x) => x.len(),
        }
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    pub fn get(self, index: usize) -> Option<CtcSpanView<'a>> {
        match self.source {
            SpanSource::Runs(text, runs) => {
                let r = runs.get(index)?;
                Some(CtcSpanView {
                    text: &text[r.text_start..r.text_end],
                    score: r.score,
                    start_column: r.start_column,
                    end_column: r.end_column,
                })
            }
            SpanSource::Owned(spans) => {
                let s = spans.get(index)?;
                Some(CtcSpanView {
                    text: &s.text,
                    score: s.score,
                    start_column: s.start_column,
                    end_column: s.end_column,
                })
            }
        }
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = CtcSpanView<'a>> {
        (0..self.len()).map(move |i| self.get(i).expect("validated span index"))
    }
}
impl Serialize for CtcSpans<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RecognitionView<'a> {
    pub text: &'a str,
    /// Mean score at each token's first frame. Public compact scores are
    /// uncalibrated projection maxima; diagnostic dense scores are probabilities.
    pub score: f32,
    /// Emitted dictionary tokens, independent of Unicode character count.
    pub emitted_count: usize,
    pub resized_width: usize,
    /// Width including normalized-zero right padding.
    pub tensor_width: usize,
    pub time_steps: usize,
    /// None if capture was disabled, Some(empty) if capture emitted no tokens.
    pub ctc_spans: Option<CtcSpans<'a>>,
}
impl RecognitionView<'_> {
    pub fn try_to_owned(self) -> ImageResult<OwnedRecognition> {
        Ok(OwnedRecognition {
            text: copy_text(self.text)?,
            score: self.score,
            emitted_count: self.emitted_count,
            resized_width: self.resized_width,
            tensor_width: self.tensor_width,
            time_steps: self.time_steps,
            ctc_spans: copy_spans(self.ctc_spans)?,
        })
    }
}
pub(crate) fn copy_text(text: &str) -> ImageResult<String> {
    let mut result = String::new();
    result.try_reserve_exact(text.len()).map_err(allocation)?;
    result.push_str(text);
    Ok(result)
}
pub(crate) fn copy_spans(spans: Option<CtcSpans<'_>>) -> ImageResult<Option<Vec<OwnedCtcSpan>>> {
    spans
        .map(|spans| {
            let mut result = Vec::new();
            result.try_reserve_exact(spans.len()).map_err(allocation)?;
            for s in spans.iter() {
                result.push(OwnedCtcSpan {
                    text: copy_text(s.text)?,
                    score: s.score,
                    start_column: s.start_column,
                    end_column: s.end_column,
                });
            }
            Ok(result)
        })
        .transpose()
}
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct OwnedRecognition {
    pub text: String,
    pub score: f32,
    pub emitted_count: usize,
    pub resized_width: usize,
    pub tensor_width: usize,
    pub time_steps: usize,
    pub ctc_spans: Option<Vec<OwnedCtcSpan>>,
}
impl OwnedRecognition {
    pub fn as_view(&self) -> RecognitionView<'_> {
        RecognitionView {
            text: &self.text,
            score: self.score,
            emitted_count: self.emitted_count,
            resized_width: self.resized_width,
            tensor_width: self.tensor_width,
            time_steps: self.time_steps,
            ctc_spans: self.ctc_spans.as_deref().map(|x| CtcSpans {
                source: SpanSource::Owned(x),
            }),
        }
    }
}
pub enum RecognitionTrace<'a> {
    Input {
        shape: Shape,
        values: &'a [f32],
        resized_width: usize,
        tensor_width: usize,
    },
    Node(NodeTrace<'a>),
    Probabilities {
        shape: Shape,
        values: &'a [f32],
    },
    DecodedDense(RecognitionView<'a>),
    Public(RecognitionView<'a>),
}

struct PublicHead {
    feature: TensorId,
    feature_node: usize,
    weights: TensorId,
    bias: Option<TensorId>,
    inner: usize,
    classes: usize,
}
impl PublicHead {
    fn compile(graph: &Graph, classes: usize) -> ImageResult<Self> {
        let fail = || {
            ImageError::new(
                "recognizer requires a MatMul/optional bias/last-axis Softmax CTC head matching dictionary",
            )
        };
        let softmax = graph.nodes.last().ok_or_else(fail)?;
        if softmax.op_type != "Softmax"
            || softmax.inputs.len() != 1
            || softmax.outputs.as_slice() != [graph.outputs.first().copied()]
            || !matches!(softmax.attributes, EffectiveAttributes::Axis(-1 | 2))
        {
            return Err(fail());
        }
        let mut index = graph.nodes.len().checked_sub(2).ok_or_else(fail)?;
        let mut target = softmax.inputs[0].ok_or_else(fail)?;
        let mut bias = None;
        if graph.nodes[index].op_type == "Add" {
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
        if matmul.op_type != "MatMul"
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
    fn project(
        &self,
        graph: &Graph,
        feature: &[f32],
        indices: &mut [usize],
        scores: &mut [f32],
    ) -> ImageResult<()> {
        let weights = match &graph.tensors[self.weights.0].constant {
            Some(TensorData::F32(t)) => t.data(),
            _ => unreachable!(),
        };
        let bias = self.bias.map(|id| match &graph.tensors[id.0].constant {
            Some(TensorData::F32(t)) => t.data(),
            _ => unreachable!(),
        });
        compact_projection(
            feature,
            weights,
            bias,
            self.inner,
            self.classes,
            indices,
            scores,
        )
    }
}

#[inline]
fn update_projection_block(
    values: [f32; 4],
    column: usize,
    maximum: &mut f32,
    best: &mut usize,
) -> ImageResult<()> {
    for (offset, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(ImageError::new("nonfinite compact recognition logits"));
        }
        if value > *maximum {
            *maximum = value;
            *best = column + offset;
        }
    }
    Ok(())
}

/// Four rows share four weight columns; each dot retains native scalar bias-first order.
fn compact_projection(
    feature: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    inner: usize,
    classes: usize,
    indices: &mut [usize],
    scores: &mut [f32],
) -> ImageResult<()> {
    if inner == 0
        || classes == 0
        || Some(feature.len()) != indices.len().checked_mul(inner)
        || Some(weights.len()) != inner.checked_mul(classes)
        || scores.len() != indices.len()
        || bias.is_some_and(|b| b.len() != classes)
    {
        return Err(ImageError::new(
            "invalid compact recognition head dimensions",
        ));
    }
    let full_rows = indices.len() / 4 * 4;
    for row in (0..full_rows).step_by(4) {
        let f0 = &feature[row * inner..(row + 1) * inner];
        let f1 = &feature[(row + 1) * inner..(row + 2) * inner];
        let f2 = &feature[(row + 2) * inner..(row + 3) * inner];
        let f3 = &feature[(row + 3) * inner..(row + 4) * inner];
        let mut maximum = [f32::NEG_INFINITY; 4];
        let mut best = [0usize; 4];
        let mut col = 0;
        while col + 4 <= classes {
            let b0 = bias.map_or(0., |b| b[col]);
            let b1 = bias.map_or(0., |b| b[col + 1]);
            let b2 = bias.map_or(0., |b| b[col + 2]);
            let b3 = bias.map_or(0., |b| b[col + 3]);
            let (mut a00, mut a01, mut a02, mut a03) = (b0, b1, b2, b3);
            let (mut a10, mut a11, mut a12, mut a13) = (b0, b1, b2, b3);
            let (mut a20, mut a21, mut a22, mut a23) = (b0, b1, b2, b3);
            let (mut a30, mut a31, mut a32, mut a33) = (b0, b1, b2, b3);
            // Explicit independent accumulators remove variable-bound tile
            // loops while preserving each F32 multiply/add in ascending K.
            for k in 0..inner {
                let w = &weights[k * classes + col..k * classes + col + 4];
                let (w0, w1, w2, w3) = (w[0], w[1], w[2], w[3]);
                let (v0, v1, v2, v3) = (f0[k], f1[k], f2[k], f3[k]);
                a00 += v0 * w0;
                a01 += v0 * w1;
                a02 += v0 * w2;
                a03 += v0 * w3;
                a10 += v1 * w0;
                a11 += v1 * w1;
                a12 += v1 * w2;
                a13 += v1 * w3;
                a20 += v2 * w0;
                a21 += v2 * w1;
                a22 += v2 * w2;
                a23 += v2 * w3;
                a30 += v3 * w0;
                a31 += v3 * w1;
                a32 += v3 * w2;
                a33 += v3 * w3;
            }
            update_projection_block([a00, a01, a02, a03], col, &mut maximum[0], &mut best[0])?;
            update_projection_block([a10, a11, a12, a13], col, &mut maximum[1], &mut best[1])?;
            update_projection_block([a20, a21, a22, a23], col, &mut maximum[2], &mut best[2])?;
            update_projection_block([a30, a31, a32, a33], col, &mut maximum[3], &mut best[3])?;
            col += 4;
        }
        for col in col..classes {
            let mut sums = [bias.map_or(0., |b| b[col]); 4];
            for k in 0..inner {
                let weight = weights[k * classes + col];
                sums[0] += f0[k] * weight;
                sums[1] += f1[k] * weight;
                sums[2] += f2[k] * weight;
                sums[3] += f3[k] * weight;
            }
            for r in 0..4 {
                if !sums[r].is_finite() {
                    return Err(ImageError::new("nonfinite compact recognition logits"));
                }
                if sums[r] > maximum[r] {
                    maximum[r] = sums[r];
                    best[r] = col;
                }
            }
        }
        indices[row..row + 4].copy_from_slice(&best);
        scores[row..row + 4].copy_from_slice(&maximum);
    }
    for row in full_rows..indices.len() {
        let input = &feature[row * inner..(row + 1) * inner];
        let mut maximum = f32::NEG_INFINITY;
        let mut best = 0;
        for col in 0..classes {
            let mut sum = bias.map_or(0., |b| b[col]);
            for k in 0..inner {
                sum += input[k] * weights[k * classes + col];
            }
            if !sum.is_finite() {
                return Err(ImageError::new("nonfinite compact recognition logits"));
            }
            if sum > maximum {
                maximum = sum;
                best = col;
            }
        }
        indices[row] = best;
        scores[row] = maximum;
    }
    Ok(())
}

#[derive(Default)]
struct DecodeWorkspace {
    text: String,
    runs: Vec<CtcRun>,
    score: f32,
    emitted: usize,
    size: Option<RecognitionSize>,
    time_steps: usize,
    capture: bool,
}
impl DecodeWorkspace {
    fn decode<F>(
        &mut self,
        dictionary: &Dictionary,
        time_steps: usize,
        size: RecognitionSize,
        capture: bool,
        budget: usize,
        winner: F,
    ) -> ImageResult<()>
    where
        F: Fn(usize) -> ImageResult<(usize, f32)>,
    {
        let mut previous = 0;
        let mut bytes = 0usize;
        let mut emitted = 0usize;
        let mut sum = 0f64;
        for step in 0..time_steps {
            let (class, score) = winner(step)?;
            if class >= dictionary.class_count() || !score.is_finite() {
                return Err(ImageError::new("invalid CTC class or score"));
            }
            if class != 0 && (step == 0 || class != previous) {
                bytes = bytes
                    .checked_add(dictionary.token(class).expect("validated class").len())
                    .ok_or_else(|| ImageError::new("decoded text size overflow"))?;
                emitted += 1;
                sum += f64::from(score);
            }
            previous = class;
        }
        let required = bytes
            .checked_add(if capture {
                emitted
                    .checked_mul(size_of::<CtcRun>())
                    .ok_or_else(|| ImageError::new("CTC span size overflow"))?
            } else {
                0
            })
            .filter(|&n| n <= budget && n <= isize::MAX as usize)
            .ok_or_else(|| ImageError::new("recognition exceeds max_result_bytes"))?;
        let _ = required;
        self.text.clear();
        self.runs.clear();
        self.text.try_reserve_exact(bytes).map_err(allocation)?;
        if capture {
            self.runs.try_reserve_exact(emitted).map_err(allocation)?;
        }
        previous = 0;
        let mut open: Option<usize> = None;
        for step in 0..time_steps {
            let (class, score) = winner(step)?;
            if class != 0 && (step == 0 || class != previous) {
                if let Some(i) = open {
                    self.runs[i].end_column = step;
                }
                let start = self.text.len();
                self.text
                    .push_str(dictionary.token(class).expect("validated class"));
                if capture {
                    open = Some(self.runs.len());
                    self.runs.push(CtcRun {
                        text_start: start,
                        text_end: self.text.len(),
                        score,
                        start_column: step,
                        end_column: time_steps,
                    });
                }
            } else if class != previous
                && let Some(i) = open.take()
            {
                self.runs[i].end_column = step;
            }
            previous = class;
        }
        self.score = if emitted == 0 {
            0.
        } else {
            (sum / emitted as f64) as f32
        };
        if !self.score.is_finite() {
            return Err(ImageError::new("nonfinite CTC result score"));
        }
        self.emitted = emitted;
        self.size = Some(size);
        self.time_steps = time_steps;
        self.capture = capture;
        Ok(())
    }
    fn view(&self) -> RecognitionView<'_> {
        let size = self.size.expect("completed decode");
        RecognitionView {
            text: &self.text,
            score: self.score,
            emitted_count: self.emitted,
            resized_width: size.resized_width,
            tensor_width: size.tensor_width,
            time_steps: self.time_steps,
            ctc_spans: self
                .capture
                .then(|| CtcSpans::from_runs(&self.text, &self.runs)),
        }
    }
    fn retained_bytes(&self) -> usize {
        self.text.capacity() + self.runs.capacity() * size_of::<CtcRun>()
    }
}

pub struct Recognizer {
    compiled: Arc<CompiledModel>,
    dictionary: Arc<Dictionary>,
    options: RecognizerOptions,
    limits: ExecutionLimits,
    head: PublicHead,
    session: Session,
    diagnostic: Option<Session>,
    resize: ResizeWorkspace,
    indices: Vec<usize>,
    scores: Vec<f32>,
    decoded: DecodeWorkspace,
    dense: DecodeWorkspace,
}
impl Recognizer {
    pub fn new(
        compiled: Arc<CompiledModel>,
        dictionary: Arc<Dictionary>,
        options: RecognizerOptions,
        limits: ExecutionLimits,
    ) -> ImageResult<Self> {
        if options.target_width == 0
            || options.target_width > i32::MAX as usize
            || options.max_image_pixels == 0
            || options.max_result_bytes == 0
        {
            return Err(ImageError::new("invalid recognizer options"));
        }
        let head = PublicHead::compile(compiled.model().graph(), dictionary.class_count())?;
        let width = if options.adaptive_width {
            32
        } else {
            options.target_width
        };
        let session = Session::new_prefix(
            compiled.clone(),
            &[1, 3, 48, width],
            limits,
            head.feature_node,
        )?;
        Self::check_feature_shape(session.output_shape(), head.inner)?;
        Ok(Self {
            compiled,
            dictionary,
            options,
            limits,
            head,
            session,
            diagnostic: None,
            resize: ResizeWorkspace::default(),
            indices: Vec::new(),
            scores: Vec::new(),
            decoded: DecodeWorkspace::default(),
            dense: DecodeWorkspace::default(),
        })
    }
    fn check_feature_shape(shape: Shape, inner: usize) -> ImageResult<usize> {
        let d = shape.dimensions();
        if d.len() != 3 || d[0] != 1 || d[1] == 0 || d[2] != inner {
            return Err(ImageError::new("recognizer feature requires [1,T,inner]"));
        }
        Ok(d[1])
    }
    pub fn select_width(&self, width: usize, height: usize) -> ImageResult<RecognitionSize> {
        if width == 0 || height == 0 || width > i32::MAX as usize || height > i32::MAX as usize {
            return Err(ImageError::new("invalid recognition source dimensions"));
        }
        let natural = (48u64 * width as u64).div_ceil(height as u64);
        let target = if self.options.adaptive_width {
            natural.div_ceil(32) * 32
        } else {
            self.options.target_width as u64
        };
        if target == 0 || target > i32::MAX as u64 {
            return Err(ImageError::new("recognition tensor width overflow"));
        }
        Ok(RecognitionSize {
            resized_width: natural.min(target) as usize,
            tensor_width: target as usize,
        })
    }
    pub fn resolved_options(&self) -> RecognizerOptions {
        self.options
    }
    pub fn compiled(&self) -> &Arc<CompiledModel> {
        &self.compiled
    }
    pub fn dictionary(&self) -> &Arc<Dictionary> {
        &self.dictionary
    }
    pub fn activation_bytes(&self) -> usize {
        self.session.activation_bytes()
    }
    pub fn retained_activation_bytes(&self) -> usize {
        self.session.retained_activation_bytes()
    }
    pub fn diagnostic_retained_activation_bytes(&self) -> usize {
        self.diagnostic
            .as_ref()
            .map_or(0, Session::retained_activation_bytes)
    }
    pub fn retained_workspace_bytes(&self) -> usize {
        self.resize.retained_bytes()
            + self.session.retained_planning_bytes()
            + self
                .diagnostic
                .as_ref()
                .map_or(0, Session::retained_planning_bytes)
            + self.indices.capacity() * size_of::<usize>()
            + self.scores.capacity() * 4
            + self.decoded.retained_bytes()
            + self.dense.retained_bytes()
    }
    pub fn diagnostic_retained_workspace_bytes(&self) -> usize {
        self.diagnostic
            .as_ref()
            .map_or(0, Session::retained_planning_bytes)
            + self.dense.retained_bytes()
    }
    /// Execute the backbone and compact scalar projection. The borrowed result
    /// remains valid until the next mutable use; warmed capacities are reused.
    pub fn recognize(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
    ) -> ImageResult<RecognitionView<'_>> {
        if image.pixel_count()? > self.options.max_image_pixels {
            return Err(ImageError::new("source image exceeds max_image_pixels"));
        }
        let size = self.select_width(image.width(), image.height())?;
        self.session.reshape(&[1, 3, 48, size.tensor_width])?;
        let steps = Self::check_feature_shape(self.session.output_shape(), self.head.inner)?;
        resize_vec(&mut self.indices, steps)?;
        resize_vec(&mut self.scores, steps)?;
        let resize = &mut self.resize;
        let feature = self.session.run_with_input(|input| {
            preprocess::rec(image, size.tensor_width, input, resize)
                .map(|_| ())
                .map_err(|e| RuntimeError::new(e.to_string()))
        })?;
        self.head.project(
            self.compiled.model().graph(),
            feature.data(),
            &mut self.indices[..steps],
            &mut self.scores[..steps],
        )?;
        self.decoded.decode(
            &self.dictionary,
            steps,
            size,
            capture,
            self.options.max_result_bytes,
            |i| Ok((self.indices[i], self.scores[i])),
        )?;
        Ok(self.decoded.view())
    }
    /// Execute the complete original graph in a lazily allocated diagnostic
    /// session, emitting dense probabilities/decode and the compatible public
    /// result separately. Observers run before activation storage can be reused.
    pub fn recognize_with_trace<F>(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
        observer: F,
    ) -> ImageResult<RecognitionView<'_>>
    where
        F: for<'a> FnMut(RecognitionTrace<'a>) -> ImageResult<()>,
    {
        if image.pixel_count()? > self.options.max_image_pixels {
            return Err(ImageError::new("source image exceeds max_image_pixels"));
        }
        let size = self.select_width(image.width(), image.height())?;
        let resize = &mut self.resize;
        Self::trace_run(
            &self.compiled,
            &self.dictionary,
            self.options,
            self.limits,
            &self.head,
            &mut self.diagnostic,
            &mut self.indices,
            &mut self.scores,
            &mut self.decoded,
            &mut self.dense,
            &[1, 3, 48, size.tensor_width],
            size,
            capture,
            |input| preprocess::rec(image, size.tensor_width, input, resize).map(|_| ()),
            observer,
        )
    }
    pub(crate) fn recognize_tensor_with_trace<F>(
        &mut self,
        input: &[f32],
        input_shape: &[usize],
        resized_width: usize,
        capture: bool,
        observer: F,
    ) -> ImageResult<RecognitionView<'_>>
    where
        F: for<'a> FnMut(RecognitionTrace<'a>) -> ImageResult<()>,
    {
        if input_shape.len() != 4
            || input_shape[..3] != [1, 3, 48]
            || input_shape[3] == 0
            || input_shape[3] > i32::MAX as usize
            || resized_width == 0
            || resized_width > input_shape[3]
        {
            return Err(ImageError::new(
                "invalid frozen recognition input shape or content width",
            ));
        }
        let shape = Shape::new(input_shape).map_err(|e| ImageError::new(e.to_string()))?;
        if shape.element_count() != input.len() {
            return Err(ImageError::new("frozen recognition input length mismatch"));
        }
        Self::trace_run(
            &self.compiled,
            &self.dictionary,
            self.options,
            self.limits,
            &self.head,
            &mut self.diagnostic,
            &mut self.indices,
            &mut self.scores,
            &mut self.decoded,
            &mut self.dense,
            input_shape,
            RecognitionSize {
                resized_width,
                tensor_width: input_shape[3],
            },
            capture,
            |destination| {
                destination.copy_from_slice(input);
                Ok(())
            },
            observer,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn trace_run<'s, I, F>(
        compiled: &Arc<CompiledModel>,
        dictionary: &Dictionary,
        options: RecognizerOptions,
        limits: ExecutionLimits,
        head: &PublicHead,
        diagnostic: &mut Option<Session>,
        indices: &mut Vec<usize>,
        scores: &mut Vec<f32>,
        decoded: &'s mut DecodeWorkspace,
        dense: &mut DecodeWorkspace,
        input_shape: &[usize],
        size: RecognitionSize,
        capture: bool,
        fill: I,
        observer: F,
    ) -> ImageResult<RecognitionView<'s>>
    where
        I: FnOnce(&mut [f32]) -> ImageResult<()>,
        F: for<'a> FnMut(RecognitionTrace<'a>) -> ImageResult<()>,
    {
        if let Some(session) = diagnostic {
            session.reshape(input_shape)?;
        } else {
            *diagnostic = Some(Session::new(compiled.clone(), input_shape, limits)?);
        }
        let session = diagnostic.as_mut().expect("initialized diagnostic session");
        let output_shape = session.output_shape();
        let dims = output_shape.dimensions();
        if dims.len() != 3 || dims[0] != 1 || dims[1] == 0 || dims[2] != dictionary.class_count() {
            return Err(ImageError::new(
                "recognizer output shape is incompatible with dictionary",
            ));
        }
        let steps = dims[1];
        resize_vec(indices, steps)?;
        resize_vec(scores, steps)?;
        let observer = RefCell::new(observer);
        let projected = Cell::new(false);
        let shape = session.input_shape();
        let probabilities = session.run_with_input_and_trace(
            |input| {
                fill(input).map_err(|e| RuntimeError::new(e.to_string()))?;
                observer.borrow_mut()(RecognitionTrace::Input {
                    shape,
                    values: input,
                    resized_width: size.resized_width,
                    tensor_width: size.tensor_width,
                })
                .map_err(|e| RuntimeError::new(e.to_string()))
            },
            |node| {
                if node.output_id == head.feature {
                    let feature_steps = Self::check_feature_shape(node.shape, head.inner)
                        .map_err(|e| RuntimeError::new(e.to_string()))?;
                    if feature_steps != steps {
                        return Err(RuntimeError::new(
                            "recognizer feature and output time steps differ",
                        ));
                    }
                    head.project(
                        compiled.model().graph(),
                        node.values,
                        &mut indices[..steps],
                        &mut scores[..steps],
                    )
                    .map_err(|e| RuntimeError::new(e.to_string()))?;
                    projected.set(true);
                }
                observer.borrow_mut()(RecognitionTrace::Node(node))
                    .map_err(|e| RuntimeError::new(e.to_string()))
            },
        )?;
        observer.borrow_mut()(RecognitionTrace::Probabilities {
            shape: output_shape,
            values: probabilities.data(),
        })?;
        dense.decode(
            dictionary,
            steps,
            size,
            capture,
            options.max_result_bytes,
            |i| {
                let row = &probabilities.data()
                    [i * dictionary.class_count()..(i + 1) * dictionary.class_count()];
                let mut best = 0;
                let mut value = f32::NEG_INFINITY;
                for (class, &score) in row.iter().enumerate() {
                    if !score.is_finite() {
                        return Err(ImageError::new("nonfinite recognition probabilities"));
                    }
                    if score > value {
                        best = class;
                        value = score;
                    }
                }
                Ok((best, value))
            },
        )?;
        observer.borrow_mut()(RecognitionTrace::DecodedDense(dense.view()))?;
        if !projected.get() {
            return Err(ImageError::new("recognizer feature was not observed"));
        }
        decoded.decode(
            dictionary,
            steps,
            size,
            capture,
            options.max_result_bytes,
            |i| Ok((indices[i], scores[i])),
        )?;
        observer.borrow_mut()(RecognitionTrace::Public(decoded.view()))?;
        Ok(decoded.view())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ctc_unicode_empty_labels_and_first_frame_scores() {
        let dictionary = Dictionary::from_utf8("你\n\n👩‍🔬\n".as_bytes()).unwrap();
        let indices = [1, 1, 0, 1, 2, 3, 3, 4, 4];
        let scores = [0.2, 0.9, 0.4, 0.8, 1.5, 2., 9., 0.3, 0.8];
        let mut w = DecodeWorkspace::default();
        let size = RecognitionSize {
            resized_width: 31,
            tensor_width: 32,
        };
        w.decode(&dictionary, indices.len(), size, true, 4096, |i| {
            Ok((indices[i], scores[i]))
        })
        .unwrap();
        let out = w.view();
        assert_eq!(out.text, "你你👩‍🔬 ");
        assert_eq!(out.emitted_count, 5);
        let spans: Vec<_> = out.ctc_spans.unwrap().iter().collect();
        assert_eq!(spans[2].text, "");
        assert_eq!((spans[0].start_column, spans[0].end_column), (0, 2));
        assert_eq!((spans[4].start_column, spans[4].end_column), (7, 9));
        assert_eq!(
            out.score,
            ((0.2f32 as f64 + 0.8f32 as f64 + 1.5 + 2. + 0.3f32 as f64) / 5.) as f32
        );
        let owned = out.try_to_owned().unwrap();
        w.decode(&dictionary, indices.len(), size, false, 4096, |i| {
            Ok((indices[i], scores[i]))
        })
        .unwrap();
        assert!(w.view().ctc_spans.is_none());
        assert_eq!(w.view().text, owned.text);
        w.decode(&dictionary, 3, size, true, 4096, |_| Ok((0, 0.9)))
            .unwrap();
        assert!(w.view().ctc_spans.unwrap().is_empty());
        assert_eq!(owned.as_view().ctc_spans.unwrap().get(2).unwrap().text, "");
    }
    #[test]
    fn compact_projection_is_bias_first_and_tie_stable() {
        let features = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.];
        let weights = [1e8, 0., 0., 0., 0., -1e8, 0., 0., 0., 0.];
        let mut indices = [0; 5];
        let mut scores = [0.; 5];
        compact_projection(
            &features,
            &weights,
            Some(&[1., 0.5, 0.5, 0.5, 0.5]),
            2,
            5,
            &mut indices,
            &mut scores,
        )
        .unwrap();
        assert_eq!(indices, [1; 5]);
        assert_eq!(scores, [0.5; 5]);
        // A separately evaluated MatMul then Add would select class zero.
        assert_eq!(features[0] * weights[0] + features[1] * weights[5] + 1., 1.);
        assert!(
            compact_projection(
                &[f32::INFINITY; 10],
                &weights,
                None,
                2,
                5,
                &mut indices,
                &mut scores
            )
            .is_err()
        );
    }
    #[test]
    fn compact_four_by_four_projection_covers_row_and_column_tails() {
        let (rows, inner, classes) = (7, 9, 7);
        let feature: Vec<_> = (0..rows * inner).map(|i| (i as f32 - 19.) / 8.).collect();
        let weights: Vec<_> = (0..inner * classes)
            .map(|i| ((i * 13 % 23) as f32 - 12.) / 16.)
            .collect();
        let bias = [-2., 3., -4., 1., -0.5, 2., -3.];
        let mut indices = vec![0; rows];
        let mut scores = vec![0.; rows];
        compact_projection(
            &feature,
            &weights,
            Some(&bias),
            inner,
            classes,
            &mut indices,
            &mut scores,
        )
        .unwrap();
        for row in 0..rows {
            let mut maximum = f32::NEG_INFINITY;
            let mut best = 0;
            for col in 0..classes {
                let mut sum = bias[col];
                for k in 0..inner {
                    sum += feature[row * inner + k] * weights[k * classes + col];
                }
                if sum > maximum {
                    maximum = sum;
                    best = col;
                }
            }
            assert_eq!(indices[row], best);
            assert_eq!(scores[row].to_bits(), maximum.to_bits());
        }
        compact_projection(
            &[1.],
            &[0., 0., 0.],
            Some(&[-9., -2., -2.]),
            1,
            3,
            &mut indices[..1],
            &mut scores[..1],
        )
        .unwrap();
        assert_eq!(indices[0], 1);
        assert_eq!(scores[0], -2.);
        // Even a non-winning candidate must be finite.
        assert!(
            compact_projection(
                &[1.],
                &[0., f32::NEG_INFINITY, 1.],
                None,
                1,
                3,
                &mut indices[..1],
                &mut scores[..1]
            )
            .is_err()
        );
    }
    #[test]
    fn decode_budget_and_bad_classes_recover() {
        let dictionary = Dictionary::from_utf8(b"a").unwrap();
        let mut w = DecodeWorkspace::default();
        let size = RecognitionSize {
            resized_width: 32,
            tensor_width: 32,
        };
        assert!(
            w.decode(&dictionary, 1, size, true, 1, |_| Ok((1, 2.)))
                .is_err()
        );
        assert!(
            w.decode(&dictionary, 1, size, false, 100, |_| Ok((3, 2.)))
                .is_err()
        );
        w.decode(&dictionary, 1, size, false, 100, |_| Ok((1, 2.)))
            .unwrap();
        assert_eq!(w.view().text, "a");
    }
}
