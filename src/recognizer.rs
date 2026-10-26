//! Reusable scalar recognition with distinct dense and compatible public scores.
use crate::profile::{StageProfile, Timer};
use crate::{
    image::{ImageError, ImageResult, ImageView, resize_vec},
    preprocess::{self, ResizeWorkspace},
    projection::ProjectionHead,
    runtime::{
        CompiledModel, CpuCompiledModel, ExecutionLimits, NodeTrace, RuntimeError, stage::Session,
    },
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
        let raw = std::str::from_utf8(bytes).map_err(|e| {
            ImageError::caused_by(
                crate::error::ErrorKind::InvalidInput,
                "dictionary is not valid UTF-8",
                e,
            )
        })?;
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
fn allocation(error: std::collections::TryReserveError) -> ImageError {
    ImageError::caused_by(
        crate::error::ErrorKind::Allocation,
        "recognition allocation failed",
        error,
    )
}

/// Public CTC score semantics. Probability mode executes the model's complete
/// Softmax head, retaining dense activations within the execution memory limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RecognitionScoreMode {
    /// Uncalibrated projection maxima, preserving the compact default path.
    #[default]
    Logit,
    /// Softmax probabilities in [0, 1], suitable for confidence thresholds.
    Probability,
}

#[derive(Clone, Copy, Debug)]
pub struct RecognizerOptions {
    /// Tensor width in fixed mode; adaptive mode selects its own width.
    pub target_width: usize,
    /// Round the natural height-48 content width up to a multiple of 32,
    /// without a 320-pixel upper cap.
    pub adaptive_width: bool,
    /// Minimum adaptive tensor width before rounding up to 32. Zero preserves
    /// natural-width selection. Fixed-width mode ignores this floor.
    pub min_width: usize,
    pub score_mode: RecognitionScoreMode,
    pub max_image_pixels: usize,
    /// Active UTF-8 and alignment records; excludes shared dictionary and activations.
    pub max_result_bytes: usize,
}
impl Default for RecognizerOptions {
    fn default() -> Self {
        Self {
            target_width: 320,
            adaptive_width: true,
            min_width: 0,
            score_mode: RecognitionScoreMode::default(),
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
    /// Mean score at each token's first frame, using the selected score mode.
    /// Diagnostic dense scores are always probabilities.
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

/// Borrowed results of one same-width recognition graph run, in input order.
/// Storage is reused by the recognizer on its next mutable call.
#[derive(Clone, Copy)]
pub struct RecognitionBatchView<'a> {
    decoded: &'a [DecodeWorkspace],
}
impl std::fmt::Debug for RecognitionBatchView<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_list().entries(self.iter()).finish()
    }
}
impl<'a> RecognitionBatchView<'a> {
    pub fn len(self) -> usize {
        self.decoded.len()
    }
    pub fn is_empty(self) -> bool {
        self.decoded.is_empty()
    }
    pub fn get(self, index: usize) -> Option<RecognitionView<'a>> {
        self.decoded.get(index).map(DecodeWorkspace::view)
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = RecognitionView<'a>> {
        self.decoded.iter().map(DecodeWorkspace::view)
    }
    pub fn try_to_owned(self) -> ImageResult<Vec<OwnedRecognition>> {
        let mut results = Vec::new();
        results.try_reserve_exact(self.len()).map_err(allocation)?;
        for result in self.iter() {
            results.push(result.try_to_owned()?);
        }
        Ok(results)
    }
}
impl Serialize for RecognitionBatchView<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
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

/// Selection made by the last successful compact production projection.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ProjectionStatistics {
    /// Compiled policy; `isa` reports the shape-selected implementation.
    pub backend: crate::runtime::Backend,
    pub isa: &'static str,
    pub family: &'static str,
    pub rows: usize,
    pub unit_rows: usize,
    pub threads: usize,
    pub retained_scratch_bytes: usize,
}
// The compact CPU tail packs only its vocabulary matrix, never the GPU backbone.
trait ProjectionSource: Sync {
    fn model(&self) -> &crate::model::Model;
    fn backend(&self) -> crate::runtime::Backend;
    fn packed_matrix(
        &self,
        id: crate::model::TensorId,
    ) -> Option<&crate::runtime::packing::PackedMatrix>;
}
impl ProjectionSource for CpuCompiledModel {
    fn model(&self) -> &crate::model::Model {
        self.model()
    }
    fn backend(&self) -> crate::runtime::Backend {
        self.backend()
    }
    fn packed_matrix(
        &self,
        id: crate::model::TensorId,
    ) -> Option<&crate::runtime::packing::PackedMatrix> {
        self.packed_matrix(id)
    }
}
struct CpuProjection {
    model: Arc<crate::model::Model>,
    backend: crate::runtime::Backend,
    id: crate::model::TensorId,
    packed: Option<Arc<crate::runtime::packing::PackedMatrix>>,
}
impl CpuProjection {
    fn new(compiled: &CompiledModel, head: &ProjectionHead) -> ImageResult<Self> {
        let model = compiled.model_arc();
        let backend = compiled.as_cpu().map(|c| c.backend()).unwrap_or(
            crate::runtime::simd::select_backend(compiled.cpu_options.kernels)
                .map_err(ImageError::new)?,
        );
        let packed = if backend != crate::runtime::Backend::Scalar {
            let Some(TensorData::F32(weight)) = &model.graph().tensors[head.weights.0].constant
            else {
                unreachable!()
            };
            Some(compiled.packed.matrix(
                head.weights,
                crate::runtime::packing::MatrixPacking::RowMajor,
                || {
                    crate::runtime::packing::PackedMatrix::new(
                        weight.data(),
                        head.inner,
                        head.classes,
                    )
                    .map_err(ImageError::new)
                },
            )?)
        } else {
            None
        };
        Ok(Self {
            model,
            backend,
            id: head.weights,
            packed,
        })
    }
}
impl ProjectionSource for CpuProjection {
    fn model(&self) -> &crate::model::Model {
        &self.model
    }
    fn backend(&self) -> crate::runtime::Backend {
        self.backend
    }
    fn packed_matrix(
        &self,
        id: crate::model::TensorId,
    ) -> Option<&crate::runtime::packing::PackedMatrix> {
        (id == self.id).then_some(self.packed.as_deref()).flatten()
    }
}

impl ProjectionHead {
    #[allow(clippy::too_many_arguments)]
    fn project_parallel(
        &self,
        compiled: &impl ProjectionSource,
        feature: &[f32],
        indices: &mut [usize],
        scores: &mut [f32],
        cpu: &crate::cpu::CpuContext,
        threads: usize,
        unit_rows: usize,
        workspace: &mut crate::runtime::simd::ArgMaxWorkspace,
    ) -> ImageResult<(&'static str, &'static str)> {
        if compiled.backend() == crate::runtime::Backend::Avx512 {
            let graph = compiled.model().graph();
            let weights = match &graph.tensors[self.weights.0].constant {
                Some(TensorData::F32(t)) => t.data(),
                _ => unreachable!(),
            };
            let bias = self
                .bias
                .map_or(&[][..], |id| match &graph.tensors[id.0].constant {
                    Some(TensorData::F32(t)) => t.data(),
                    _ => unreachable!(),
                });
            if crate::runtime::simd::try_compact_columns(
                compiled.backend(),
                feature,
                weights,
                bias,
                compiled.packed_matrix(self.weights),
                indices,
                scores,
                unit_rows,
                self.inner,
                self.classes,
                cpu,
                threads,
                workspace,
            )
            .map_err(ImageError::new)?
            {
                return Ok(("packed_column_shards", "Avx512"));
            }
        }
        #[cfg(feature = "parallel")]
        if threads > 1 && indices.len() >= 8 && indices.len() * self.inner * self.classes >= 262_144
        {
            use rayon::prelude::*;
            let rows = indices.len().div_ceil(threads).max(4).next_multiple_of(4);
            cpu.install(|| {
                indices
                    .par_chunks_mut(rows)
                    .zip(scores.par_chunks_mut(rows))
                    .enumerate()
                    .try_for_each(|(chunk, (indices, scores))| {
                        let first = chunk * rows * self.inner;
                        self.project(
                            compiled,
                            &feature[first..first + indices.len() * self.inner],
                            indices,
                            scores,
                        )
                    })
            })?;
            let first = self.projection_isa(compiled, rows);
            let last = self.projection_isa(
                compiled,
                match indices.len() % rows {
                    0 => rows,
                    n => n,
                },
            );
            return Ok((
                "row_shards",
                if first == last {
                    first
                } else {
                    "Avx512+Avx2Fma"
                },
            ));
        }
        #[cfg(not(feature = "parallel"))]
        let _ = threads;
        cpu.install(|| self.project(compiled, feature, indices, scores))?;
        Ok(("sequential", self.projection_isa(compiled, indices.len())))
    }
    fn projection_isa(&self, compiled: &impl ProjectionSource, rows: usize) -> &'static str {
        use crate::runtime::{Backend, simd::wide_projection_eligible};
        match compiled.backend() {
            Backend::Scalar => "Scalar",
            Backend::Portable4 => "Portable4",
            Backend::NeonFma => "NeonFma",
            Backend::Avx => "Avx",
            Backend::AvxFma => "AvxFma",
            Backend::Avx2Fma => "Avx2Fma",
            Backend::Avx512
                if wide_projection_eligible(
                    rows,
                    self.inner,
                    self.classes,
                    compiled.packed_matrix(self.weights).is_some(),
                ) =>
            {
                "Avx512"
            }
            Backend::Avx512 => "Avx2Fma",
        }
    }
    fn project(
        &self,
        compiled: &impl ProjectionSource,
        feature: &[f32],
        indices: &mut [usize],
        scores: &mut [f32],
    ) -> ImageResult<()> {
        let graph = compiled.model().graph();
        let weights = match &graph.tensors[self.weights.0].constant {
            Some(TensorData::F32(t)) => t.data(),
            _ => unreachable!(),
        };
        let bias = self.bias.map(|id| match &graph.tensors[id.0].constant {
            Some(TensorData::F32(t)) => t.data(),
            _ => unreachable!(),
        });
        if compiled.backend() != crate::runtime::Backend::Scalar {
            return crate::runtime::simd::compact_argmax(
                compiled.backend(),
                feature,
                weights,
                bias.unwrap_or(&[]),
                compiled.packed_matrix(self.weights),
                indices,
                scores,
                self.inner,
                self.classes,
            )
            .map_err(ImageError::new);
        }
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
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    fn assemble_gpu(
        &mut self,
        dictionary: &Dictionary,
        source: crate::runtime::gpu::heads::Decoded<'_>,
        size: RecognitionSize,
        capture: bool,
        budget: usize,
    ) -> ImageResult<()> {
        let needed = source
            .text_bytes
            .checked_add(if capture {
                source
                    .tokens
                    .len()
                    .checked_mul(size_of::<CtcRun>())
                    .ok_or_else(|| ImageError::new("CTC spans overflow"))?
            } else {
                0
            })
            .filter(|&n| n <= budget)
            .ok_or_else(|| ImageError::new("recognition exceeds max_result_bytes"))?;
        let _ = needed;
        self.text.clear();
        self.runs.clear();
        self.text
            .try_reserve(source.text_bytes)
            .map_err(allocation)?;
        if capture {
            self.runs
                .try_reserve(source.tokens.len())
                .map_err(allocation)?;
        }
        self.emitted = source.tokens.len();
        self.score = source.score;
        for token in source.tokens.iter() {
            let start = self.text.len();
            self.text.push_str(
                dictionary
                    .token(token.class)
                    .ok_or_else(|| ImageError::new("invalid GPU token"))?,
            );
            if capture {
                self.runs.push(CtcRun {
                    text_start: start,
                    text_end: self.text.len(),
                    score: token.score,
                    start_column: token.start,
                    end_column: token.end,
                });
            }
        }
        if self.text.len() != source.text_bytes {
            return Err(ImageError::new("GPU dictionary length mismatch"));
        }
        self.size = Some(size);
        self.time_steps = source.steps;
        self.capture = capture;
        Ok(())
    }
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

#[derive(Clone)]
pub struct RecognizerPlan {
    execution: crate::runtime::stage::RequestSettings,
    compiled: CompiledModel,
    options: RecognizerOptions,
    limits: ExecutionLimits,
    head: ProjectionHead,
    dictionary: Arc<Dictionary>,
    projection_cpu: Arc<std::sync::OnceLock<ImageResult<CpuProjection>>>,
}
pub struct Recognizer {
    plan: Arc<RecognizerPlan>,
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    gpu: Option<crate::runtime::gpu::heads::Recognition>,
    session: Session,
    diagnostic: Option<Session>,
    resize: ResizeWorkspace,
    indices: Vec<usize>,
    scores: Vec<f32>,
    decoded: DecodeWorkspace,
    dense: DecodeWorkspace,
    batch_sizes: Vec<RecognitionSize>,
    batch_decoded: Vec<DecodeWorkspace>,
    projection: crate::runtime::simd::ArgMaxWorkspace,
    last_projection: Option<ProjectionStatistics>,
}
impl Recognizer {
    pub fn plan(&self) -> &Arc<RecognizerPlan> {
        &self.plan
    }
    pub fn uses_gpu(&self) -> bool {
        self.session.is_gpu()
    }
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub(crate) fn reset_run_statistics(&mut self) {
        if let Some(gpu) = &mut self.gpu {
            gpu.statistics = Default::default();
        }
    }
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub fn gpu_pipeline_statistics(&self) -> Option<&crate::runtime::gpu::GpuPipelineStatistics> {
        self.gpu.as_ref().map(|gpu| &gpu.statistics)
    }
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    fn recognize_gpu<const PROFILE: bool>(
        &mut self,
        images: &[ImageView<'_>],
        capture: bool,
        profile: &mut StageProfile,
    ) -> ImageResult<()> {
        self.batch_sizes.clear();
        for image in images {
            if image.pixel_count()? > self.plan.options.max_image_pixels {
                return Err(ImageError::new("source image exceeds max_image_pixels"));
            }
            let size = self.select_width(image.width(), image.height())?;
            if self
                .batch_sizes
                .first()
                .is_some_and(|first| first.tensor_width != size.tensor_width)
            {
                return Err(ImageError::new(
                    "recognition batch requires the same selected tensor width",
                ));
            }
            self.batch_sizes.push(size);
        }
        if self.gpu.is_none() {
            self.gpu = Some(crate::runtime::gpu::heads::Recognition::new(
                self.plan.compiled.as_gpu().unwrap().clone(),
                &self.plan.dictionary,
                self.batch_sizes[0].tensor_width,
                self.plan.limits,
            )?);
        }
        let (bytes, steps) = self.gpu.as_mut().unwrap().run(
            images,
            &self.batch_sizes,
            self.plan.options.score_mode == RecognitionScoreMode::Probability,
            capture,
            self.plan.options.max_result_bytes,
            PROFILE,
        )?;
        if PROFILE {
            let statistics = &self.gpu.as_ref().unwrap().statistics;
            profile.record_gpu(statistics);
            profile.projection_ns += statistics.projection_encoding_ns;
            profile.recognized_lines += images.len();
        }
        let decoded = crate::runtime::gpu::heads::Recognition::parse(
            &bytes,
            images.len(),
            steps,
            self.plan.dictionary.class_count(),
        )?;
        self.batch_decoded
            .resize_with(images.len(), DecodeWorkspace::default);
        let mut remaining = self.plan.options.max_result_bytes;
        for ((destination, source), &size) in self
            .batch_decoded
            .iter_mut()
            .zip(decoded)
            .zip(&self.batch_sizes)
        {
            destination.assemble_gpu(&self.plan.dictionary, source, size, capture, remaining)?;
            remaining -= destination.text.len() + destination.runs.len() * size_of::<CtcRun>();
        }
        self.last_projection = None;
        Ok(())
    }
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    fn resident_fallback(&mut self, error: ImageError) -> ImageResult<()> {
        let compiled = self
            .plan
            .compiled
            .resident_fallback(RuntimeError::from(error))?;
        let mut replacement = Self::new(
            compiled,
            self.plan.dictionary.clone(),
            self.plan.options,
            self.plan.limits,
        )?;
        replacement.set_execution_context(
            self.session.cpu_context().clone(),
            self.session.intra_op_threads(),
        )?;
        *self = replacement;
        Ok(())
    }
    pub fn fork(&self) -> ImageResult<Self> {
        self.plan.request()
    }
    pub fn set_execution_context(
        &mut self,
        cpu: Arc<crate::cpu::CpuContext>,
        threads: usize,
    ) -> ImageResult<()> {
        self.session.set_execution_context(cpu.clone(), threads)?;
        if let Some(diagnostic) = &mut self.diagnostic {
            diagnostic.set_execution_context(cpu.clone(), threads)?;
        }
        let execution = &mut Arc::make_mut(&mut self.plan).execution;
        execution.cpu = cpu;
        execution.threads = threads.min(16);
        Ok(())
    }
    fn validate(
        compiled: impl Into<CompiledModel>,
        dictionary: Arc<Dictionary>,
        options: RecognizerOptions,
        limits: ExecutionLimits,
    ) -> ImageResult<RecognizerPlan> {
        let compiled = compiled.into();
        if options.target_width == 0
            || options.target_width > i32::MAX as usize
            || options.min_width > (i32::MAX as usize / 32) * 32
            || options.max_image_pixels == 0
            || options.max_result_bytes == 0
        {
            return Err(ImageError::new("invalid recognizer options"));
        }
        let head = ProjectionHead::compile(
            compiled.model().graph(),
            dictionary.class_count(),
            3,
            "recognizer",
        )?;
        let width = if options.adaptive_width {
            options.min_width.max(1).div_ceil(32) * 32
        } else {
            options.target_width
        };
        let prefix =
            (options.score_mode == RecognitionScoreMode::Logit).then_some(head.feature_node);
        let output = compiled.validate_shape(&[1, 3, 48, width], limits, prefix)?;
        let output_width = match options.score_mode {
            RecognitionScoreMode::Logit => head.inner,
            RecognitionScoreMode::Probability => head.classes,
        };
        Self::check_feature_shape(output, output_width)?;
        let projection_cpu = Arc::new(std::sync::OnceLock::new());
        if !compiled.is_gpu() {
            projection_cpu
                .set(Ok(CpuProjection::new(&compiled, &head)?))
                .ok();
        }
        Ok(RecognizerPlan {
            execution: Default::default(),
            compiled,
            options,
            limits,
            head,
            dictionary,
            projection_cpu,
        })
    }
    pub fn new(
        compiled: impl Into<CompiledModel>,
        dictionary: Arc<Dictionary>,
        options: RecognizerOptions,
        limits: ExecutionLimits,
    ) -> ImageResult<Self> {
        RecognizerPlan::new(compiled, dictionary, options, limits)?.request()
    }
    fn from_plan(plan: Arc<RecognizerPlan>) -> ImageResult<Self> {
        let width = if plan.options.adaptive_width {
            plan.options.min_width.max(1).div_ceil(32) * 32
        } else {
            plan.options.target_width
        };
        let prefix = (plan.options.score_mode == RecognitionScoreMode::Logit)
            .then_some(plan.head.feature_node);
        let mut session = Session::deferred(
            plan.compiled.clone(),
            &[1, 3, 48, width],
            plan.limits,
            prefix,
        )?;
        session.set_execution_context(plan.execution.cpu.clone(), plan.execution.threads)?;
        Ok(Self {
            plan,
            #[cfg(all(feature = "gpu", not(target_os = "macos")))]
            gpu: None,
            session,
            diagnostic: None,
            resize: ResizeWorkspace::default(),
            indices: Vec::new(),
            scores: Vec::new(),
            decoded: DecodeWorkspace::default(),
            dense: DecodeWorkspace::default(),
            batch_sizes: Vec::new(),
            batch_decoded: Vec::new(),
            projection: crate::runtime::simd::ArgMaxWorkspace::default(),
            last_projection: None,
        })
    }
    fn output_width(&self) -> usize {
        match self.plan.options.score_mode {
            RecognitionScoreMode::Logit => self.plan.head.inner,
            RecognitionScoreMode::Probability => self.plan.head.classes,
        }
    }
    fn check_feature_shape(shape: Shape, inner: usize) -> ImageResult<usize> {
        Self::check_batch_feature_shape(shape, inner, 1)
    }
    fn check_batch_feature_shape(shape: Shape, inner: usize, batch: usize) -> ImageResult<usize> {
        let d = shape.dimensions();
        if d.len() != 3 || d[0] != batch || d[1] == 0 || d[2] != inner {
            return Err(ImageError::new("recognizer feature requires [N,T,inner]"));
        }
        Ok(d[1])
    }
    pub fn select_width(&self, width: usize, height: usize) -> ImageResult<RecognitionSize> {
        self.plan.select_width(width, height)
    }
    pub fn resolved_options(&self) -> RecognizerOptions {
        self.plan.options
    }
    pub fn compiled(&self) -> &CompiledModel {
        &self.plan.compiled
    }
    pub(crate) fn counter_snapshot(&self) -> crate::runtime::StatisticsCounters {
        let mut primary = self.session.counter_snapshot();
        if let Some(diagnostic) = &self.diagnostic {
            primary.add_diagnostics(diagnostic.counter_snapshot());
        }
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        if let Some(gpu) = &self.gpu {
            let mut counters = gpu.counter_snapshot();
            counters.add_diagnostics(primary);
            return counters;
        }
        primary
    }
    pub fn execution_statistics(&self) -> crate::runtime::ExecutionStatistics {
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        if let Some(gpu) = &self.gpu {
            return gpu
                .execution_statistics()
                .with_diagnostics(self.counter_snapshot());
        }
        self.session
            .statistics()
            .with_diagnostics(self.counter_snapshot())
    }
    pub fn execution_limits(&self) -> ExecutionLimits {
        self.plan.limits
    }
    pub fn dictionary(&self) -> &Arc<Dictionary> {
        &self.plan.dictionary
    }
    pub fn projection_statistics(&self) -> Option<ProjectionStatistics> {
        self.last_projection
    }
    pub fn activation_bytes(&self) -> usize {
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        let resident = self
            .gpu
            .as_ref()
            .map_or(0, |g| g.session.activation_bytes());
        #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
        let resident = 0;
        resident + self.session.activation_bytes()
    }
    pub fn retained_device_bytes(&self) -> usize {
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        let resident = self.gpu.as_ref().map_or(0, |g| g.retained_bytes());
        #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
        let resident = 0;
        resident
            + self.session.retained_device_bytes()
            + self
                .diagnostic
                .as_ref()
                .map_or(0, Session::retained_device_bytes)
    }
    pub fn retained_activation_bytes(&self) -> usize {
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        let resident = self
            .gpu
            .as_ref()
            .map_or(0, |g| g.session.retained_activation_bytes());
        #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
        let resident = 0;
        resident + self.session.retained_activation_bytes()
    }
    pub fn diagnostic_retained_activation_bytes(&self) -> usize {
        self.diagnostic
            .as_ref()
            .map_or(0, Session::retained_activation_bytes)
    }
    pub fn retained_workspace_bytes(&self) -> usize {
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        let resident = self
            .gpu
            .as_ref()
            .map_or(0, |g| g.session.retained_planning_bytes());
        #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
        let resident = 0;
        resident
            + self.resize.retained_bytes()
            + (self.session.retained_planning_bytes() + self.session.retained_scratch_bytes())
            + self.diagnostic.as_ref().map_or(0, |session| {
                session.retained_planning_bytes() + session.retained_scratch_bytes()
            })
            + self.indices.capacity() * size_of::<usize>()
            + self.scores.capacity() * 4
            + self.projection.retained_bytes()
            + self.decoded.retained_bytes()
            + self.dense.retained_bytes()
            + self.batch_sizes.capacity() * size_of::<RecognitionSize>()
            + self.batch_decoded.capacity() * size_of::<DecodeWorkspace>()
            + self
                .batch_decoded
                .iter()
                .map(DecodeWorkspace::retained_bytes)
                .sum::<usize>()
    }
    pub fn diagnostic_retained_workspace_bytes(&self) -> usize {
        self.diagnostic.as_ref().map_or(0, |session| {
            session.retained_planning_bytes() + session.retained_scratch_bytes()
        }) + self.dense.retained_bytes()
    }
    /// Retained execution and compact-head scratch. This is a subset of
    /// retained workspace bytes and excludes fixed local stack tiles.
    pub fn retained_kernel_scratch_bytes(&self) -> usize {
        self.session.retained_scratch_bytes()
            + self
                .diagnostic
                .as_ref()
                .map_or(0, Session::retained_scratch_bytes)
            + self.projection.retained_bytes()
    }
    /// Execute the backbone and compact scalar projection. The borrowed result
    /// remains valid until the next mutable use; warmed capacities are reused.
    pub fn recognize(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
    ) -> ImageResult<RecognitionView<'_>> {
        let cpu = self.session.cpu_context().clone();
        if self.session.is_gpu() {
            return self.recognize_impl::<false>(image, capture, &mut StageProfile::default());
        }
        cpu.install(|| self.recognize_impl::<false>(image, capture, &mut StageProfile::default()))
    }
    /// Run crops with the same selected tensor width as one `[N,3,48,W]` graph.
    /// Content widths, scores and CTC spans remain independent per image. The
    /// result byte limit applies to the whole batch; an empty batch is valid.
    pub fn recognize_batch(
        &mut self,
        images: &[ImageView<'_>],
        capture: bool,
    ) -> ImageResult<RecognitionBatchView<'_>> {
        let cpu = self.session.cpu_context().clone();
        if self.session.is_gpu() {
            return self.recognize_batch_impl::<false, _>(
                images.len(),
                |i| Ok(images[i]),
                capture,
                &mut StageProfile::default(),
            );
        }
        cpu.install(|| {
            self.recognize_batch_impl::<false, _>(
                images.len(),
                |index| Ok(images[index]),
                capture,
                &mut StageProfile::default(),
            )
        })
    }
    pub(crate) fn recognize_batch_impl<'a, const PROFILE: bool, F>(
        &mut self,
        count: usize,
        image: F,
        capture: bool,
        profile: &mut StageProfile,
    ) -> ImageResult<RecognitionBatchView<'_>>
    where
        F: Fn(usize) -> ImageResult<ImageView<'a>>,
    {
        #[cfg(test)]
        self.session
            .cpu_context()
            .assert_production_thread("REC batch");
        if count == 0 {
            return Ok(RecognitionBatchView { decoded: &[] });
        }
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        if self.session.is_gpu() {
            let images = (0..count).map(image).collect::<ImageResult<Vec<_>>>()?;
            if let Err(error) = self.recognize_gpu::<PROFILE>(&images, capture, profile) {
                self.resident_fallback(error)?;
                let cpu = self.session.cpu_context().clone();
                let source: &(dyn Fn(usize) -> ImageResult<ImageView<'a>> + Sync) =
                    &|i| Ok(images[i]);
                return cpu.install(|| {
                    self.recognize_batch_impl::<PROFILE, _>(count, source, capture, profile)
                });
            }
            return Ok(RecognitionBatchView {
                decoded: &self.batch_decoded[..count],
            });
        }
        let binding_timer = Timer::<PROFILE>::start();
        self.batch_sizes.clear();
        self.batch_sizes
            .try_reserve_exact(count)
            .map_err(allocation)?;
        for index in 0..count {
            let source = image(index)?;
            if source.pixel_count()? > self.plan.options.max_image_pixels {
                return Err(ImageError::new("source image exceeds max_image_pixels"));
            }
            let size = self.select_width(source.width(), source.height())?;
            if self
                .batch_sizes
                .first()
                .is_some_and(|first| first.tensor_width != size.tensor_width)
            {
                return Err(ImageError::new(
                    "recognition batch requires the same selected tensor width",
                ));
            }
            self.batch_sizes.push(size);
        }
        let width = self.batch_sizes[0].tensor_width;
        if PROFILE {
            profile.binding_ns = profile.binding_ns.saturating_add(binding_timer.elapsed());
        }
        let reshape_timer = Timer::<PROFILE>::start();
        let reshape = self.session.reshape(&[count, 3, 48, width]);
        if PROFILE {
            profile.reshape_ns = profile.reshape_ns.saturating_add(reshape_timer.elapsed());
        }
        reshape?;
        let binding_timer = Timer::<PROFILE>::start();
        let steps = Self::check_batch_feature_shape(
            self.session.output_shape(),
            self.output_width(),
            count,
        )?;
        let rows = count
            .checked_mul(steps)
            .ok_or_else(|| ImageError::new("recognition batch row count overflow"))?;
        resize_vec(&mut self.indices, rows)?;
        resize_vec(&mut self.scores, rows)?;
        self.batch_decoded
            .try_reserve_exact(count.saturating_sub(self.batch_decoded.len()))
            .map_err(allocation)?;
        while self.batch_decoded.len() < count {
            self.batch_decoded.push(DecodeWorkspace::default());
        }
        if PROFILE {
            profile.binding_ns = profile.binding_ns.saturating_add(binding_timer.elapsed());
        }
        let sample_elements = width
            .checked_mul(3 * 48)
            .ok_or_else(|| ImageError::new("recognition batch input size overflow"))?;
        let resize = &mut self.resize;
        let cpu = self.session.cpu_context().clone();
        let threads = self.session.intra_op_threads();
        let backbone_timer = Timer::<PROFILE>::start();
        let mut preprocessing_ns = 0;
        let feature =
            self.session
                .run_with_layout_input_profiled::<PROFILE, _>(|input, nhwc| {
                    let timer = Timer::<PROFILE>::start();
                    for (index, sample) in input.chunks_exact_mut(sample_elements).enumerate() {
                        let source = image(index).map_err(RuntimeError::from)?;
                        preprocess::rec_with_layout(source, width, sample, resize, nhwc)
                            .map_err(RuntimeError::from)?;
                    }
                    preprocessing_ns = timer.elapsed();
                    Ok(())
                })?;
        if PROFILE {
            profile.rec_preprocess_ns = profile.rec_preprocess_ns.saturating_add(preprocessing_ns);
            profile.rec_backbone_ns = profile
                .rec_backbone_ns
                .saturating_add(backbone_timer.elapsed().saturating_sub(preprocessing_ns));
        }
        let projection_timer = Timer::<PROFILE>::start();
        self.last_projection = if self.plan.options.score_mode == RecognitionScoreMode::Probability
        {
            probability_argmax(
                feature.data(),
                self.plan.head.classes,
                &mut self.indices[..rows],
                &mut self.scores[..rows],
            )?;
            None
        } else {
            let (family, isa) = self.plan.head.project_parallel(
                self.plan
                    .projection_cpu
                    .get_or_init(|| CpuProjection::new(&self.plan.compiled, &self.plan.head))
                    .as_ref()
                    .map_err(Clone::clone)?,
                feature.data(),
                &mut self.indices[..rows],
                &mut self.scores[..rows],
                &cpu,
                threads,
                steps,
                &mut self.projection,
            )?;
            Some(ProjectionStatistics {
                backend: self
                    .plan
                    .projection_cpu
                    .get()
                    .unwrap()
                    .as_ref()
                    .map_err(Clone::clone)?
                    .backend(),
                isa,
                family,
                rows,
                unit_rows: steps,
                threads,
                retained_scratch_bytes: self.projection.retained_bytes(),
            })
        };
        if PROFILE {
            profile.projection_ns = profile
                .projection_ns
                .saturating_add(projection_timer.elapsed());
        }
        self.session.accumulate_gpu_profile::<PROFILE>(profile);
        let ctc_timer = Timer::<PROFILE>::start();
        let mut remaining = self.plan.options.max_result_bytes;
        for (index, (decoded, &size)) in self.batch_decoded[..count]
            .iter_mut()
            .zip(&self.batch_sizes)
            .enumerate()
        {
            let offset = index * steps;
            decoded.decode(
                &self.plan.dictionary,
                steps,
                size,
                capture,
                remaining,
                |step| Ok((self.indices[offset + step], self.scores[offset + step])),
            )?;
            let bytes = decoded
                .runs
                .len()
                .checked_mul(size_of::<CtcRun>())
                .and_then(|bytes| bytes.checked_add(decoded.text.len()))
                .ok_or_else(|| ImageError::new("recognition batch result size overflow"))?;
            remaining = remaining
                .checked_sub(bytes)
                .ok_or_else(|| ImageError::new("recognition batch exceeds max_result_bytes"))?;
        }
        if PROFILE {
            profile.ctc_ns = profile.ctc_ns.saturating_add(ctc_timer.elapsed());
            profile.recognized_lines = profile.recognized_lines.saturating_add(count);
        }
        Ok(RecognitionBatchView {
            decoded: &self.batch_decoded[..count],
        })
    }
    /// Profile the compact production path without initializing dense diagnostics.
    #[cfg(feature = "profiling")]
    pub fn recognize_profiled(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
    ) -> ImageResult<(RecognitionView<'_>, StageProfile)> {
        let timer = Timer::<true>::start();
        let mut profile = StageProfile::default();
        let cpu = self.session.cpu_context().clone();
        let result = if self.session.is_gpu() {
            self.recognize_impl::<true>(image, capture, &mut profile)
        } else {
            cpu.install(|| self.recognize_impl::<true>(image, capture, &mut profile))
        }?;
        profile.total_ns = timer.elapsed();
        Ok((result, profile))
    }
    pub(crate) fn recognize_impl<const PROFILE: bool>(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
        profile: &mut StageProfile,
    ) -> ImageResult<RecognitionView<'_>> {
        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        if self.session.is_gpu() {
            if let Err(error) = self.recognize_gpu::<PROFILE>(&[image], capture, profile) {
                self.resident_fallback(error)?;
                let cpu = self.session.cpu_context().clone();
                return cpu.install(|| self.recognize_impl::<PROFILE>(image, capture, profile));
            }
            return Ok(self.batch_decoded[0].view());
        }
        #[cfg(test)]
        if !self.session.is_gpu() {
            self.session.cpu_context().assert_production_thread("REC");
        }
        let binding_timer = Timer::<PROFILE>::start();
        if image.pixel_count()? > self.plan.options.max_image_pixels {
            return Err(ImageError::new("source image exceeds max_image_pixels"));
        }
        let size = self.select_width(image.width(), image.height())?;
        if PROFILE {
            profile.binding_ns = profile.binding_ns.saturating_add(binding_timer.elapsed());
        }
        let reshape_timer = Timer::<PROFILE>::start();
        let reshape = self.session.reshape(&[1, 3, 48, size.tensor_width]);
        if PROFILE {
            profile.reshape_ns = profile.reshape_ns.saturating_add(reshape_timer.elapsed());
        }
        reshape?;
        let binding_timer = Timer::<PROFILE>::start();
        let steps = Self::check_feature_shape(self.session.output_shape(), self.output_width())?;
        resize_vec(&mut self.indices, steps)?;
        resize_vec(&mut self.scores, steps)?;
        if PROFILE {
            profile.binding_ns = profile.binding_ns.saturating_add(binding_timer.elapsed());
        }
        let resize = &mut self.resize;
        let cpu = self.session.cpu_context().clone();
        let threads = self.session.intra_op_threads();
        let backbone_timer = Timer::<PROFILE>::start();
        let mut preprocessing_ns = 0;
        let feature =
            self.session
                .run_with_layout_input_profiled::<PROFILE, _>(|input, nhwc| {
                    let timer = Timer::<PROFILE>::start();
                    let result =
                        preprocess::rec_with_layout(image, size.tensor_width, input, resize, nhwc)
                            .map(|_| ())
                            .map_err(RuntimeError::from);
                    preprocessing_ns = timer.elapsed();
                    result
                })?;
        if PROFILE {
            profile.rec_preprocess_ns = profile.rec_preprocess_ns.saturating_add(preprocessing_ns);
            profile.rec_backbone_ns = profile
                .rec_backbone_ns
                .saturating_add(backbone_timer.elapsed().saturating_sub(preprocessing_ns));
        }
        let projection_timer = Timer::<PROFILE>::start();
        self.last_projection = if self.plan.options.score_mode == RecognitionScoreMode::Probability
        {
            probability_argmax(
                feature.data(),
                self.plan.head.classes,
                &mut self.indices[..steps],
                &mut self.scores[..steps],
            )?;
            None
        } else {
            let (family, isa) = self.plan.head.project_parallel(
                self.plan
                    .projection_cpu
                    .get_or_init(|| CpuProjection::new(&self.plan.compiled, &self.plan.head))
                    .as_ref()
                    .map_err(Clone::clone)?,
                feature.data(),
                &mut self.indices[..steps],
                &mut self.scores[..steps],
                &cpu,
                threads,
                steps,
                &mut self.projection,
            )?;
            Some(ProjectionStatistics {
                backend: self
                    .plan
                    .projection_cpu
                    .get()
                    .unwrap()
                    .as_ref()
                    .map_err(Clone::clone)?
                    .backend(),
                isa,
                family,
                rows: steps,
                unit_rows: steps,
                threads,
                retained_scratch_bytes: self.projection.retained_bytes(),
            })
        };
        if PROFILE {
            profile.projection_ns = profile
                .projection_ns
                .saturating_add(projection_timer.elapsed());
        }
        self.session.accumulate_gpu_profile::<PROFILE>(profile);
        let ctc_timer = Timer::<PROFILE>::start();
        self.decoded.decode(
            &self.plan.dictionary,
            steps,
            size,
            capture,
            self.plan.options.max_result_bytes,
            |i| Ok((self.indices[i], self.scores[i])),
        )?;
        if PROFILE {
            profile.ctc_ns = profile.ctc_ns.saturating_add(ctc_timer.elapsed());
            profile.recognized_lines += 1;
        }
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
        if image.pixel_count()? > self.plan.options.max_image_pixels {
            return Err(ImageError::new("source image exceeds max_image_pixels"));
        }
        let size = self.select_width(image.width(), image.height())?;
        let resize = &mut self.resize;
        Self::trace_run(
            &self.plan.compiled,
            self.plan
                .projection_cpu
                .get_or_init(|| CpuProjection::new(&self.plan.compiled, &self.plan.head))
                .as_ref()
                .map_err(Clone::clone)?,
            &self.plan.dictionary,
            self.plan.options,
            self.plan.limits,
            &self.plan.head,
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
    #[allow(clippy::too_many_arguments)]
    fn trace_run<'s, I, F>(
        compiled: &CompiledModel,
        projection_cpu: &CpuProjection,
        dictionary: &Dictionary,
        options: RecognizerOptions,
        limits: ExecutionLimits,
        head: &ProjectionHead,
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
            *diagnostic = Some(Session::new_diagnostic(
                compiled.clone(),
                input_shape,
                limits,
            )?);
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
                fill(input).map_err(RuntimeError::from)?;
                observer.borrow_mut()(RecognitionTrace::Input {
                    shape,
                    values: input,
                    resized_width: size.resized_width,
                    tensor_width: size.tensor_width,
                })
                .map_err(RuntimeError::from)
            },
            |node| {
                if options.score_mode == RecognitionScoreMode::Logit
                    && node.output_id == head.feature
                {
                    let feature_steps = Self::check_feature_shape(node.shape, head.inner)
                        .map_err(RuntimeError::from)?;
                    if feature_steps != steps {
                        return Err(RuntimeError::new(
                            "recognizer feature and output time steps differ",
                        ));
                    }
                    head.project(
                        projection_cpu,
                        node.values,
                        &mut indices[..steps],
                        &mut scores[..steps],
                    )
                    .map_err(RuntimeError::from)?;
                    projected.set(true);
                }
                observer.borrow_mut()(RecognitionTrace::Node(node)).map_err(RuntimeError::from)
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
        if options.score_mode == RecognitionScoreMode::Probability {
            probability_argmax(
                probabilities.data(),
                dictionary.class_count(),
                &mut indices[..steps],
                &mut scores[..steps],
            )?;
        } else if !projected.get() {
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

fn probability_argmax(
    probabilities: &[f32],
    classes: usize,
    indices: &mut [usize],
    scores: &mut [f32],
) -> ImageResult<()> {
    if classes == 0
        || probabilities.len() / classes != indices.len()
        || !probabilities.len().is_multiple_of(classes)
        || indices.len() != scores.len()
    {
        return Err(ImageError::new("invalid recognition probability shape"));
    }
    for ((row, index), score) in probabilities.chunks_exact(classes).zip(indices).zip(scores) {
        *index = 0;
        *score = -1.;
        for (class, &value) in row.iter().enumerate() {
            if !value.is_finite() || !(0. ..=1.).contains(&value) {
                return Err(ImageError::new("invalid recognition probability"));
            }
            if value > *score {
                *index = class;
                *score = value;
            }
        }
    }
    Ok(())
}

impl RecognizerPlan {
    pub fn execution_limits(&self) -> ExecutionLimits {
        self.limits
    }
    pub fn dictionary(&self) -> &Arc<Dictionary> {
        &self.dictionary
    }
    pub fn select_width(&self, width: usize, height: usize) -> ImageResult<RecognitionSize> {
        if width == 0 || height == 0 || width > i32::MAX as usize || height > i32::MAX as usize {
            return Err(ImageError::new("invalid recognition source dimensions"));
        }
        let natural = (48u64 * width as u64).div_ceil(height as u64);
        let target = if self.options.adaptive_width {
            natural.max(self.options.min_width as u64).div_ceil(32) * 32
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

    pub fn new(
        compiled: impl Into<CompiledModel>,
        dictionary: Arc<Dictionary>,
        options: RecognizerOptions,
        limits: ExecutionLimits,
    ) -> ImageResult<Arc<Self>> {
        Ok(Arc::new(Recognizer::validate(
            compiled, dictionary, options, limits,
        )?))
    }
    pub fn request(self: &Arc<Self>) -> ImageResult<Recognizer> {
        Recognizer::from_plan(self.clone())
    }
    pub fn options(&self) -> RecognizerOptions {
        self.options
    }
    pub fn compiled(&self) -> &CompiledModel {
        &self.compiled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cpu::CpuContext, image::PixelFormat};

    #[test]
    fn probability_scores_match_softmax_in_reused_single_batch_and_trace_requests() {
        let mut pipeline = crate::ocr::tests::synthetic_ocr(CpuContext::serial());
        let template = pipeline.recognizer_mut();
        let pixels = vec![255; 64 * 48 * 3];
        let image = ImageView::new(&pixels, 64, 48, 0, PixelFormat::Bgr24).unwrap();
        assert_eq!(template.recognize(image, true).unwrap().score, 3.);
        let expected_score = 3_f32.exp() / (3_f32.exp() + 2.);
        for capacity in 1..=if cfg!(feature = "parallel") { 2 } else { 1 } {
            let mut recognizer = Recognizer::new(
                template.compiled().clone(),
                template.dictionary().clone(),
                RecognizerOptions {
                    score_mode: RecognitionScoreMode::Probability,
                    ..Default::default()
                },
                ExecutionLimits::default(),
            )
            .unwrap();
            recognizer
                .set_execution_context(Arc::new(CpuContext::new(capacity).unwrap()), capacity)
                .unwrap();
            let expected = recognizer
                .recognize(image, true)
                .unwrap()
                .try_to_owned()
                .unwrap();
            assert_eq!(expected.text, "A");
            assert!((expected.score - expected_score).abs() < 1e-6);
            assert!(recognizer.projection_statistics().is_none());
            for _ in 0..2 {
                for result in recognizer
                    .recognize_batch(&[image, image], true)
                    .unwrap()
                    .iter()
                {
                    assert_eq!(result.try_to_owned().unwrap(), expected);
                }
                assert_eq!(
                    recognizer
                        .recognize(image, true)
                        .unwrap()
                        .try_to_owned()
                        .unwrap(),
                    expected
                );
            }
            let mut traced_score = None;
            let traced = recognizer
                .recognize_with_trace(image, true, |event| {
                    if let RecognitionTrace::DecodedDense(view) = event {
                        traced_score = Some(view.score);
                    }
                    Ok(())
                })
                .unwrap()
                .try_to_owned()
                .unwrap();
            assert_eq!(traced, expected);
            assert_eq!(traced_score, Some(expected.score));
            assert_eq!(
                recognizer
                    .fork()
                    .unwrap()
                    .recognize(image, true)
                    .unwrap()
                    .try_to_owned()
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn probability_scores_reject_invalid_outputs() {
        let mut indices = [0];
        let mut scores = [0.];
        for values in [[0., f32::NAN], [0., f32::INFINITY], [0., 1.1], [-0.1, 1.]] {
            assert!(probability_argmax(&values, 2, &mut indices, &mut scores).is_err());
        }
        assert!(probability_argmax(&[0.5], 2, &mut indices, &mut scores).is_err());
        assert!(probability_argmax(&[], 0, &mut indices, &mut scores).is_err());
        probability_argmax(&[0.5, 0.5], 2, &mut indices, &mut scores).unwrap();
        assert_eq!(indices, [0]);
        assert_eq!(scores, [0.5]);
    }

    #[test]
    fn same_width_batch_is_one_graph_and_preserves_independent_unicode_ctc() {
        let mut pipeline = crate::ocr::tests::synthetic_ocr(CpuContext::serial());
        let template = pipeline.recognizer_mut();
        let mut recognizer = Recognizer::new(
            template.compiled().clone(),
            Arc::new(Dictionary::from_utf8("你👩‍🔬\n".as_bytes()).unwrap()),
            RecognizerOptions::default(),
            ExecutionLimits::default(),
        )
        .unwrap();
        let widths = [33, 48, 63];
        let pixels: Vec<Vec<u8>> = widths
            .iter()
            .enumerate()
            .map(|(index, &width)| {
                let mut pixels = vec![0; width * 48 * 3];
                for y in 0..48 {
                    for x in 0..width {
                        if index == 0 || (index == 1 && x % 16 < 8) {
                            pixels[(y * width + x) * 3..(y * width + x) * 3 + 3].fill(255);
                        }
                    }
                }
                pixels
            })
            .collect();
        let images: Vec<_> = pixels
            .iter()
            .zip(widths)
            .map(|(pixels, width)| {
                ImageView::new(pixels, width, 48, 0, PixelFormat::Bgr24).unwrap()
            })
            .collect();
        let expected: Vec<_> = images
            .iter()
            .map(|&image| {
                recognizer
                    .recognize(image, true)
                    .unwrap()
                    .try_to_owned()
                    .unwrap()
            })
            .collect();
        assert!(expected[0].text.contains("你👩‍🔬"));
        assert!(expected[2].text.is_empty());
        let actual = recognizer.recognize_batch(&images, true).unwrap();
        assert_eq!(actual.len(), 3);
        assert!(!actual.is_empty());
        assert!(actual.get(3).is_none());
        assert_eq!(actual.try_to_owned().unwrap(), expected);
        assert_eq!(
            recognizer.session.input_shape().dimensions(),
            &[3, 3, 48, 64]
        );
        let retained = recognizer.retained_workspace_bytes();
        recognizer.recognize_batch(&images, false).unwrap();
        assert_eq!(
            recognizer
                .recognize_batch(&images, true)
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
        assert_eq!(recognizer.retained_workspace_bytes(), retained);
        assert!(recognizer.recognize_batch(&[], true).unwrap().is_empty());
        assert_eq!(
            recognizer
                .recognize(images[0], true)
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected[0]
        );
    }

    #[test]
    fn batch_validation_and_aggregate_result_budget_recover() {
        let mut pipeline = crate::ocr::tests::synthetic_ocr(CpuContext::serial());
        let template = pipeline.recognizer_mut();
        let mut recognizer = Recognizer::new(
            template.compiled().clone(),
            Arc::new(Dictionary::from_utf8("你\n".as_bytes()).unwrap()),
            RecognizerOptions {
                max_result_bytes: 3,
                ..Default::default()
            },
            ExecutionLimits::default(),
        )
        .unwrap();
        let pixels = vec![255; 64 * 48 * 3];
        let image = ImageView::new(&pixels, 64, 48, 0, PixelFormat::Bgr24).unwrap();
        assert_eq!(recognizer.recognize(image, false).unwrap().text, "你");
        assert!(
            recognizer
                .recognize_batch(&[image, image], false)
                .unwrap_err()
                .to_string()
                .contains("max_result_bytes")
        );
        assert_eq!(
            recognizer
                .recognize_batch(&[image], false)
                .unwrap()
                .get(0)
                .unwrap()
                .text,
            "你"
        );
        let different_pixels = vec![255; 65 * 48 * 3];
        let different = ImageView::new(&different_pixels, 65, 48, 0, PixelFormat::Bgr24).unwrap();
        let previous = recognizer.session.input_shape();
        assert!(
            recognizer
                .recognize_batch(&[image, different], false)
                .unwrap_err()
                .to_string()
                .contains("same selected tensor width")
        );
        assert_eq!(recognizer.session.input_shape(), previous);
        assert_eq!(recognizer.recognize(image, false).unwrap().text, "你");
    }

    #[test]
    fn batch_activation_limit_failure_preserves_single_request_reuse() {
        let mut pipeline = crate::ocr::tests::synthetic_ocr(CpuContext::serial());
        let template = pipeline.recognizer_mut();
        let pixels = vec![255; 16 * 48 * 3];
        let image = ImageView::new(&pixels, 16, 48, 0, PixelFormat::Bgr24).unwrap();
        template.recognize(image, false).unwrap();
        let mut recognizer = Recognizer::new(
            template.compiled().clone(),
            template.dictionary().clone(),
            RecognizerOptions::default(),
            ExecutionLimits {
                max_activation_bytes: template.retained_activation_bytes(),
            },
        )
        .unwrap();
        let pixels = vec![255; 16 * 48 * 3];
        let image = ImageView::new(&pixels, 16, 48, 0, PixelFormat::Bgr24).unwrap();
        let expected = recognizer
            .recognize(image, true)
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert!(recognizer.recognize_batch(&[image, image], true).is_err());
        assert_eq!(
            recognizer.session.input_shape().dimensions(),
            &[1, 3, 48, 32]
        );
        assert_eq!(
            recognizer
                .recognize(image, true)
                .unwrap()
                .try_to_owned()
                .unwrap(),
            expected
        );
    }
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
