//! Compact GPU projections and CTC. Vocabulary logits never leave device storage.
use super::{resident::*, *};
use crate::{image::ImageView, projection::ProjectionHead, recognizer::Dictionary};

pub(crate) struct Head {
    pub metadata: ProjectionHead,
    owned_bytes: usize,
    weights: GpuBuffer,
    bias: GpuBuffer,
    lengths: GpuBuffer,
    context: Arc<GpuContext>,
}
impl Head {
    pub fn shared(
        compiled: &Arc<GpuCompiledModel>,
        classes: usize,
        rank: usize,
        dictionary: Option<&Dictionary>,
    ) -> RuntimeResult<Arc<Self>> {
        if classes > u32::MAX as usize
            || (0..classes).any(|i| {
                dictionary
                    .and_then(|d| d.token(i))
                    .is_some_and(|s| s.len() > u32::MAX as usize)
            })
        {
            return Err(resource(
                "GPU dictionary exceeds 32-bit token metadata capacity",
            ));
        }
        let mut key = vec![classes as u32, rank as u32];
        key.extend((0..classes).map(|i| {
            dictionary
                .and_then(|d| d.token(i))
                .map_or(0, |s| s.len() as u32)
        }));
        let mut cache = compiled
            .heads
            .lock()
            .map_err(|_| RuntimeError::new("GPU projection cache poisoned"))?;
        if let Some(head) = cache.get(&key) {
            return Ok(head.clone());
        }
        let head = Arc::new(Self::new(compiled, classes, rank, dictionary)?);
        // A model has one public head. Bound optional dictionary-length variants.
        if cache.len() >= 8 {
            cache.clear();
        }
        cache.insert(key, head.clone());
        Ok(head)
    }
    pub fn new(
        compiled: &Arc<GpuCompiledModel>,
        classes: usize,
        rank: usize,
        dictionary: Option<&Dictionary>,
    ) -> RuntimeResult<Self> {
        let context = compiled.context.clone();
        let model = compiled.model();
        let mut owned_bytes = 0;
        let metadata = ProjectionHead::compile(model.graph(), classes, rank, "GPU projection")
            .map_err(RuntimeError::from)?;
        let data = |id: TensorId| match &model.graph().tensors[id.0].constant {
            Some(TensorData::F32(t)) => t.data(),
            _ => unreachable!(),
        };
        let mut upload = |label,
                          id: Option<TensorId>,
                          fallback: &[f32]|
         -> RuntimeResult<GpuBuffer> {
            if compiled.precision == GpuPrecision::F32
                && let Some(buffer) = id.and_then(|id| compiled.weights.constants[id.0].as_ref())
            {
                return Ok(buffer.clone());
            }
            let bytes = bytemuck::cast_slice(id.map(data).unwrap_or(fallback));
            let buffer = context.create_buffer_init(label, bytes, wgpu::BufferUsages::STORAGE)?;
            owned_bytes += buffer.bytes();
            Ok(buffer)
        };
        let weights = upload("GPU projection weights F32", Some(metadata.weights), &[])?;
        let bias = upload(
            "GPU projection bias F32",
            metadata.bias,
            &vec![0f32; classes],
        )?;
        let lengths: Vec<u32> = (0..classes)
            .map(|class| {
                dictionary
                    .and_then(|d| d.token(class))
                    .map_or(0, |s| s.len() as u32)
            })
            .collect();
        let lengths = context.create_buffer_init(
            "GPU dictionary token lengths",
            bytemuck::cast_slice(&lengths),
            wgpu::BufferUsages::STORAGE,
        )?;
        owned_bytes += lengths.bytes();
        Ok(Self {
            owned_bytes,
            metadata,
            weights,
            bias,
            lengths,
            context,
        })
    }
    pub fn retained_bytes(&self) -> usize {
        self.owned_bytes
    }
    pub fn project(
        &self,
        batch: &mut Batch,
        features: &Tensor,
        probability: bool,
    ) -> RuntimeResult<GpuBuffer> {
        batch.ensure_context(&self.context)?;
        batch.ensure_context(&features.context)?;
        if features.descriptor.layout != GpuLayout::Canonical
            || !features
                .descriptor
                .shape
                .element_count()
                .is_multiple_of(self.metadata.inner)
        {
            return Err(RuntimeError::new("noncanonical projection input"));
        }
        let rows = features.descriptor.shape.element_count() / self.metadata.inner;
        let tiles = self.metadata.classes.div_ceil(64);
        let partial = batch.storage(
            "projection partial winners",
            rows.checked_mul(tiles)
                .and_then(|n| n.checked_mul(16))
                .ok_or_else(|| resource("projection scratch overflow"))?,
        )?;
        let winners = batch.storage("projection winners", rows * 16)?;
        let precision = features.descriptor.precision;
        let rows_per_group =
            if rows >= 16 && self.metadata.classes >= 4096 && precision == GpuPrecision::F32 {
                4
            } else {
                1
            };
        let kernel = if rows_per_group == 4 {
            shader("heads4.wgsl", include_str!("heads4.wgsl"))
        } else {
            shader("heads.wgsl", include_str!("heads.wgsl"))
        };
        let source = format!(
            "{}{}{}",
            if precision == GpuPrecision::MixedF16 {
                "enable f16;\n"
            } else {
                ""
            },
            PARAMETERS,
            kernel.replace(
                "$INPUT",
                if precision == GpuPrecision::MixedF16 {
                    "f16"
                } else {
                    "f32"
                }
            )
        );
        let words = [
            rows as u32,
            self.metadata.inner as u32,
            self.metadata.classes as u32,
            tiles as u32,
            u32::from(probability),
        ];
        batch.dispatch(
            &format!("projection_tiles{rows_per_group}_{precision:?}"),
            &source,
            &[
                (&features.buffer, true),
                (&self.weights, true),
                (&self.bias, true),
                (&partial, false),
            ],
            &words,
            [tiles as u32, rows.div_ceil(rows_per_group) as u32, 1],
        )?;
        let source = format!(
            "{PARAMETERS}{}",
            shader("head_reduce.wgsl", include_str!("head_reduce.wgsl"))
        );
        batch.dispatch(
            "projection_winners",
            &source,
            &[(&partial, true), (&winners, false)],
            &words,
            [(rows as u32).div_ceil(64), 1, 1],
        )?;
        Ok(winners)
    }
    pub fn decode(
        &self,
        batch: &mut Batch,
        winners: &GpuBuffer,
        sequences: usize,
        steps: usize,
        capture: bool,
        max_bytes: usize,
    ) -> RuntimeResult<GpuBuffer> {
        let records = steps
            .checked_add(1)
            .and_then(|n| n.checked_mul(sequences))
            .and_then(|n| n.checked_mul(24))
            .ok_or_else(|| resource("CTC output overflow"))?;
        let output = batch.storage("compact CTC records", records)?;
        let words = [
            sequences as u32,
            steps as u32,
            self.metadata.classes as u32,
            u32::from(capture),
            max_bytes.min(u32::MAX as usize) as u32,
            std::mem::size_of::<crate::recognizer::CtcRun>() as u32,
        ];
        let source = format!(
            "{PARAMETERS}{}",
            shader("ctc.wgsl", include_str!("ctc.wgsl"))
        );
        batch.dispatch(
            "ctc_compaction",
            &source,
            &[(winners, true), (&self.lengths, true), (&output, false)],
            &words,
            [sequences as u32, 1, 1],
        )?;
        Ok(output)
    }
}

pub(crate) struct Recognition {
    head: Arc<Head>,
    pub session: GpuSession,
    pub statistics: GpuPipelineStatistics,
}
pub(crate) struct Classification {
    head: Arc<Head>,
    pub session: GpuSession,
    pub statistics: GpuPipelineStatistics,
}
impl Classification {
    pub fn new(compiled: Arc<GpuCompiledModel>, limits: ExecutionLimits) -> RuntimeResult<Self> {
        compiled.prepare_images()?;
        let head = Head::shared(&compiled, 2, 2, None)?;
        let session = GpuSession::new_resident(
            compiled,
            &[1, 3, 80, 160],
            limits,
            Some(head.metadata.feature_node),
        )?;
        Ok(Self {
            head,
            session,
            statistics: Default::default(),
        })
    }
    pub fn retained_bytes(&self) -> usize {
        self.session.retained_device_bytes()
    }
    pub fn record(
        &mut self,
        batch: &mut Batch,
        image: &Image,
    ) -> RuntimeResult<(GpuBuffer, usize)> {
        self.prepare_batch(1)?;
        let (winners, widths) = self.record_batch(batch, std::slice::from_ref(image))?;
        Ok((winners, widths[0]))
    }
    /// Choose the batch before recording any graphs. Keep this shape fixed
    /// through a submission: queue parameter writes precede encoded GPU work.
    pub fn prepare_batch(&mut self, capacity: usize) -> RuntimeResult<usize> {
        if capacity == 0 || capacity > 4 {
            return Err(RuntimeError::new(
                "invalid resident classifier batch capacity",
            ));
        }
        let graph = self.session.compiled.model().graph();
        let mut capacity = match &graph.tensors[graph.inputs[0].0].shape {
            crate::tensor::ShapeSpec::Ranked(dims)
                if matches!(dims.first(), Some(crate::tensor::Dim::Known(1))) =>
            {
                1
            }
            _ => capacity,
        };
        loop {
            match self.session.reshape(&[capacity, 3, 80, 160]) {
                Ok(()) => return Ok(capacity),
                // Batching is optional. Preserve requests whose activation
                // budget admits the original one-crop classifier only.
                Err(error) if error.kind() == ErrorKind::ResourceLimit && capacity > 1 => {
                    capacity = capacity.div_ceil(2);
                }
                Err(error) => return Err(error),
            }
        }
    }
    pub fn record_batch(
        &mut self,
        batch: &mut Batch,
        images: &[Image],
    ) -> RuntimeResult<(GpuBuffer, Vec<usize>)> {
        let capacity = self.session.input_shape().dimensions()[0];
        if images.is_empty() || images.len() > capacity {
            return Err(RuntimeError::new("invalid resident classifier batch"));
        }
        let input = self.session.resident_input();
        let mut widths = Vec::with_capacity(images.len());
        for (index, image) in images.iter().enumerate() {
            let window = image.width.min(image.height.saturating_mul(4));
            let width = (80u64 * window as u64)
                .div_ceil(image.height as u64)
                .min(160) as usize;
            preprocess_at(
                batch,
                image,
                &input,
                [window, image.height],
                width,
                false,
                index,
            )?;
            widths.push(width);
        }
        // Tail rows are independent batch elements and their winners are never
        // consumed. Keeping their storage avoids a shape/parameter change while
        // earlier classifier graphs are still in the same command buffer.
        let features = batch.graph(&mut self.session, "classifier")?;
        Ok((self.head.project(batch, &features, true)?, widths))
    }
    pub fn run(
        &mut self,
        image: ImageView<'_>,
        profiling: bool,
    ) -> RuntimeResult<(Vec<u8>, usize)> {
        let mut batch = Batch::new(self.head.context.clone())?;
        batch.profiling = profiling;
        let image = Image::upload(&mut batch, image)?;
        let (winner, width) = self.record(&mut batch, &image)?;
        let (bytes, statistics) = batch.read(&winner, 16, false)?;
        super::vision::merge_statistics(&mut self.statistics, statistics);
        Ok((bytes, width))
    }
}
pub(crate) struct Token {
    pub class: usize,
    pub score: f32,
    pub start: usize,
    pub end: usize,
}
/// A validated view into the compact readback, without a second token allocation.
#[derive(Clone, Copy)]
pub(crate) struct Tokens<'a>(&'a [u8]);
impl Tokens<'_> {
    pub fn len(self) -> usize {
        self.0.len() / 24
    }
    #[cfg(test)]
    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = Token> {
        self.0.as_chunks::<24>().0.iter().map(|record| {
            let word = |i: usize| u32::from_le_bytes(record[i * 4..i * 4 + 4].try_into().unwrap());
            Token {
                class: word(0) as usize,
                score: f32::from_bits(word(1)),
                start: word(2) as usize,
                end: word(3) as usize,
            }
        })
    }
}
pub(crate) struct Decoded<'a> {
    pub tokens: Tokens<'a>,
    pub score: f32,
    pub text_bytes: usize,
    pub steps: usize,
}
impl Recognition {
    pub fn new(
        compiled: Arc<GpuCompiledModel>,
        dictionary: &Dictionary,
        width: usize,
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        // FP16 weight rounding can move a close CTC token boundary even with
        // FP32 attention residuals. Preserve canonical spans in resident OCR.
        let compiled = compiled.with_full_precision()?;
        compiled.prepare_images()?;
        let head = Head::shared(&compiled, dictionary.class_count(), 3, Some(dictionary))?;
        let session = GpuSession::new_resident(
            compiled,
            &[1, 3, 48, width],
            limits,
            Some(head.metadata.feature_node),
        )?;
        Ok(Self {
            head,
            session,
            statistics: Default::default(),
        })
    }
    pub fn retained_bytes(&self) -> usize {
        self.session.retained_device_bytes()
    }
    pub fn fork(&self) -> RuntimeResult<Self> {
        let session = GpuSession::new_resident(
            self.session.compiled.clone(),
            &[1, 3, 48, 32],
            self.session.limits,
            Some(self.head.metadata.feature_node),
        )?;
        Ok(Self {
            head: self.head.clone(),
            session,
            statistics: Default::default(),
        })
    }
    pub fn execution_statistics(&self) -> crate::runtime::ExecutionStatistics {
        self.session.statistics()
    }
    pub fn run(
        &mut self,
        images: &[ImageView<'_>],
        sizes: &[crate::recognizer::RecognitionSize],
        probability: bool,
        capture: bool,
        max_bytes: usize,
        profiling: bool,
    ) -> RuntimeResult<(Vec<u8>, usize)> {
        let mut batch = Batch::new(self.head.context.clone())?;
        batch.profiling = profiling;
        let images = images
            .iter()
            .map(|&image| Image::upload(&mut batch, image))
            .collect::<RuntimeResult<Vec<_>>>()?;
        let (buffer, steps) =
            self.record(&mut batch, &images, sizes, probability, capture, max_bytes)?;
        let (bytes, statistics) = batch.read(&buffer, (steps + 1) * images.len() * 24, false)?;
        super::vision::merge_statistics(&mut self.statistics, statistics);
        Ok((bytes, steps))
    }
    pub fn record(
        &mut self,
        batch: &mut Batch,
        images: &[Image],
        sizes: &[crate::recognizer::RecognitionSize],
        probability: bool,
        capture: bool,
        max_bytes: usize,
    ) -> RuntimeResult<(GpuBuffer, usize)> {
        if images.is_empty()
            || sizes.len() != images.len()
            || sizes
                .iter()
                .any(|s| s.tensor_width != sizes[0].tensor_width)
        {
            return Err(RuntimeError::new("incompatible resident recognition batch"));
        }
        self.session
            .reshape(&[images.len(), 3, 48, sizes[0].tensor_width])?;
        let input = self.session.resident_input();
        for (i, (image, size)) in images.iter().zip(sizes).enumerate() {
            preprocess_at(
                batch,
                image,
                &input,
                [image.width, image.height],
                size.resized_width,
                false,
                i,
            )?;
        }
        let features = batch.graph(&mut self.session, "recognizer")?;
        let dimensions = features.descriptor.shape.dimensions();
        if dimensions.len() != 3
            || dimensions[0] != images.len()
            || dimensions[2] != self.head.metadata.inner
        {
            return Err(RuntimeError::new(
                "invalid resident recognition feature shape",
            ));
        }
        let steps = dimensions[1];
        let projection_timer = batch.profiling.then(std::time::Instant::now);
        let winners = self.head.project(batch, &features, probability)?;
        batch.statistics.projection_encoding_ns +=
            projection_timer.map_or(0, |t| t.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        let output = self
            .head
            .decode(batch, &winners, images.len(), steps, capture, max_bytes)?;
        Ok((output, steps))
    }
    pub fn parse(
        bytes: &[u8],
        count: usize,
        steps: usize,
        classes: usize,
    ) -> RuntimeResult<Vec<Decoded<'_>>> {
        if bytes.len() != count * (steps + 1) * 24 {
            return Err(RuntimeError::new("invalid compact CTC length"));
        }
        let word = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        let mut decoded = Vec::with_capacity(count);
        for sequence in 0..count {
            let base = sequence * (steps + 1) * 6;
            let error = word(base + 3);
            if error & 2 != 0 {
                return Err(resource("recognition exceeds max_result_bytes"));
            }
            if error != 0 {
                return Err(RuntimeError::new("invalid GPU projection class or score"));
            }
            let emitted = word(base) as usize;
            if emitted > steps {
                return Err(RuntimeError::new("invalid GPU CTC record count"));
            }
            let start = (base + 6) * 4;
            let tokens = Tokens(&bytes[start..start + emitted * 24]);
            for token in tokens.iter() {
                if token.class == 0
                    || token.class >= classes
                    || !token.score.is_finite()
                    || token.start >= token.end
                    || token.end > steps
                {
                    return Err(RuntimeError::new("invalid GPU CTC record"));
                }
            }
            let score = f32::from_bits(word(base + 2));
            if !score.is_finite() {
                return Err(RuntimeError::new("invalid GPU CTC score"));
            }
            decoded.push(Decoded {
                tokens,
                score,
                text_bytes: word(base + 1) as usize,
                steps,
            });
        }
        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projection_context(shader_f16: bool) -> Arc<GpuContext> {
        let backend = match std::env::var("SIMD_OCR_GPU_BACKEND").as_deref() {
            Ok("dx12") => GpuBackendPreference::Dx12,
            Ok("metal") => GpuBackendPreference::Metal,
            _ => GpuBackendPreference::Vulkan,
        };
        let context = GpuContext::new_with_options(GpuContextOptions {
            backend,
            shader_f16,
            ..Default::default()
        })
        .unwrap();
        assert!(matches!(
            context.adapter_info().device_type,
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
        ));
        context
    }

    fn projection_fixture(
        context: &Arc<GpuContext>,
        [rows, inner, classes]: [usize; 3],
        precision: GpuPrecision,
        [features, weights, bias]: [&[f32]; 3],
    ) -> (Head, Tensor) {
        let upload = |label, bytes: &[u8]| {
            context
                .create_buffer_init(
                    label,
                    bytes,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                )
                .unwrap()
        };
        let head = Head {
            owned_bytes: 0,
            metadata: ProjectionHead {
                feature: TensorId(0),
                feature_node: 0,
                weights: TensorId(0),
                bias: None,
                inner,
                classes,
            },
            weights: upload("projection test weights", bytemuck::cast_slice(weights)),
            bias: upload("projection test bias", bytemuck::cast_slice(bias)),
            lengths: upload(
                "projection test token lengths",
                bytemuck::cast_slice(&vec![1u32; classes]),
            ),
            context: context.clone(),
        };
        let features = upload("projection test features", &precision.encode(features));
        let features = Tensor {
            context: context.clone(),
            buffer: features,
            descriptor: GpuTensorDescriptor::canonical(
                Shape::new(&[rows, 1, inner]).unwrap(),
                precision,
            ),
        };
        (head, features)
    }

    #[test]
    #[ignore = "requires a physical GPU with shader-f16 support"]
    fn gpu_projection_score_modes_preserve_contract() {
        let context = projection_context(true);
        for (rows, classes, precision) in [
            (5, 130, GpuPrecision::F32),
            (19, 4097, GpuPrecision::F32),
            (5, 130, GpuPrecision::MixedF16),
        ] {
            let inner = 7;
            let mut features = vec![0f32; rows * inner];
            for row in 0..rows {
                features[row * inner] = if row % 2 == 0 { 1. } else { -1. };
            }
            let mut weights = vec![0f32; inner * classes];
            weights[63] = 2.;
            weights[64] = 2.;
            weights[classes - 1] = -2.;
            let mut bias = vec![0f32; classes];
            for class in [63, 64, classes - 1] {
                bias[class] = 4.;
            }
            let (head, features) = projection_fixture(
                &context,
                [rows, inner, classes],
                precision,
                [&features, &weights, &bias],
            );
            for probability in [false, true] {
                let mut batch = Batch::new(context.clone()).unwrap();
                let winners = head.project(&mut batch, &features, probability).unwrap();
                let (actual, _) = batch.read(&winners, rows * 16, false).unwrap();
                for (row, record) in actual.as_chunks::<16>().0.iter().enumerate() {
                    let word =
                        |i: usize| u32::from_le_bytes(record[i * 4..i * 4 + 4].try_into().unwrap());
                    // Equal winners straddle tiles 0 and 1; odd rows win in the
                    // final, partially occupied class tile. Nineteen rows also
                    // leave the fourth row group partially occupied.
                    assert_eq!(
                        word(0),
                        if row % 2 == 0 {
                            63
                        } else {
                            (classes - 1) as u32
                        }
                    );
                    let score = if probability {
                        let high = if row % 2 == 0 { 2. } else { 1. };
                        let low = 3. - high;
                        1. / (high + low * (-4f32).exp() + (classes - 3) as f32 * (-6f32).exp())
                    } else {
                        6.
                    };
                    assert!((f32::from_bits(word(1)) - score).abs() <= 1e-6 + 1e-5 * score.abs());
                    assert_eq!(word(3), 0);
                }
                for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    bias[65] = invalid;
                    context
                        .upload_resident(&head.bias, bytemuck::cast_slice(&bias))
                        .unwrap();
                    let mut batch = Batch::new(context.clone()).unwrap();
                    let winners = head.project(&mut batch, &features, probability).unwrap();
                    let decoded = head
                        .decode(&mut batch, &winners, 1, rows, true, usize::MAX)
                        .unwrap();
                    let (actual, _) = batch.read(&decoded, (rows + 1) * 24, false).unwrap();
                    assert!(Recognition::parse(&actual, 1, rows, classes).is_err());
                }
                bias[65] = 0.;
                context
                    .upload_resident(&head.bias, bytemuck::cast_slice(&bias))
                    .unwrap();
            }
        }
    }

    // The probability tail remains byte-for-byte identical to the original
    // shader. Remove only the Logit early return to recover that baseline.
    fn legacy_projection_source(kernel: &str) -> String {
        let start = kernel
            .find("    // Logit mode already has everything")
            .unwrap();
        let end = start + kernel[start..].find("    sums[lane]=").unwrap();
        assert!(kernel[start..end].contains("if p(4u)==0u{"));
        assert!(kernel[start..end].contains("return;"));
        format!("{}{}", &kernel[..start], &kernel[end..])
    }

    #[test]
    #[ignore = "bounded physical GPU comparison: three shapes, one correctness warmup and three measured calls per version"]
    fn benchmark_gpu_logit_projection_reduction() {
        if cfg!(debug_assertions) {
            panic!("release build required");
        }
        let context = projection_context(false);
        let mut execution = [std::time::Duration::ZERO; 2];
        for (rows, inner, classes) in [(8, 80, 6906), (80, 80, 6906), (80, 192, 18710)] {
            let values = |n: usize| {
                (0..n)
                    .map(|i| ((i * 37 + 19) % 127) as f32 / 63. - 1.)
                    .collect::<Vec<_>>()
            };
            let (head, features) = projection_fixture(
                &context,
                [rows, inner, classes],
                GpuPrecision::F32,
                [
                    &values(rows * inner),
                    &values(inner * classes),
                    &values(classes),
                ],
            );
            let rows_per_group = if rows >= 16 && classes >= 4096 { 4 } else { 1 };
            let kernel = if rows_per_group == 4 {
                include_str!("heads4.wgsl")
            } else {
                include_str!("heads.wgsl")
            };
            let legacy = legacy_projection_source(kernel);
            let mut expected = None;
            for (version, kernel) in [legacy.as_str(), kernel].into_iter().enumerate() {
                let source = format!("{PARAMETERS}{}", kernel.replace("$INPUT", "f32"));
                let reduce = format!("{PARAMETERS}{}", include_str!("head_reduce.wgsl"));
                let tiles = classes.div_ceil(64);
                let words = [rows as u32, inner as u32, classes as u32, tiles as u32, 0];
                let mut wall_us = Vec::new();
                let mut device_us = Vec::new();
                for iteration in 0..4 {
                    assert!(
                        execution[version].as_secs_f64() < 60.,
                        "projection execution budget exhausted"
                    );
                    let start = std::time::Instant::now();
                    let mut batch = Batch::new(context.clone()).unwrap();
                    batch.profiling = true;
                    let partial = batch
                        .storage("projection comparison partial", rows * tiles * 16)
                        .unwrap();
                    let winners = batch
                        .storage("projection comparison winners", rows * 16)
                        .unwrap();
                    batch
                        .dispatch(
                            &format!("projection_comparison_{version}_{rows_per_group}"),
                            &source,
                            &[
                                (&features.buffer, true),
                                (&head.weights, true),
                                (&head.bias, true),
                                (&partial, false),
                            ],
                            &words,
                            [tiles as u32, rows.div_ceil(rows_per_group) as u32, 1],
                        )
                        .unwrap();
                    batch
                        .dispatch(
                            "projection_comparison_winners",
                            &reduce,
                            &[(&partial, true), (&winners, false)],
                            &words,
                            [(rows as u32).div_ceil(64), 1, 1],
                        )
                        .unwrap();
                    let (actual, stats) = batch.read(&winners, rows * 16, false).unwrap();
                    let elapsed = start.elapsed();
                    execution[version] += elapsed;
                    assert!(
                        execution[version].as_secs_f64() < 60.,
                        "projection execution budget exhausted"
                    );
                    if iteration == 0 {
                        if let Some(expected) = &expected {
                            assert_eq!(&actual, expected, "Logit winners changed");
                        } else {
                            expected = Some(actual);
                        }
                    } else {
                        wall_us.push(elapsed.as_secs_f64() * 1e6);
                        device_us.push(
                            stats
                                .profile
                                .and_then(|p| p.compute_ns())
                                .map(|ns| ns as f64 / 1000.),
                        );
                    }
                }
                eprintln!(
                    "PROJECTION {}",
                    serde_json::json!({
                        "shape": [rows, inner, classes], "rows_per_group": rows_per_group,
                        "version": if version == 0 { "legacy" } else { "logit_fast_path" },
                        "wall_us": wall_us, "device_us": device_us,
                        "cumulative_execution_seconds": execution[version].as_secs_f64(),
                        "warmup_calls": 1, "measured_calls": 3,
                    })
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a physical GPU"]
    fn gpu_resident_classifier_batches_preserve_rows_and_tails() {
        use crate::{image::PixelFormat, orientation::Classifier};
        let backend = match std::env::var("SIMD_OCR_GPU_BACKEND").as_deref() {
            Ok("dx12") => GpuBackendPreference::Dx12,
            Ok("metal") => GpuBackendPreference::Metal,
            _ => GpuBackendPreference::Vulkan,
        };
        let context = GpuContext::new_with_options(GpuContextOptions {
            backend,
            ..Default::default()
        })
        .unwrap();
        let cpu = crate::ocr::tests::synthetic_classifier_with_batch(None);
        let compiled = Arc::new(
            GpuCompiledModel::compile(cpu.program.model.clone(), context.clone()).unwrap(),
        );
        let mut reference = Classifier::new(cpu, ExecutionLimits::default()).unwrap();
        let mut classifier = Classification::new(compiled, ExecutionLimits::default()).unwrap();
        let pixels = (0..5)
            .map(|i| {
                (0..(13 + i * 2) * 9)
                    .flat_map(|p| {
                        let v = if i % 2 == 0 {
                            8 + (p % 16) as u8
                        } else {
                            224 + (p % 16) as u8
                        };
                        [v, v, v, 255]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let views = pixels
            .iter()
            .enumerate()
            .map(|(i, p)| ImageView::new(p, 13 + i * 2, 9, 0, PixelFormat::Bgra32).unwrap())
            .collect::<Vec<_>>();
        let expected = views
            .iter()
            .map(|&image| reference.classify(image).unwrap())
            .collect::<Vec<_>>();
        assert!(expected.iter().any(|c| c.label == 0));
        assert!(expected.iter().any(|c| c.label == 1));
        classifier.prepare_batch(4).unwrap();
        let mut batch = Batch::new(context.clone()).unwrap();
        let output_bytes = 5 * 16 + pixels.iter().map(Vec::len).sum::<usize>();
        let output = batch
            .storage("batched classifier and rotation", output_bytes)
            .unwrap();
        let mut offset = 5 * 16;
        for (group, inputs) in views.chunks(4).enumerate() {
            let images = inputs
                .iter()
                .map(|&image| Image::upload(&mut batch, image).unwrap())
                .collect::<Vec<_>>();
            let (winners, widths) = classifier.record_batch(&mut batch, &images).unwrap();
            batch.copy(
                &winners,
                0,
                &output,
                (group * 4 * 16) as u64,
                (inputs.len() * 16) as u64,
            );
            for (row, image) in images.iter().enumerate() {
                assert_eq!(widths[row], expected[group * 4 + row].resized_width);
                let oriented =
                    super::super::vision::orient_at(&mut batch, image, &winners, row, 0.5).unwrap();
                let bytes = image.width * image.height * 4;
                batch.copy(&oriented.buffer, 0, &output, offset as u64, bytes as u64);
                offset += bytes;
            }
        }
        let (actual, _) = batch.read(&output, output_bytes, false).unwrap();
        let mut offset = 5 * 16;
        for (i, expected) in expected.iter().enumerate() {
            let class = crate::orientation::parse_gpu_classification(
                &actual[i * 16..(i + 1) * 16],
                expected.resized_width,
            )
            .unwrap();
            assert_eq!(class.label, expected.label);
            assert!((class.score - expected.score).abs() <= 1e-3);
            let mut oriented = pixels[i].as_chunks::<4>().0.to_vec();
            if expected.label == 1 && expected.score > 0.5 {
                oriented.reverse();
            }
            assert_eq!(
                &actual[offset..offset + pixels[i].len()],
                oriented.as_flattened()
            );
            offset += pixels[i].len();
        }
        // A standalone call can reuse the classifier after batched work drains.
        let (actual, _) = classifier.run(views[1], false).unwrap();
        let actual =
            crate::orientation::parse_gpu_classification(&actual, expected[1].resized_width)
                .unwrap();
        assert_eq!(actual.label, expected[1].label);
        assert!((actual.score - expected[1].score).abs() <= 1e-3);

        // Fixed-batch models and tight activation budgets retain the original
        // one-image behavior; failed optional growth leaves the session usable.
        for fixed in [false, true] {
            let cpu = crate::ocr::tests::synthetic_classifier_with_batch(fixed.then_some(1));
            let compiled = Arc::new(
                GpuCompiledModel::compile(cpu.program.model.clone(), context.clone()).unwrap(),
            );
            let mut classifier = Classification::new(compiled, ExecutionLimits::default()).unwrap();
            if !fixed {
                classifier.session.limits.max_activation_bytes =
                    classifier.session.memory.total_bytes();
            }
            assert_eq!(classifier.prepare_batch(4).unwrap(), 1);
            let (actual, _) = classifier.run(views[1], false).unwrap();
            let actual =
                crate::orientation::parse_gpu_classification(&actual, expected[1].resized_width)
                    .unwrap();
            assert_eq!(actual.label, expected[1].label);
            assert!((actual.score - expected[1].score).abs() <= 1e-3);
        }
    }
    #[test]
    #[ignore = "requires a physical GPU"]
    fn gpu_resident_projection_ctc_contract() {
        let backend = match std::env::var("SIMD_OCR_GPU_BACKEND").as_deref() {
            Ok("dx12") => GpuBackendPreference::Dx12,
            Ok("metal") => GpuBackendPreference::Metal,
            _ => GpuBackendPreference::Vulkan,
        };
        let context = GpuContext::new_with_options(GpuContextOptions {
            backend,
            ..Default::default()
        })
        .unwrap();
        assert!(matches!(
            context.adapter_info().device_type,
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
        ));
        let classes = 130;
        let inner = 7;
        let weights = vec![0f32; classes * inner];
        let mut bias = vec![0f32; classes];
        bias[63] = 4.;
        bias[64] = 4.;
        let mut lengths = vec![1u32; classes];
        lengths[0] = 0;
        lengths[1] = 3;
        lengths[2] = 0;
        let buffer = |label, data: &[u8]| {
            context
                .create_buffer_init(
                    label,
                    data,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                )
                .unwrap()
        };
        let head = Head {
            owned_bytes: 0,
            metadata: ProjectionHead {
                feature: TensorId(0),
                feature_node: 0,
                weights: TensorId(0),
                bias: None,
                inner,
                classes,
            },
            weights: buffer("weights", bytemuck::cast_slice(&weights)),
            bias: buffer("bias", bytemuck::cast_slice(&bias)),
            lengths: buffer("lengths", bytemuck::cast_slice(&lengths)),
            context: context.clone(),
        };
        let input = Tensor::allocate(
            context.clone(),
            Shape::new(&[2, 3, inner]).unwrap(),
            GpuPrecision::F32,
        )
        .unwrap();
        for probability in [false, true] {
            let mut batch = Batch::new(context.clone()).unwrap();
            let winners = head.project(&mut batch, &input, probability).unwrap();
            let (data, stats) = batch.read(&winners, 6 * 16, false).unwrap();
            let expected = if probability {
                1.0 / (2.0 + 128.0 * (-4f32).exp())
            } else {
                4.
            };
            for row in data.as_chunks::<16>().0.iter() {
                assert_eq!(u32::from_le_bytes(row[..4].try_into().unwrap()), 63);
                assert!(
                    (f32::from_le_bytes(row[4..8].try_into().unwrap()) - expected).abs() < 1e-6
                );
            }
            assert_eq!(stats.output_readback_bytes, 96);
        }
        let steps = 131;
        let mut records = vec![0u32; 2 * steps * 4];
        // Repeats cross a workgroup scan tile; blanks reset repeats; Unicode and empty labels count correctly.
        for (start, end, class) in [(1, 4, 1), (60, 68, 2), (69, 71, 1), (130, 131, 129)] {
            for t in start..end {
                records[t * 4] = class;
                records[t * 4 + 1] = 0.75f32.to_bits();
            }
        }
        let winners = buffer("winners", bytemuck::cast_slice(&records));
        for capture in [false, true] {
            let budget = 7 + if capture {
                4 * std::mem::size_of::<crate::recognizer::CtcRun>()
            } else {
                0
            };
            for limit in [budget, budget - 1] {
                let mut batch = Batch::new(context.clone()).unwrap();
                let output = head
                    .decode(&mut batch, &winners, 2, steps, capture, limit)
                    .unwrap();
                let (data, _) = batch.read(&output, 2 * (steps + 1) * 24, false).unwrap();
                let decoded = Recognition::parse(&data, 2, steps, classes);
                if limit < budget {
                    assert!(
                        decoded
                            .err()
                            .unwrap()
                            .to_string()
                            .contains("max_result_bytes")
                    );
                    continue;
                }
                let decoded = decoded.unwrap();
                assert_eq!(decoded[0].text_bytes, 7);
                assert_eq!(decoded[0].score, 0.75);
                assert_eq!(
                    decoded[0]
                        .tokens
                        .iter()
                        .map(|t| (t.class, t.start, t.end))
                        .collect::<Vec<_>>(),
                    vec![(1, 1, 4), (2, 60, 68), (1, 69, 71), (129, 130, 131)]
                );
                assert!(decoded[1].tokens.is_empty());
                assert_eq!(decoded[1].score, 0.);
            }
        }
        let other = GpuContext::new_with_options(GpuContextOptions {
            backend,
            ..Default::default()
        })
        .unwrap();
        assert!(
            head.project(&mut Batch::new(other).unwrap(), &input, false)
                .is_err()
        );
    }
}
