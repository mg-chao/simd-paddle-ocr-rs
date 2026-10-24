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
        let window = image.width.min(image.height.saturating_mul(4));
        let width = (80u64 * window as u64)
            .div_ceil(image.height as u64)
            .min(160) as usize;
        preprocess(
            batch,
            image,
            &self.session.resident_input(),
            [window, image.height],
            width,
            false,
        )?;
        let features = batch.graph(&mut self.session, "classifier")?;
        Ok((self.head.project(batch, &features, true)?, width))
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
