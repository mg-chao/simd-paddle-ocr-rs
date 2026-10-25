//! Metal pipeline cache and optional MSL specialization admission.
use super::*;
use operators::ShaderKind;
use std::collections::HashMap;
const MAX_SPATIAL_SPECIALIZATIONS: usize = 256;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ShaderSpecialization {
    Generic,
    PointwiseKnownEpilogue,
    PointwiseFixedBiasGelu,
    PointwiseFixedBiasResidualGelu,
    PointwiseFixedAdds,
    PointwiseStatic(pointwise_static::Specialization),
    DepthwiseChannelBias,
    AttentionSimd,
    SpatialInput([u32; 3]),
    StagedChannels(conv_mpp_staged::Specialization),
}
type PipelineKey = (ShaderKind, [GpuPrecision; 6], ShaderSpecialization);
pub(super) struct Pipelines {
    owner: Arc<Ledger>,
    variants: Mutex<HashMap<PipelineKey, Arc<executor::NativePipeline>>>,
    spatial_unavailable: Mutex<HashMap<PipelineKey, RuntimeError>>,
}
impl Pipelines {
    pub fn shared(context: &GpuContext) -> RuntimeResult<Arc<Self>> {
        context.check_alive()?;
        let mut cache = context
            .pipelines
            .lock()
            .map_err(|_| RuntimeError::new("GPU pipeline cache poisoned"))?;
        if let Some(value) = &*cache {
            value.ensure_context(context)?;
            return Ok(value.clone());
        }
        let result = Arc::new(Self {
            owner: context.ledger.clone(),
            variants: Mutex::new(HashMap::new()),
            spatial_unavailable: Mutex::new(HashMap::new()),
        });
        *cache = Some(result.clone());
        Ok(result)
    }
    pub fn get(
        &self,
        context: &GpuContext,
        kind: ShaderKind,
        storage: [GpuPrecision; 6],
    ) -> RuntimeResult<Arc<executor::NativePipeline>> {
        self.ensure_context(context)?;
        self.get_specialized(context, kind, storage, ShaderSpecialization::Generic)
    }
    pub fn get_for_recipe(
        &self,
        context: &GpuContext,
        kind: ShaderKind,
        storage: [GpuPrecision; 6],
        words: &[u32; 128],
    ) -> RuntimeResult<Arc<executor::NativePipeline>> {
        self.ensure_context(context)?;
        if attention_packed::native::eligible(kind, storage, words, &context.limits) {
            let specialization = ShaderSpecialization::AttentionSimd;
            let key = (kind, storage, specialization);
            context.check_alive()?;
            if !self.rejected(&key)? {
                match self.get_specialized(context, kind, storage, specialization) {
                    Ok(pipeline) => return Ok(pipeline),
                    Err(error)
                        if matches!(
                            error.kind(),
                            ErrorKind::Unsupported | ErrorKind::GpuValidation
                        ) =>
                    {
                        context.check_alive()?;
                        self.spatial_unavailable
                            .lock()
                            .map_err(|_| RuntimeError::new("GPU capability cache poisoned"))?
                            .insert(key, error);
                    }
                    Err(error) => return Err(error),
                }
            }
            return self.get(context, kind, storage);
        }
        let specialization = if conv_mpp_staged::supports_kind(kind) {
            ShaderSpecialization::StagedChannels(
                conv_mpp_staged::Specialization::for_shader(kind, words).ok_or_else(|| {
                    RuntimeError::with_kind(
                        ErrorKind::Unsupported,
                        "invalid staged convolution channels",
                    )
                })?,
            )
        } else if super::conv_mpp::Kernel::from_shader(kind).is_some() {
            ShaderSpecialization::SpatialInput([words[3], words[5], words[6]])
        } else if let Some(specialization) =
            pointwise_static::Specialization::for_recipe(kind, storage, words)
        {
            ShaderSpecialization::PointwiseStatic(specialization)
        } else if half_native::fixed_pointwise_bias_gelu(kind, storage, words) {
            ShaderSpecialization::PointwiseFixedBiasGelu
        } else if half_native::fixed_pointwise_epilogue(kind, storage, words) {
            ShaderSpecialization::PointwiseFixedBiasResidualGelu
        } else if half_native::fixed_pointwise_adds(kind, storage, words) {
            ShaderSpecialization::PointwiseFixedAdds
        } else if half_native::known_pointwise_epilogue(kind, storage, words) {
            ShaderSpecialization::PointwiseKnownEpilogue
        } else if half_native::fixed_depthwise_bias(kind, storage, words) {
            ShaderSpecialization::DepthwiseChannelBias
        } else {
            ShaderSpecialization::Generic
        };
        let static_fallback =
            if let ShaderSpecialization::PointwiseStatic(reduction) = specialization {
                Some([
                    specialization,
                    if reduction.is_bias_gelu() {
                        ShaderSpecialization::PointwiseFixedBiasGelu
                    } else {
                        ShaderSpecialization::PointwiseFixedAdds
                    },
                    ShaderSpecialization::PointwiseKnownEpilogue,
                ])
            } else {
                None
            };
        let optional = match specialization {
            ShaderSpecialization::PointwiseStatic(_) => {
                static_fallback.as_ref().unwrap().as_slice()
            }
            ShaderSpecialization::PointwiseFixedBiasGelu
                if half_native::known_pointwise_epilogue(kind, storage, words) =>
            {
                &[
                    ShaderSpecialization::PointwiseFixedBiasGelu,
                    ShaderSpecialization::PointwiseKnownEpilogue,
                ][..]
            }
            ShaderSpecialization::PointwiseFixedBiasGelu => {
                // K48 only qualified the exact bias/GELU source. Its previous
                // production path was Generic; do not introduce Known here.
                &[ShaderSpecialization::PointwiseFixedBiasGelu][..]
            }
            ShaderSpecialization::PointwiseFixedBiasResidualGelu => &[
                ShaderSpecialization::PointwiseFixedBiasResidualGelu,
                ShaderSpecialization::PointwiseKnownEpilogue,
            ][..],
            ShaderSpecialization::PointwiseFixedAdds => &[
                ShaderSpecialization::PointwiseFixedAdds,
                ShaderSpecialization::PointwiseKnownEpilogue,
            ][..],
            ShaderSpecialization::PointwiseKnownEpilogue => {
                &[ShaderSpecialization::PointwiseKnownEpilogue][..]
            }
            ShaderSpecialization::DepthwiseChannelBias => {
                &[ShaderSpecialization::DepthwiseChannelBias][..]
            }
            _ => &[],
        };
        if !optional.is_empty() {
            context.check_alive()?;
            for &candidate in optional {
                let key = (kind, storage, candidate);
                if self.rejected(&key)? {
                    continue;
                }
                match self.get_specialized(context, kind, storage, candidate) {
                    Ok(pipeline) => return Ok(pipeline),
                    Err(error)
                        if matches!(
                            error.kind(),
                            ErrorKind::Unsupported | ErrorKind::GpuValidation
                        ) =>
                    {
                        context.check_alive()?;
                        self.spatial_unavailable
                            .lock()
                            .map_err(|_| RuntimeError::new("GPU matrix capability cache poisoned"))?
                            .insert(key, error);
                    }
                    Err(error) => return Err(error),
                }
            }
            // Optional source rejection never poisons or changes Generic.
            return self.get(context, kind, storage);
        }
        self.get_specialized(context, kind, storage, specialization)
    }

    /// Qualify a geometry before changing its operand layout. Unsupported
    /// native compilation keeps the generic native schedule and is cached separately.
    pub fn supports_matrix(
        &self,
        context: &GpuContext,
        recipe: &operators::Recipe,
    ) -> RuntimeResult<bool> {
        self.ensure_context(context)?;
        let operators::RecipeKind::Compute(kind) = recipe.kind else {
            return Ok(false);
        };
        let mut storage = [GpuPrecision::F32; 6];
        if recipe.words[38] == 1 {
            storage = [GpuPrecision::MixedF16; 6];
        } else if recipe.words[37] == 1 {
            storage[0] = GpuPrecision::MixedF16;
            storage[1] = GpuPrecision::MixedF16;
        }
        let specialization = if conv_mpp_staged::supports_kind(kind) {
            let Some(channels) = conv_mpp_staged::Specialization::for_shader(kind, &recipe.words)
            else {
                return Ok(false);
            };
            ShaderSpecialization::StagedChannels(channels)
        } else if super::conv_mpp::Kernel::from_shader(kind).is_some() {
            ShaderSpecialization::SpatialInput([recipe.words[3], recipe.words[5], recipe.words[6]])
        } else if matches!(
            kind,
            ShaderKind::PointwiseF16
                | ShaderKind::LinearNative
                | ShaderKind::LinearMpp
                | ShaderKind::TailProjection
        ) {
            ShaderSpecialization::Generic
        } else {
            return Ok(false);
        };
        let key = (kind, storage, specialization);
        if self.rejected(&key)? {
            return Ok(false);
        }
        match self.get_specialized(context, kind, storage, specialization) {
            Ok(_) => Ok(true),
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::Unsupported | ErrorKind::GpuValidation
                ) =>
            {
                context.check_alive()?;
                let mut rejected = self
                    .spatial_unavailable
                    .lock()
                    .map_err(|_| RuntimeError::new("GPU matrix capability cache poisoned"))?;
                if rejected.len() < MAX_SPATIAL_SPECIALIZATIONS {
                    rejected.insert(key, error);
                }
                Ok(false)
            }
            Err(error) => Err(error),
        }
    }

    pub fn supports_half(&self, context: &GpuContext, kind: ShaderKind) -> RuntimeResult<bool> {
        self.ensure_context(context)?;
        let key = (
            kind,
            [GpuPrecision::MixedF16; 6],
            ShaderSpecialization::Generic,
        );
        context.check_alive()?;
        if self.rejected(&key)? {
            return Ok(false);
        }
        match self.get(context, kind, [GpuPrecision::MixedF16; 6]) {
            Ok(_) => Ok(true),
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::Unsupported | ErrorKind::GpuValidation
                ) =>
            {
                context.check_alive()?;
                let mut rejected = self
                    .spatial_unavailable
                    .lock()
                    .map_err(|_| RuntimeError::new("GPU matrix capability cache poisoned"))?;
                if rejected.len() < MAX_SPATIAL_SPECIALIZATIONS {
                    rejected.insert(key, error);
                }
                Ok(false)
            }
            Err(error) => Err(error),
        }
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn reject_pointwise_static_for_test(
        &self,
        specialization: pointwise_static::Specialization,
        error: RuntimeError,
    ) {
        self.spatial_unavailable.lock().unwrap().insert(
            (
                ShaderKind::PointwiseNative64,
                [GpuPrecision::MixedF16; 6],
                ShaderSpecialization::PointwiseStatic(specialization),
            ),
            error,
        );
    }

    #[cfg(test)]
    pub(super) fn reject_depthwise7_for_test(&self, error: RuntimeError) {
        self.spatial_unavailable.lock().unwrap().insert(
            (
                ShaderKind::DepthwiseNative7,
                [GpuPrecision::MixedF16; 6],
                ShaderSpecialization::Generic,
            ),
            error,
        );
    }

    #[cfg(test)]
    pub(super) fn reject_linear_mpp_for_test(&self, error: RuntimeError) {
        self.spatial_unavailable.lock().unwrap().insert(
            (
                ShaderKind::LinearMpp,
                [GpuPrecision::MixedF16; 6],
                ShaderSpecialization::Generic,
            ),
            error,
        );
    }

    #[cfg(test)]
    pub(super) fn reject_tail_projection_for_test(&self, error: RuntimeError) {
        self.spatial_unavailable.lock().unwrap().insert(
            (
                ShaderKind::TailProjection,
                [GpuPrecision::MixedF16; 6],
                ShaderSpecialization::Generic,
            ),
            error,
        );
    }

    fn ensure_context(&self, context: &GpuContext) -> RuntimeResult<()> {
        if !Arc::ptr_eq(&self.owner, &context.ledger) {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "native pipeline cache belongs to another GPU context",
            ));
        }
        context.check_alive()
    }

    fn rejected(&self, key: &PipelineKey) -> RuntimeResult<bool> {
        let errors = self
            .spatial_unavailable
            .lock()
            .map_err(|_| RuntimeError::new("GPU shader capability cache poisoned"))?;
        match errors.get(key) {
            Some(error)
                if matches!(
                    error.kind(),
                    ErrorKind::Unsupported | ErrorKind::GpuValidation
                ) =>
            {
                Ok(true)
            }
            Some(error) => Err(error.clone()),
            None => Ok(false),
        }
    }

    fn get_specialized(
        &self,
        context: &GpuContext,
        kind: ShaderKind,
        storage: [GpuPrecision; 6],
        specialization: ShaderSpecialization,
    ) -> RuntimeResult<Arc<executor::NativePipeline>> {
        self.ensure_context(context)?;

        let mut variants = self
            .variants
            .lock()
            .map_err(|_| RuntimeError::new("GPU pipeline variants poisoned"))?;
        let key = (kind, storage, specialization);
        if let Some(pipeline) = variants.get(&key) {
            return Ok(pipeline.clone());
        }
        if matches!(
            specialization,
            ShaderSpecialization::SpatialInput(_) | ShaderSpecialization::StagedChannels(_)
        ) && variants
            .keys()
            .filter(|(_, _, s)| {
                matches!(
                    s,
                    ShaderSpecialization::SpatialInput(_) | ShaderSpecialization::StagedChannels(_)
                )
            })
            .count()
            >= MAX_SPATIAL_SPECIALIZATIONS
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::Unsupported,
                "native convolution specialization cache is full",
            ));
        }
        let pipeline = if specialization == ShaderSpecialization::AttentionSimd {
            attention_packed::native::compile(context)?
        } else if conv_mpp_staged::supports_kind(kind) {
            let ShaderSpecialization::StagedChannels(channels) = specialization else {
                return Err(RuntimeError::with_kind(
                    ErrorKind::Unsupported,
                    "staged convolution requires explicit channels",
                ));
            };
            let source = conv_mpp_staged::source(channels, storage)?;
            executor::compile_msl(
                context,
                &source,
                channels.entry(),
                conv_mpp_staged::WORKGROUP,
            )?
        } else if let Some(kernel) = conv_mpp::Kernel::from_shader(kind) {
            let ShaderSpecialization::SpatialInput(dimensions) = specialization else {
                return Err(RuntimeError::with_kind(
                    ErrorKind::Unsupported,
                    "native convolution requires explicit source dimensions",
                ));
            };
            let source = conv_mpp::source_storage(kernel, dimensions, storage)?;
            executor::compile_msl(context, &source, kernel.entry(), [128, 1, 1])?
        } else if storage == [GpuPrecision::MixedF16; 6] && half_native::supports_kind(kind) {
            let (source, entry, groups) = match specialization {
                ShaderSpecialization::PointwiseStatic(reduction) => (
                    pointwise_static::source(reduction, storage)?,
                    pointwise_static::ENTRY,
                    pointwise_static::WORKGROUP_SIZE,
                ),
                ShaderSpecialization::PointwiseFixedBiasGelu => (
                    half_native::pointwise_bias_gelu_source(),
                    "pointwise_mpp64",
                    [128, 1, 1],
                ),
                ShaderSpecialization::PointwiseFixedBiasResidualGelu => (
                    half_native::pointwise_fixed_epilogue_source(),
                    "pointwise_mpp64",
                    [128, 1, 1],
                ),
                ShaderSpecialization::PointwiseFixedAdds => (
                    half_native::pointwise_adds_source(),
                    "pointwise_mpp64",
                    [128, 1, 1],
                ),
                ShaderSpecialization::PointwiseKnownEpilogue => (
                    half_native::pointwise_epilogue_source(),
                    "pointwise_mpp64",
                    [128, 1, 1],
                ),
                ShaderSpecialization::DepthwiseChannelBias => (
                    half_native::depthwise_bias_source(),
                    "depthwise_native",
                    [64, 1, 1],
                ),
                _ => half_native::source(kind)?,
            };
            executor::compile_msl(context, &source, entry, groups)?
        } else if kind == ShaderKind::PointwiseF16 {
            if storage
                != [
                    GpuPrecision::MixedF16,
                    GpuPrecision::MixedF16,
                    GpuPrecision::F32,
                    GpuPrecision::F32,
                    GpuPrecision::F32,
                    GpuPrecision::F32,
                ]
            {
                return Err(RuntimeError::with_kind(
                    ErrorKind::Unsupported,
                    "pointwise F16 requires Half operands and F32 results",
                ));
            }
            executor::compile_msl(
                context,
                &pointwise_f16::source(),
                "pointwise_f16",
                [128, 1, 1],
            )?
        } else if kind == ShaderKind::LinearNative {
            executor::compile_msl(
                context,
                &linear_native::source(storage)?,
                linear_native::ENTRY,
                linear_native::WORKGROUP_SIZE,
            )?
        } else if kind == ShaderKind::LinearMpp {
            executor::compile_msl(
                context,
                &linear_mpp::source(storage)?,
                linear_mpp::ENTRY,
                linear_mpp::WORKGROUP_SIZE,
            )?
        } else if kind == ShaderKind::TailProjection {
            executor::compile_msl(
                context,
                &tail_projection::source(storage)?,
                tail_projection::ENTRY,
                tail_projection::WORKGROUP,
            )?
        } else if half_native::supports_kind(kind) {
            return Err(RuntimeError::with_kind(
                ErrorKind::Unsupported,
                "native kernel requires canonical Half storage",
            ));
        } else {
            executor::compile_wgsl(context, &wgsl_source(kind, storage), "main", [0; 3])?
        };
        variants.insert(key, pipeline.clone());
        Ok(pipeline)
    }
}
fn wgsl_source(kind: ShaderKind, storage: [GpuPrecision; 6]) -> String {
    let mut source = format!("{}\n{}", kind.common_source(), kind.source()).replace(
        "$ENABLE",
        if storage.contains(&GpuPrecision::MixedF16) {
            "enable f16;"
        } else {
            ""
        },
    );
    source = source.replace(
        "$HALF_OUTPUT",
        if storage[5] == GpuPrecision::MixedF16 {
            "true"
        } else {
            "false"
        },
    );
    for (name, precision) in ["A", "B", "C", "D", "E", "O"].into_iter().zip(storage) {
        source = source.replace(
            &format!("${name}"),
            if precision == GpuPrecision::MixedF16 {
                "f16"
            } else {
                "f32"
            },
        );
    }
    source
}

#[cfg(test)]
#[path = "pipelines_tests.rs"]
mod tests;
