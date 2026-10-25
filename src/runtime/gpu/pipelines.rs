//! Typed Vulkan pipeline cache. Platform-specific kernel admission stays here.
use super::*;
use operators::{Recipe, ShaderKind};
use std::collections::HashMap;
pub(super) struct Pipelines {
    owner: Arc<Ledger>,
    variants: Mutex<HashMap<KernelKey, Arc<executor::NativePipeline>>>,
}
impl Pipelines {
    pub fn shared(context: &GpuContext) -> RuntimeResult<Arc<Self>> {
        context.check_alive()?;
        let mut cache = context
            .pipelines
            .lock()
            .map_err(|_| RuntimeError::new("GPU pipeline cache poisoned"))?;
        if let Some(p) = &*cache {
            p.ensure_context(context)?;
            return Ok(p.clone());
        }
        let p = Arc::new(Self {
            owner: context.ledger.clone(),
            variants: Mutex::new(HashMap::new()),
        });
        *cache = Some(p.clone());
        Ok(p)
    }
    fn ensure_context(&self, context: &GpuContext) -> RuntimeResult<()> {
        if !Arc::ptr_eq(&self.owner, &context.ledger) {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "foreign pipeline cache",
            ));
        }
        context.check_alive()
    }
    pub fn get(
        &self,
        context: &GpuContext,
        kind: ShaderKind,
        storage: [GpuPrecision; 6],
    ) -> RuntimeResult<Arc<executor::NativePipeline>> {
        self.ensure_context(context)?;
        let id = match kind {
            ShaderKind::Attention => KernelId::Attention,
            ShaderKind::AttentionViews => KernelId::AttentionViews,
            ShaderKind::SqueezeExcite => KernelId::SqueezeExcite,
            ShaderKind::SmallProjection => KernelId::SmallProjection,
            ShaderKind::Transposed => KernelId::Transposed,
            ShaderKind::Convolution => KernelId::Convolution,
            ShaderKind::General => KernelId::General,
            ShaderKind::PackNhwc => KernelId::PackNhwc,
            ShaderKind::MatMul16 => KernelId::Matmul16,
            ShaderKind::MatMul32 => KernelId::Matmul32,
            ShaderKind::Conv16 => KernelId::Conv16,
            ShaderKind::Conv32 => KernelId::Conv32,
            ShaderKind::Depthwise3 => KernelId::Depthwise3,
            ShaderKind::Depthwise5 => KernelId::Depthwise5,
            ShaderKind::Reduce => KernelId::Reduce,
            ShaderKind::Softmax => KernelId::Softmax,
            ShaderKind::LayerNorm => KernelId::LayerNorm,
            ShaderKind::ReducePartial => KernelId::ReducePartial,
            ShaderKind::ReduceFinish => KernelId::ReduceFinish,
            ShaderKind::SoftmaxPartial => KernelId::SoftmaxPartial,
            ShaderKind::SoftmaxFinish => KernelId::SoftmaxFinish,
            _ => {
                return Err(RuntimeError::with_kind(
                    ErrorKind::Unsupported,
                    "kernel is not available on this device",
                ));
            }
        };
        let precision = storage.iter().enumerate().fold(0, |bits, (i, p)| {
            bits | ((u8::from(*p == GpuPrecision::MixedF16)) << i)
        });
        let key = KernelKey {
            id,
            precision: precision & id.precision_mask(),
        };
        let mut cache = self
            .variants
            .lock()
            .map_err(|_| RuntimeError::new("pipeline variants poisoned"))?;
        if let Some(p) = cache.get(&key) {
            return Ok(p.clone());
        }
        let pipeline = executor::compile(context, key)?;
        cache.insert(key, pipeline.clone());
        Ok(pipeline)
    }
    pub fn get_for_recipe(
        &self,
        context: &GpuContext,
        kind: ShaderKind,
        storage: [GpuPrecision; 6],
        _: &[u32; 128],
    ) -> RuntimeResult<Arc<executor::NativePipeline>> {
        self.get(context, kind, storage)
    }
    pub fn supports_matrix(&self, context: &GpuContext, _: &Recipe) -> RuntimeResult<bool> {
        self.ensure_context(context)?;
        Ok(false)
    }
    pub fn supports_half(&self, context: &GpuContext, kind: ShaderKind) -> RuntimeResult<bool> {
        self.ensure_context(context)?;
        Ok(context.backend.supports_half() && kind.is_portable())
    }
}
