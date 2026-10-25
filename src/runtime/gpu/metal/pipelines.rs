use super::*;
use operators::{Recipe, RecipeKind, ShaderKind};
use std::collections::HashMap;

type Key = (ShaderKind, [GpuPrecision; 6], [u32; 3]);
pub(crate) struct Pipelines {
    owner: Arc<Ledger>,
    variants: Mutex<HashMap<Key, Arc<execution::NativePipeline>>>,
}
impl Pipelines {
    pub(crate) fn shared(context: &GpuContext) -> RuntimeResult<Arc<Self>> {
        let mut cache = context
            .pipelines
            .lock()
            .map_err(|_| RuntimeError::new("Metal pipeline cache poisoned"))?;
        Ok(cache
            .get_or_insert_with(|| {
                Arc::new(Self {
                    owner: context.ledger.clone(),
                    variants: Mutex::new(HashMap::new()),
                })
            })
            .clone())
    }
    pub(crate) fn get(
        &self,
        context: &GpuContext,
        kind: ShaderKind,
        storage: [GpuPrecision; 6],
    ) -> RuntimeResult<Arc<execution::NativePipeline>> {
        self.get_for_recipe(context, kind, storage, &[0; 128])
    }
    pub(crate) fn get_for_recipe(
        &self,
        context: &GpuContext,
        kind: ShaderKind,
        storage: [GpuPrecision; 6],
        words: &[u32; 128],
    ) -> RuntimeResult<Arc<execution::NativePipeline>> {
        if !Arc::ptr_eq(&self.owner, &context.ledger) {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "foreign Metal pipeline cache",
            ));
        }
        context.check_alive()?;
        let dimensions = if conv_mpp::Kernel::from_shader(kind).is_some() {
            [words[3], words[5], words[6]]
        } else if conv_mpp_staged::supports_kind(kind) {
            [words[3], 0, 0]
        } else {
            [0; 3]
        };
        let key = (kind, storage, dimensions);
        let mut cache = self
            .variants
            .lock()
            .map_err(|_| RuntimeError::new("Metal pipeline cache poisoned"))?;
        if let Some(pipeline) = cache.get(&key) {
            return Ok(pipeline.clone());
        }
        let pipeline = shaders::neural(context, kind, storage, words)?;
        if cache.len() < 512 {
            cache.insert(key, pipeline.clone());
        }
        Ok(pipeline)
    }
    pub(crate) fn supports_matrix(
        &self,
        context: &GpuContext,
        recipe: &Recipe,
    ) -> RuntimeResult<bool> {
        let RecipeKind::Compute(kind) = recipe.kind else {
            return Ok(false);
        };
        if !context
            .backend
            .owner
            .raw
            .supportsFamily(MTLGPUFamily::Apple7)
        {
            return Ok(false);
        }
        let mut storage = [GpuPrecision::F32; 6];
        if recipe.words[38] == 1 {
            storage = [GpuPrecision::MixedF16; 6];
        } else if recipe.words[37] == 1 {
            storage[0] = GpuPrecision::MixedF16;
            storage[1] = GpuPrecision::MixedF16;
        }
        self.get_for_recipe(context, kind, storage, &recipe.words)?;
        Ok(true)
    }
    pub(crate) fn supports_half(
        &self,
        context: &GpuContext,
        kind: ShaderKind,
    ) -> RuntimeResult<bool> {
        if !context
            .backend
            .owner
            .raw
            .supportsFamily(MTLGPUFamily::Apple7)
        {
            return Ok(false);
        }
        self.get(context, kind, [GpuPrecision::MixedF16; 6])?;
        Ok(true)
    }
}
