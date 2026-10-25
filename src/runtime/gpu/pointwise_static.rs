//! Five qualified Half reductions; width and batch remain runtime geometry.
use super::*;
use operators::ShaderKind;

pub(super) const ENTRY: &str = "pointwise_mpp64";
pub(super) const WORKGROUP_SIZE: [u32; 3] = [128, 1, 1];

/// A finite cache-key domain, independent of model identity or ROI dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) enum Specialization {
    BiasGelu160,
    BiasGelu192,
    BiasGelu384,
    Adds320,
    Adds768,
}
impl Specialization {
    #[cfg(test)]
    pub(super) const ALL: [Self; 5] = [
        Self::BiasGelu160,
        Self::BiasGelu192,
        Self::BiasGelu384,
        Self::Adds320,
        Self::Adds768,
    ];

    pub(super) fn inner(self) -> u32 {
        match self {
            Self::BiasGelu160 => 160,
            Self::BiasGelu192 => 192,
            Self::BiasGelu384 => 384,
            Self::Adds320 => 320,
            Self::Adds768 => 768,
        }
    }

    #[cfg(test)]
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::BiasGelu160 => "PointwiseStaticBiasGelu160",
            Self::BiasGelu192 => "PointwiseStaticBiasGelu192",
            Self::BiasGelu384 => "PointwiseStaticBiasGelu384",
            Self::Adds320 => "PointwiseStaticAdds320",
            Self::Adds768 => "PointwiseStaticAdds768",
        }
    }

    pub(super) fn is_bias_gelu(self) -> bool {
        matches!(
            self,
            Self::BiasGelu160 | Self::BiasGelu192 | Self::BiasGelu384
        )
    }

    pub(super) fn for_recipe(
        kind: ShaderKind,
        storage: [GpuPrecision; 6],
        words: &[u32; 128],
    ) -> Option<Self> {
        if half_native::fixed_pointwise_bias_gelu(kind, storage, words) {
            match words[3] {
                160 => Some(Self::BiasGelu160),
                192 => Some(Self::BiasGelu192),
                384 => Some(Self::BiasGelu384),
                _ => None,
            }
        } else if half_native::fixed_pointwise_adds(kind, storage, words) {
            match words[3] {
                320 => Some(Self::Adds320),
                768 => Some(Self::Adds768),
                _ => None,
            }
        } else {
            None
        }
    }
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    RuntimeError::with_kind(ErrorKind::Unsupported, message)
}

/// Preserve the qualified Fixed source, including conditional bias
/// initialization, Float arithmetic, complete runtime epilogue and final Half
/// store. Only descriptor/interior K extents and a uniform mismatch guard change.
pub(super) fn source(
    specialization: Specialization,
    storage: [GpuPrecision; 6],
) -> RuntimeResult<String> {
    if storage != [GpuPrecision::MixedF16; 6] {
        return Err(unsupported(
            "static pointwise reduction requires six Half bindings",
        ));
    }
    let mut source = if specialization.is_bias_gelu() {
        half_native::pointwise_bias_gelu_source()
    } else {
        half_native::pointwise_adds_source()
    };
    let inner = specialization.inner();
    for (before, after) in [
        (
            "if (width != 32u) {",
            format!("if (width != 32u || inner != {inner}u) {{"),
        ),
        (
            "64, 64, static_cast<int>(dynamic_extent), false, false, false,",
            format!("64, 64, {inner}, false, false, false,"),
        ),
        (
            "using WA = tensor<device half, extents<int, dynamic_extent, 64>, tensor_inline>;",
            format!("using WA = tensor<device half, extents<int, {inner}, 64>, tensor_inline>;"),
        ),
        (
            "using XB = tensor<device half, extents<int, 64, dynamic_extent>, tensor_inline>;",
            format!("using XB = tensor<device half, extents<int, 64, {inner}>, tensor_inline>;"),
        ),
        (
            "extents<int, dynamic_extent, 64>(int(inner)),",
            format!("extents<int, {inner}, 64>(),"),
        ),
        (
            "extents<int, 64, dynamic_extent>(int(inner)),",
            format!("extents<int, 64, {inner}>(),"),
        ),
    ] {
        if source.matches(before).count() != 1 {
            return Err(unsupported(
                "static pointwise source no longer matches the qualified kernel",
            ));
        }
        source = source.replacen(before, &after, 1);
    }
    Ok(source)
}

#[cfg(test)]
#[path = "pointwise_static_tests.rs"]
mod tests;
