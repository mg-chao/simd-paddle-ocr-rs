//! MPP spatial convolution over packed NHWC/HWIO operands with FP32 accumulation.
//!
//! Keep input geometry in the pipeline cache key: the Metal descriptor must
//! contain actual source dimensions for correct zero padding at image borders.
//! Admission and compilation happen before the physical schedule packs inputs.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Kernel {
    K2S1,
    K3S1,
    K3S2,
    K3S21,
    K3S12,
    K7S1,
    K7S2,
}

impl Kernel {
    pub(super) fn shader(self) -> operators::ShaderKind {
        use operators::ShaderKind;
        match self {
            Self::K2S1 => ShaderKind::ConvMpp2s1,
            Self::K3S1 => ShaderKind::ConvMpp3s1,
            Self::K3S2 => ShaderKind::ConvMpp3s2,
            Self::K3S21 => ShaderKind::ConvMpp3s21,
            Self::K3S12 => ShaderKind::ConvMpp3s12,
            Self::K7S1 => ShaderKind::ConvMpp7s1,
            Self::K7S2 => ShaderKind::ConvMpp7s2,
        }
    }

    pub(super) fn from_shader(shader: operators::ShaderKind) -> Option<Self> {
        use operators::ShaderKind;
        match shader {
            ShaderKind::ConvMpp2s1 => Some(Self::K2S1),
            ShaderKind::ConvMpp3s1 => Some(Self::K3S1),
            ShaderKind::ConvMpp3s2 => Some(Self::K3S2),
            ShaderKind::ConvMpp3s21 => Some(Self::K3S21),
            ShaderKind::ConvMpp3s12 => Some(Self::K3S12),
            ShaderKind::ConvMpp7s1 => Some(Self::K7S1),
            ShaderKind::ConvMpp7s2 => Some(Self::K7S2),
            _ => None,
        }
    }

    pub(super) fn from_words(words: &[u32; 128]) -> Option<Self> {
        if words[11] != words[12] {
            return None;
        }
        match (words[11], words[13], words[14]) {
            (2, 1, 1) => Some(Self::K2S1),
            (3, 1, 1) => Some(Self::K3S1),
            (3, 2, 2) => Some(Self::K3S2),
            (3, 2, 1) => Some(Self::K3S21),
            (3, 1, 2) => Some(Self::K3S12),
            (7, 1, 1) => Some(Self::K7S1),
            (7, 2, 2) => Some(Self::K7S2),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Specialization {
    pub(super) kernel: Kernel,
    /// [input channels, input height, input width].
    pub(super) input: [u32; 3],
}

impl Specialization {
    pub(super) fn from_words(words: &[u32; 128]) -> Option<Self> {
        let kernel = Kernel::from_words(words)?;
        let input = [words[3], words[5], words[6]];
        valid_input(input).then_some(Self { kernel, input })
    }
}

fn valid_input(input: [u32; 3]) -> bool {
    input.iter().all(|&n| n > 0 && n <= i32::MAX as u32)
        && input
            .into_iter()
            .try_fold(1u32, u32::checked_mul)
            .is_some_and(|product| product <= i32::MAX as u32)
}

/// Admit canonical logical operands before physical lowering. The caller must
/// first compile the geometry-specific pipeline, then pack NHWC activations and
/// bind the shared zero-padded HWIO constant cache for the returned kernel.
pub(super) fn prepare_recipe(
    recipe: &mut operators::Recipe,
    descriptors: &[GpuTensorDescriptor],
    limits: &crate::runtime::gpu::GpuLimits,
) -> RuntimeResult<Option<Kernel>> {
    use operators::{RecipeKind, ShaderKind};
    if !matches!(
        recipe.kind,
        RecipeKind::Compute(ShaderKind::Convolution | ShaderKind::Conv16 | ShaderKind::Conv32)
    ) {
        return Ok(None);
    }
    let Some(specialization) = Specialization::from_words(&recipe.words) else {
        return Ok(None);
    };
    let Some(storage) = recipe
        .inputs
        .first()
        .and_then(|id| descriptors.get(id.0))
        .map(|descriptor| descriptor.precision)
    else {
        return Ok(None);
    };
    let words = &recipe.words;
    if words[2] == 0
        || words[2] > i32::MAX as u32
        || words[3] == 0
        || words[4] < 8
        || words[9] != words[3]
        || words[10] != words[4]
        || words[15] != 1
        || words[16] != 1
        || words[52] != 0
        || (3..=18).any(|i| words[i] > i32::MAX as u32)
        || words[5..=8].contains(&0)
        || !recipe
            .inputs
            .iter()
            .chain(std::iter::once(&recipe.output))
            .all(|id| {
                descriptors.get(id.0).is_some_and(|descriptor| {
                    descriptor.precision == storage && descriptor.layout == GpuLayout::Canonical
                })
            })
    {
        return Ok(None);
    }
    let Some(padded_channels) = words[4].checked_add(31).map(|n| n / 32 * 32) else {
        return Ok(None);
    };
    // Tensor strides and the absolute center coordinates are signed Metal ints.
    // Bounds include batch address products, weights, and final padded tile lanes.
    for factors in [
        [words[2], words[3], words[5], words[6]],
        [words[2], words[4], words[7], words[8]],
        [padded_channels, words[3], words[11], words[12]],
    ] {
        if factors
            .into_iter()
            .try_fold(1u32, u32::checked_mul)
            .is_none_or(|product| product > i32::MAX as u32)
        {
            return Ok(None);
        }
    }
    for (extent, tile, stride, pad, kernel) in [
        (words[7], 4u32, words[13], words[17], words[11]),
        (words[8], 8u32, words[14], words[18], words[12]),
    ] {
        if extent
            .checked_add(tile - 1)
            .and_then(|n| n.checked_mul(stride))
            .and_then(|n| n.checked_add(kernel))
            .is_none_or(|n| n > i32::MAX as u32)
            || pad > i32::MAX as u32 - kernel / 2
        {
            return Ok(None);
        }
    }
    let channel_tiles = words[4].div_ceil(32);
    let spatial_tiles = words[7]
        .div_ceil(4)
        .checked_mul(words[8].div_ceil(8))
        .ok_or_else(|| resource("MPP convolution spatial tile overflow"))?;
    let groups = (channel_tiles as usize)
        .checked_mul(spatial_tiles as usize)
        .and_then(|n| n.checked_mul(words[2] as usize))
        .and_then(|n| n.checked_mul(64))
        .ok_or_else(|| resource("MPP convolution dispatch overflow"))?;
    // dispatch_grid uses its 64-item unit. The native entry metadata
    // and encoder launch 128 threads for each of the resulting workgroups.
    let (x, y) = dispatch_grid(groups, limits)?;
    recipe.words[32] = channel_tiles;
    recipe.words[33] = spatial_tiles;
    recipe.words[34] = x;
    recipe.words[35] = recipe.words[2];
    recipe.words[36] = padded_channels;
    recipe.x = x;
    recipe.y = y;
    Ok(Some(specialization.kernel))
}
