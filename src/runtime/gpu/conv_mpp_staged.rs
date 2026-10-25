//! Width-independent spatial convolution for explicit all-Half matrix mode.
//! A bounded threadgroup NHWC patch replaces the global input conversion.
use super::*;
use operators::{Recipe, RecipeKind, ShaderKind};

pub(super) const WORKGROUP: [u32; 3] = [128, 1, 1];
const MAX_CHANNELS: u32 = 64;
const PATCH_ELEMENTS: u32 = 9 * 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Family {
    K2S1,
    K3S2,
}

pub(super) fn supports_kind(kind: ShaderKind) -> bool {
    matches!(
        kind,
        ShaderKind::ConvMpp2Staged | ShaderKind::ConvMpp3s2Staged
    )
}

/// Only Cin affects generated code. Image dimensions, batch, Cout and padding
/// remain runtime parameters and must not multiply the immutable cache entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Specialization {
    channels: u32,
    family: Family,
}
impl Specialization {
    pub(super) fn from_words(words: &[u32; 128]) -> Option<Self> {
        let family = match (words[11], words[12], words[13], words[14]) {
            (2, 2, 1, 1) if (1..=MAX_CHANNELS).contains(&words[3]) => Family::K2S1,
            (3, 3, 2, 2) if matches!(words[3], 3 | 24) => Family::K3S2,
            _ => return None,
        };
        Some(Self {
            channels: words[3],
            family,
        })
    }

    fn shader(self) -> ShaderKind {
        match self.family {
            Family::K2S1 => ShaderKind::ConvMpp2Staged,
            Family::K3S2 => ShaderKind::ConvMpp3s2Staged,
        }
    }

    fn shared_bytes(self) -> u32 {
        // Match Metal's 16-byte static threadgroup allocation rounding.
        let patch = match self.family {
            Family::K2S1 => PATCH_ELEMENTS,
            Family::K3S2 => 17 * 9,
        };
        (self.channels * patch * 2).div_ceil(16) * 16
    }

    fn fits(self, limits: &crate::runtime::gpu::GpuLimits) -> bool {
        limits.max_compute_invocations_per_workgroup >= WORKGROUP[0]
            && limits.max_compute_workgroup_size_x >= WORKGROUP[0]
            && self.shared_bytes() <= limits.max_compute_workgroup_storage_size
    }
}

/// Upgrade an already validated, canonical full-view candidate. Its checked
/// signed addressing, HWIO padding and dispatch packet are retained verbatim.
/// Unsupported local capacity is a capability rejection, not a layout rewrite.
pub(super) fn prepare_recipe(
    recipe: &mut Recipe,
    descriptors: &[GpuTensorDescriptor],
    limits: &crate::runtime::gpu::GpuLimits,
) -> bool {
    let Some(specialization) = Specialization::from_words(&recipe.words) else {
        return false;
    };
    let words = &recipe.words;
    let family_matches = match (specialization.family, &recipe.kind) {
        (Family::K2S1, RecipeKind::Compute(ShaderKind::ConvMpp2s1)) => {
            (8..=MAX_CHANNELS).contains(&words[4]) && words[11..=16] == [2, 2, 1, 1, 1, 1]
        }
        (Family::K3S2, RecipeKind::Compute(ShaderKind::ConvMpp3s2)) => {
            // Keep admission to measured channel pairs. Cin24 -> Cout48
            // removes Tiny's global pack; RGB -> Cout24 and wide reductions
            // retain the original full-view path.
            matches!((words[3], words[4]), (3, 48 | 64) | (24, 48))
                && words[11..=16] == [3, 3, 2, 2, 1, 1]
        }
        _ => false,
    };
    if !family_matches
        || words[9] != words[3]
        || words[10] != words[4]
        || words[37] != 1
        || words[38] != 1
        || words[52] != 0
        || !specialization.fits(limits)
        || !recipe
            .inputs
            .iter()
            .chain(std::iter::once(&recipe.output))
            .all(|id| {
                descriptors.get(id.0).is_some_and(|descriptor| {
                    descriptor.precision == GpuPrecision::MixedF16
                        && descriptor.layout == GpuLayout::Canonical
                })
            })
    {
        return false;
    }
    // Both families retain the full-view 32x8x4 output tile and grid packet.
    recipe.kind = RecipeKind::Compute(specialization.shader());
    true
}
