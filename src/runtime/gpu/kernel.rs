//! Typed shader identities shared by the OCR schedule and platform compilers.
use super::GpuPrecision;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum KernelId {
    #[cfg(all(test, not(target_os = "macos")))]
    BindingProbe,
    #[cfg(test)]
    CounterProbe,
    #[cfg(test)]
    RoundHalfProbe,
    #[cfg(test)]
    HalfStoreProbe,
    ProjectionMatrixF32,
    ProjectionMatrixF16,
    #[cfg(not(target_os = "macos"))]
    Attention,
    #[cfg(not(target_os = "macos"))]
    AttentionViews,
    #[cfg(any(not(target_os = "macos"), test))]
    SqueezeExcite,
    #[cfg(any(not(target_os = "macos"), test))]
    SmallProjection,
    #[cfg(not(target_os = "macos"))]
    Transposed,
    #[cfg(not(target_os = "macos"))]
    Convolution,
    #[cfg(not(target_os = "macos"))]
    General,
    #[cfg(not(target_os = "macos"))]
    PackNhwc,
    #[cfg(not(target_os = "macos"))]
    Depthwise3,
    #[cfg(not(target_os = "macos"))]
    Depthwise5,
    #[cfg(not(target_os = "macos"))]
    Reduce,
    #[cfg(not(target_os = "macos"))]
    Softmax,
    #[cfg(not(target_os = "macos"))]
    LayerNorm,
    #[cfg(not(target_os = "macos"))]
    ReducePartial,
    #[cfg(not(target_os = "macos"))]
    ReduceFinish,
    #[cfg(not(target_os = "macos"))]
    SoftmaxPartial,
    #[cfg(not(target_os = "macos"))]
    SoftmaxFinish,
    #[cfg(not(target_os = "macos"))]
    Matmul16,
    #[cfg(not(target_os = "macos"))]
    Conv16,
    #[cfg(not(target_os = "macos"))]
    Matmul32,
    #[cfg(not(target_os = "macos"))]
    Conv32,
    Crop,
    CropMeta,
    PackBgr,
    Rotate,
    DbRows,
    DbSort,
    HeadReduce,
    HeadReduceParallel,
    Ctc,
    DbLabel,
    DbGeometry,
    Heads,
    Heads4,
    #[cfg(not(target_os = "macos"))]
    Heads4Subgroup,
    DbGeometryFast,
    HeadsFeaturesF32,
    Image,
}
impl KernelId {
    pub fn precision_mask(self) -> u8 {
        match self {
            #[cfg(test)]
            Self::CounterProbe | Self::RoundHalfProbe => 0,
            #[cfg(all(test, not(target_os = "macos")))]
            Self::BindingProbe => 0,
            #[cfg(test)]
            Self::HalfStoreProbe => 32,
            Self::ProjectionMatrixF32 | Self::ProjectionMatrixF16 => 0,
            #[cfg(not(target_os = "macos"))]
            Self::Attention => 47,
            #[cfg(not(target_os = "macos"))]
            Self::AttentionViews => 47,
            #[cfg(any(not(target_os = "macos"), test))]
            Self::SqueezeExcite => 47,
            #[cfg(any(not(target_os = "macos"), test))]
            Self::SmallProjection => 63,
            #[cfg(not(target_os = "macos"))]
            Self::Transposed => 63,
            #[cfg(not(target_os = "macos"))]
            Self::Convolution => 63,
            #[cfg(not(target_os = "macos"))]
            Self::General => 63,
            #[cfg(not(target_os = "macos"))]
            Self::PackNhwc => 35,
            #[cfg(not(target_os = "macos"))]
            Self::Depthwise3 => 63,
            #[cfg(not(target_os = "macos"))]
            Self::Depthwise5 => 63,
            #[cfg(not(target_os = "macos"))]
            Self::Reduce => 39,
            #[cfg(not(target_os = "macos"))]
            Self::Softmax => 39,
            #[cfg(not(target_os = "macos"))]
            Self::LayerNorm => 39,
            #[cfg(not(target_os = "macos"))]
            Self::ReducePartial => 35,
            #[cfg(not(target_os = "macos"))]
            Self::ReduceFinish => 35,
            #[cfg(not(target_os = "macos"))]
            Self::SoftmaxPartial => 35,
            #[cfg(not(target_os = "macos"))]
            Self::SoftmaxFinish => 35,
            #[cfg(not(target_os = "macos"))]
            Self::Matmul16 => 63,
            #[cfg(not(target_os = "macos"))]
            Self::Conv16 => 63,
            #[cfg(not(target_os = "macos"))]
            Self::Matmul32 => 63,
            #[cfg(not(target_os = "macos"))]
            Self::Conv32 => 63,
            Self::Crop => 0,
            Self::CropMeta => 0,
            Self::PackBgr => 0,
            Self::Rotate => 0,
            Self::DbRows => 0,
            Self::DbSort => 0,
            Self::HeadReduce => 0,
            Self::HeadReduceParallel => 0,
            Self::Ctc => 0,
            Self::DbLabel => 1,
            Self::DbGeometry => 1,
            Self::Heads => 1,
            Self::Heads4 => 1,
            #[cfg(not(target_os = "macos"))]
            Self::Heads4Subgroup => 1,
            Self::DbGeometryFast => 1,
            Self::HeadsFeaturesF32 => 1,
            Self::Image => 4,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct KernelKey {
    pub id: KernelId,
    pub precision: u8,
}
impl KernelKey {
    pub fn new(id: KernelId) -> Self {
        Self { id, precision: 0 }
    }
    pub fn storage(id: KernelId, precision: GpuPrecision) -> Self {
        Self {
            id,
            precision: if precision == GpuPrecision::MixedF16 {
                id.precision_mask()
            } else {
                0
            },
        }
    }
}
