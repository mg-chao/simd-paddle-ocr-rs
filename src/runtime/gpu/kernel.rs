//! Typed shader identities shared by the OCR schedule and platform compilers.
use super::GpuPrecision;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum KernelId {
    #[cfg(test)]
    BindingProbe,
    #[cfg(test)]
    CounterProbe,
    #[cfg(test)]
    RoundHalfProbe,
    #[cfg(test)]
    HalfStoreProbe,
    ProjectionMatrixF32,
    ProjectionMatrixF16,
    Attention,
    AttentionViews,
    SqueezeExcite,
    SmallProjection,
    Transposed,
    Convolution,
    General,
    Depthwise3,
    Depthwise5,
    #[cfg(target_os = "macos")]
    Depthwise7,
    Reduce,
    Softmax,
    LayerNorm,
    ReducePartial,
    ReduceFinish,
    SoftmaxPartial,
    SoftmaxFinish,
    Matmul16,
    Conv16,
    Matmul32,
    Conv32,
    #[cfg(target_os = "macos")]
    Conv64,
    #[cfg(target_os = "macos")]
    Matmul64,
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
            #[cfg(test)]
            Self::BindingProbe => 0,
            #[cfg(test)]
            Self::HalfStoreProbe => 32,
            Self::ProjectionMatrixF32 | Self::ProjectionMatrixF16 => 0,
            Self::Attention => 47,
            Self::AttentionViews => 47,
            Self::SqueezeExcite => 47,
            Self::SmallProjection => 63,
            Self::Transposed => 63,
            Self::Convolution => 63,
            Self::General => 63,
            Self::Depthwise3 => 63,
            Self::Depthwise5 => 63,
            #[cfg(target_os = "macos")]
            Self::Depthwise7 => 63,
            Self::Reduce => 39,
            Self::Softmax => 39,
            Self::LayerNorm => 39,
            Self::ReducePartial => 35,
            Self::ReduceFinish => 35,
            Self::SoftmaxPartial => 35,
            Self::SoftmaxFinish => 35,
            Self::Matmul16 => 63,
            Self::Conv16 => 63,
            Self::Matmul32 => 63,
            Self::Conv32 => 63,
            #[cfg(target_os = "macos")]
            Self::Conv64 | Self::Matmul64 => 63,
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
