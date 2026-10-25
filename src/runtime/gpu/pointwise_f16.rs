//! Explicit fast-mode MPP pointwise over physical FP16 NHWC/HWIO operands.
//!
//! Logical activations, bias, residuals, fused epilogues and output remain F32.
//! Physical lowering converts the two matrix operands once; the native MPP
//! operation uses FP32 accumulation and relaxed_precision=false.
pub(super) fn source() -> String {
    format!(
        "{}\n{}",
        include_str!("common_native.metal"),
        include_str!("pointwise_f16.metal")
    )
}
