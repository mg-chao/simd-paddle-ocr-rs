//! Source template for canonical Half pointwise convolution with F32 accumulation.
//!
//! The Half source constructor replaces the buffer and tensor element types;
//! the four-SIMD-group tile, epilogue, and accumulation order remain shared.

pub(super) fn source() -> String {
    format!(
        "{}\n{}",
        include_str!("common_native.metal"),
        include_str!("pointwise_mpp64.metal")
    )
}
