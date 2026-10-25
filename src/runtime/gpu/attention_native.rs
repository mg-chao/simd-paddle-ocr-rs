//! Same four-query online attention with uniform SIMD32 score/weight exchange.
use super::super::*;
use operators::ShaderKind;

pub(in crate::runtime::gpu) const ENTRY: &str = "main_";
pub(in crate::runtime::gpu) const WORKGROUP_SIZE: [u32; 3] = [64, 1, 1];
const SOURCE: &str = include_str!("attention_native.metal");
const SIZED_BINDINGS: &[usize] = &[0, 1, 2, 3, 4, 5];
const WORKGROUP_MEMORY: &[usize] = &[512, 2048, 2048, 256, 16, 16, 16];
const REQUIRED_BINDINGS: &[(usize, u64)] = &[(0, 2), (1, 2), (2, 2), (3, 2), (5, 2), (6, 512)];

pub(in crate::runtime::gpu) fn source() -> &'static str {
    SOURCE
}

/// Packed flags7 is produced only by the closed, explicit matrix-F16 QKV
/// fusion. Unpacked/escaping views and F32-matrix storage policies retain the
/// generic native recipe. Admission validates this source before recording.
pub(in crate::runtime::gpu) fn eligible(
    kind: ShaderKind,
    storage: [GpuPrecision; 6],
    words: &[u32; 128],
    limits: &wgpu::Limits,
) -> bool {
    if kind != ShaderKind::AttentionViews
        || storage != [GpuPrecision::MixedF16; 6]
        || words[2] != 7
        || words[3] == 0
        || words[3] != words[4]
        || !(1..=32).contains(&words[5])
        || words[5] != words[6]
        || !f32::from_bits(words[7]).is_finite()
        || words[8] != words[3].div_ceil(4)
        || words[15] == 0
        || words[0] == 0
        || !words[0].is_multiple_of(words[8])
        || words[1] == 0
        || words[1] > limits.max_compute_workgroups_per_dimension
        || words[0].div_ceil(words[1]) > limits.max_compute_workgroups_per_dimension
        || words[9..15].iter().any(|&n| n != 0)
        || words[64] != 0
        || limits.max_compute_workgroup_size_x < WORKGROUP_SIZE[0]
        || limits.max_compute_invocations_per_workgroup < WORKGROUP_SIZE[0]
        || limits.max_compute_workgroup_storage_size < 4912
    {
        return false;
    }
    let batches = words[0] / words[8];
    if !batches.is_multiple_of(words[15]) {
        return false;
    }
    let n = batches / words[15];
    let elements = n
        .checked_mul(words[3])
        .and_then(|count| count.checked_mul(words[15]))
        .and_then(|count| count.checked_mul(words[5]));
    elements.is_some_and(|count| {
        count.checked_mul(3).is_some() && words[60].checked_add(count).is_some()
    })
}

pub(in crate::runtime::gpu) fn compile(
    context: &GpuContext,
) -> RuntimeResult<Arc<executor::NativePipeline>> {
    executor::compile_msl_metadata(
        context,
        source(),
        ENTRY,
        WORKGROUP_SIZE,
        SIZED_BINDINGS.to_vec(),
        WORKGROUP_MEMORY.to_vec(),
        REQUIRED_BINDINGS.to_vec(),
    )
}

#[cfg(test)]
#[path = "attention_native_tests.rs"]
mod tests;
