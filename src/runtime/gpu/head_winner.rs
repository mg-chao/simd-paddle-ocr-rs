//! Optional row-parallel Logit reduction over immutable projection partials.
use super::*;

pub(super) const NAME: &str = "projection_winners_logit_parallel";

fn eligible(
    half_head: bool,
    probability: bool,
    [rows, tiles]: [usize; 2],
    limits: &crate::runtime::gpu::GpuLimits,
) -> bool {
    half_head
        && !probability
        && (32..=512).contains(&tiles)
        && rows >= 32
        && rows <= limits.max_compute_workgroups_per_dimension as usize
        && limits.max_compute_workgroup_size_x >= 64
        && limits.max_compute_invocations_per_workgroup >= 64
        && limits.max_compute_workgroup_storage_size >= 1024
}

pub(super) fn kernel(
    context: &GpuContext,
    half_head: bool,
    probability: bool,
    dimensions: [usize; 2],
) -> RuntimeResult<Option<Arc<Kernel>>> {
    kernel_with(context, half_head, probability, dimensions, Kernel::get)
}

fn kernel_with(
    context: &GpuContext,
    half_head: bool,
    probability: bool,
    dimensions: [usize; 2],
    mut compile: impl FnMut(&GpuContext, KernelKey, &[bool]) -> RuntimeResult<Arc<Kernel>>,
) -> RuntimeResult<Option<Arc<Kernel>>> {
    if !eligible(half_head, probability, dimensions, &context.limits) {
        return Ok(None);
    }
    match compile(
        context,
        KernelKey::new(KernelId::HeadReduceParallel),
        &[true, false],
    ) {
        Ok(kernel) => Ok(Some(kernel)),
        Err(error) if matches!(error.kind(), ErrorKind::Unsupported) => {
            context.check_alive()?;
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
#[path = "head_winner_tests.rs"]
mod tests;
