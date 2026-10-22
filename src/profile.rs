//! Optional measurements of the production path, independent from diagnostic tracing.
use serde::Serialize;
use std::time::Instant;

/// One recognition owner's dispatch interval. CPU time covers its calling
/// thread, excluding nested kernel workers; offsets share one dispatch origin.
#[cfg(feature = "profiling")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RecognitionOwnerProfile {
    pub owner_index: usize,
    pub rayon_worker_index: Option<usize>,
    pub dispatch_start_offset_ns: u64,
    pub dispatch_end_offset_ns: u64,
    pub wall_ns: u64,
    /// Available on macOS when the thread CPU clock succeeds.
    pub thread_cpu_ns: Option<u64>,
    pub job_count: usize,
    pub line_count: usize,
    pub scheduled_width_cost: usize,
    pub maximum_tensor_width: usize,
    pub kernel_threads: usize,
}

/// Instrumented stage durations in nanoseconds. These include timer overhead
/// and are for profiling, not official latency measurements. Storage assembly
/// copies borrowed line results into the OCR slab; owned conversion is separate.
/// Worker durations are summed work, whereas the explicitly named wall fields
/// retain elapsed phase time. These counters are not an additive decomposition
/// of total wall time; do not subtract summed worker work from wall durations.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct StageProfile {
    pub total_ns: u64,
    /// Input validation, shape selection and result preparation, excluding the
    /// direct Session::reshape calls measured separately below.
    pub binding_ns: u64,
    /// Sum of DET and REC Session::reshape call durations, including cached or
    /// unchanged shapes. Concurrent REC worker durations can exceed wall time.
    /// CLS uses a fixed shape created with its request; that initial binding is
    /// part of request compilation, not per-image reshape work.
    pub reshape_ns: u64,
    /// Calling-thread wall time waiting for GPU context admission. These GPU
    /// counters cover production runs only, are nested in stage inference wall
    /// times, and do not measure device kernel execution time.
    pub gpu_admission_wait_ns: u64,
    /// Host upload, command encoding, queue submission and mapping registration.
    pub gpu_submission_ns: u64,
    /// Host wait for submitted work and its readback mapping callback.
    pub gpu_completion_wait_ns: u64,
    /// Mapped-buffer access, host decoding/validation and unmapping.
    pub gpu_readback_ns: u64,
    pub gpu_compute_ns: Option<u64>,
    pub det_preprocess_ns: u64,
    pub det_inference_ns: u64,
    pub det_postprocess_ns: u64,
    pub crop_ns: u64,
    /// Sum of crop worker time, which can exceed the phase's wall time.
    pub crop_work_ns: u64,
    pub classification_wall_ns: u64,
    pub recognition_wall_ns: u64,
    /// Summed classification and rotation worker durations, including their
    /// validation and orchestration. The following CLS fields are nested work
    /// counters, not additional wall phases.
    pub cls_orientation_ns: u64,
    pub cls_preprocess_ns: u64,
    /// Production feature-prefix execution, excluding input preprocessing.
    /// GPU execution includes upload and synchronous feature readback.
    pub cls_backbone_ns: u64,
    /// CPU compact two-class projection and public probability/label calculation.
    pub cls_head_ns: u64,
    /// Only applied 180-degree pixel rotations; geometric crop rotation belongs
    /// to crop_ns instead.
    pub cls_rotation_ns: u64,
    pub rec_preprocess_ns: u64,
    /// GPU execution includes upload and synchronous feature readback.
    pub rec_backbone_ns: u64,
    /// Compact CPU projection and row argmax work on either execution target.
    pub projection_ns: u64,
    pub ctc_ns: u64,
    pub result_storage_ns: u64,
    pub recognized_lines: usize,
    /// Valid entries in recognition_owners. Inline serial recognition has no
    /// dispatched owners. These records are a snapshot, not summed stage work.
    #[cfg(feature = "profiling")]
    pub recognition_owner_count: usize,
    /// Fixed capacity matches the sixteen-owner stage limit. Unused records
    /// stay default; merging profiles retains the last nonempty owner snapshot.
    #[cfg(feature = "profiling")]
    pub recognition_owners: [RecognitionOwnerProfile; 16],
}

impl StageProfile {
    pub(crate) fn merge(&mut self, other: Self) {
        macro_rules! add { ($($field:ident),*) => { $(self.$field = self.$field.saturating_add(other.$field);)* }; }
        add!(
            total_ns,
            binding_ns,
            reshape_ns,
            gpu_admission_wait_ns,
            gpu_submission_ns,
            gpu_completion_wait_ns,
            gpu_readback_ns,
            det_preprocess_ns,
            det_inference_ns,
            det_postprocess_ns,
            crop_ns,
            crop_work_ns,
            classification_wall_ns,
            recognition_wall_ns,
            cls_orientation_ns,
            cls_preprocess_ns,
            cls_backbone_ns,
            cls_head_ns,
            cls_rotation_ns,
            rec_preprocess_ns,
            rec_backbone_ns,
            projection_ns,
            ctc_ns,
            result_storage_ns,
            recognized_lines
        );
        if let Some(ns) = other.gpu_compute_ns {
            self.gpu_compute_ns = Some(self.gpu_compute_ns.unwrap_or(0).saturating_add(ns));
        }
        #[cfg(feature = "profiling")]
        if other.recognition_owner_count != 0 {
            self.recognition_owner_count = other.recognition_owner_count;
            self.recognition_owners = other.recognition_owners;
        }
    }
}

#[cfg(feature = "profiling")]
pub(crate) fn thread_cpu_ns<const ENABLED: bool>() -> Option<u64> {
    if !ENABLED {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        let mut time = std::mem::MaybeUninit::<libc::timespec>::uninit();
        // SAFETY: clock_gettime initializes the valid timespec pointer on
        // success. The value is read only after a successful return code.
        if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, time.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: the successful clock_gettime call initialized this value.
        let time = unsafe { time.assume_init() };
        u64::try_from(time.tv_sec)
            .ok()?
            .checked_mul(1_000_000_000)?
            .checked_add(u64::try_from(time.tv_nsec).ok()?)
    }
    #[cfg(not(target_os = "macos"))]
    None
}

#[cfg(feature = "profiling")]
pub(crate) fn rayon_worker_index() -> Option<usize> {
    #[cfg(feature = "parallel")]
    {
        rayon::current_thread_index()
    }
    #[cfg(not(feature = "parallel"))]
    {
        None
    }
}
#[cfg(feature = "profiling")]
impl std::ops::AddAssign for StageProfile {
    fn add_assign(&mut self, other: Self) {
        self.merge(other);
    }
}

pub(crate) struct Timer<const ENABLED: bool>(Option<Instant>);
impl<const ENABLED: bool> Timer<ENABLED> {
    #[inline]
    pub(crate) fn start() -> Self {
        Self(if ENABLED && cfg!(feature = "profiling") {
            Some(Instant::now())
        } else {
            None
        })
    }
    #[inline]
    pub(crate) fn elapsed(&self) -> u64 {
        self.0.map_or(0, |start| {
            u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
        })
    }
}

#[cfg(test)]
mod tests;
