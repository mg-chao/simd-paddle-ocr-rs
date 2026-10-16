//! Optional measurements of the production path, independent from diagnostic tracing.
use serde::Serialize;
use std::time::Instant;

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
    pub det_preprocess_ns: u64,
    pub det_inference_ns: u64,
    pub det_postprocess_ns: u64,
    pub crop_ns: u64,
    /// Sum of crop worker time, which can exceed the phase's wall time.
    pub crop_work_ns: u64,
    pub classification_wall_ns: u64,
    pub recognition_wall_ns: u64,
    pub cls_orientation_ns: u64,
    pub rec_preprocess_ns: u64,
    pub rec_backbone_ns: u64,
    pub projection_ns: u64,
    pub ctc_ns: u64,
    pub result_storage_ns: u64,
    pub recognized_lines: usize,
}

impl StageProfile {
    pub(crate) fn merge(&mut self, other: Self) {
        macro_rules! add { ($($field:ident),*) => { $(self.$field = self.$field.saturating_add(other.$field);)* }; }
        add!(
            total_ns,
            binding_ns,
            reshape_ns,
            det_preprocess_ns,
            det_inference_ns,
            det_postprocess_ns,
            crop_ns,
            crop_work_ns,
            classification_wall_ns,
            recognition_wall_ns,
            cls_orientation_ns,
            rec_preprocess_ns,
            rec_backbone_ns,
            projection_ns,
            ctc_ns,
            result_storage_ns,
            recognized_lines
        );
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
