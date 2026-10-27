//! Bounded, reusable OCR requests with borrowed results tied to an RAII lease.
use crate::{
    cpu::CpuContext,
    image::{ImageError, ImageResult, ImageView},
    ocr::{OcrPlan, OcrRequest, OcrTrace, OcrView, OwnedOcr, RunOptions},
};
use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, Condvar, Mutex, MutexGuard},
};

#[derive(Clone, Copy, Debug)]
pub struct EngineLimits {
    /// Maximum requests being created, checked out, or cached together.
    pub max_requests: usize,
    /// Aggregate idle request capacities, excluding shared compiled weights.
    /// Zero disables caching. This is not a process-wide peak allocation cap.
    pub max_idle_bytes: usize,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineStats {
    pub live_requests: usize,
    pub idle_requests: usize,
    pub idle_bytes: usize,
    pub idle_host_bytes: usize,
    /// Request-owned device buffers; shared compiled resources are excluded.
    pub idle_device_bytes: usize,
}

struct State<T> {
    live: usize,
    bytes: usize,
    device_bytes: usize,
    idle: Vec<(T, usize, usize)>,
}
struct Pool<T> {
    limits: EngineLimits,
    state: Mutex<State<T>>,
    available: Condvar,
    factory: Box<dyn Fn() -> ImageResult<T> + Send + Sync>,
    measure: fn(&T) -> crate::runtime::MemorySnapshot,
}
impl<T> Pool<T> {
    fn new(
        limits: EngineLimits,
        factory: impl Fn() -> ImageResult<T> + Send + Sync + 'static,
        measure: fn(&T) -> crate::runtime::MemorySnapshot,
    ) -> ImageResult<Self> {
        if limits.max_requests == 0 {
            return Err(ImageError::new("max_requests must be positive"));
        }
        Ok(Self {
            limits,
            state: Mutex::new(State {
                live: 0,
                bytes: 0,
                device_bytes: 0,
                idle: Vec::new(),
            }),
            available: Condvar::new(),
            factory: Box::new(factory),
            measure,
        })
    }
    fn lock(&self) -> MutexGuard<'_, State<T>> {
        // User code never runs under this lock; unwind only discards a lease.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    fn stats(&self) -> EngineStats {
        let state = self.lock();
        EngineStats {
            live_requests: state.live,
            idle_requests: state.idle.len(),
            idle_bytes: state.bytes,
            idle_host_bytes: state.bytes.saturating_sub(state.device_bytes),
            idle_device_bytes: state.device_bytes,
        }
    }
    fn acquire(&self, wait: bool) -> ImageResult<Option<Lease<'_, T>>> {
        let mut state = self.lock();
        loop {
            if let Some((value, bytes, device_bytes)) = state.idle.pop() {
                state.bytes -= bytes;
                state.device_bytes -= device_bytes;
                return Ok(Some(Lease {
                    pool: self,
                    value: Some(value),
                }));
            }
            if state.live < self.limits.max_requests {
                state.live += 1;
                drop(state);
                // A factory error or panic must release its admission slot.
                let mut reservation = Reservation {
                    pool: self,
                    armed: true,
                };
                let value = (self.factory)()?;
                reservation.armed = false;
                return Ok(Some(Lease {
                    pool: self,
                    value: Some(value),
                }));
            }
            if !wait {
                return Ok(None);
            }
            state = self
                .available
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}
struct Reservation<'a, T> {
    pool: &'a Pool<T>,
    armed: bool,
}
impl<T> Drop for Reservation<'_, T> {
    fn drop(&mut self) {
        if self.armed {
            self.pool.lock().live -= 1;
            self.pool.available.notify_one();
        }
    }
}
struct Lease<'a, T> {
    pool: &'a Pool<T>,
    value: Option<T>,
}
impl<T> Deref for Lease<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().expect("live request lease")
    }
}
impl<T> DerefMut for Lease<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("live request lease")
    }
}
impl<T> Drop for Lease<'_, T> {
    fn drop(&mut self) {
        let Some(value) = self.value.take() else {
            return;
        };
        let memory = (self.pool.measure)(&value);
        let bytes = memory.total_bytes();
        let device_bytes = memory.device_bytes;
        let mut state = self.pool.lock();
        if !std::thread::panicking()
            && bytes <= self.pool.limits.max_idle_bytes.saturating_sub(state.bytes)
            && self.pool.limits.max_idle_bytes != 0
            && state.idle.try_reserve(1).is_ok()
        {
            state.bytes += bytes;
            state.device_bytes += device_bytes;
            state.idle.push((value, bytes, device_bytes));
            drop(state);
        } else {
            state.live -= 1;
            drop(state);
            drop(value);
        }
        self.pool.available.notify_one();
    }
}

/// Shared immutable models and CPU capacity, with bounded mutable request leases.
/// Requests are pinned to the engine context. Leases allow controlled execution
/// and immutable inspection; configure stage budgets on the template first.
pub struct OcrEngine {
    cpu: Arc<CpuContext>,
    pool: Pool<OcrRequest>,
}
impl OcrEngine {
    /// Admit independent requests from a shared immutable plan.
    pub fn new(plan: Arc<OcrPlan>, limits: EngineLimits) -> ImageResult<Self> {
        let cpu = plan.cpu_context().clone();
        Self::from_factory(cpu, limits, move || plan.request())
    }
    /// The factory should share immutable models. Every created request is
    /// rebound to this engine's executor before it can execute any work.
    fn from_factory(
        cpu: Arc<CpuContext>,
        limits: EngineLimits,
        factory: impl Fn() -> ImageResult<OcrRequest> + Send + Sync + 'static,
    ) -> ImageResult<Self> {
        let shared = cpu.clone();
        let pool = Pool::new(
            limits,
            move || {
                let mut request = factory()?;
                request.pin_cpu_context(shared.clone())?;
                Ok(request)
            },
            OcrRequest::memory_snapshot,
        )?;
        Ok(Self { cpu, pool })
    }
    pub fn cpu_context(&self) -> &Arc<CpuContext> {
        &self.cpu
    }
    pub fn stats(&self) -> EngineStats {
        self.pool.stats()
    }
    /// Wait for a request outside the CPU executor. Reentrant acquisition from
    /// an engine worker or trace callback is rejected instead of risking an
    /// admission deadlock. `try_acquire` always remains nonblocking.
    pub fn acquire(&self) -> ImageResult<OcrLease<'_>> {
        if self.cpu.is_worker_thread() || self.cpu.is_active_request() {
            return Err(ImageError::new(
                "blocking request acquisition from an executing CPU context is unsupported; use try_acquire",
            ));
        }
        Ok(OcrLease {
            lease: self
                .pool
                .acquire(true)?
                .expect("blocking acquisition returns a lease"),
        })
    }
    pub fn try_acquire(&self) -> ImageResult<Option<OcrLease<'_>>> {
        Ok(self.pool.acquire(false)?.map(|lease| OcrLease { lease }))
    }
    /// Lease a request, waiting until the configured admission limit permits it.
    pub fn request(&self) -> ImageResult<OcrLease<'_>> {
        self.acquire()
    }
    /// Lease an available request without blocking when the engine is full.
    pub fn try_request(&self) -> ImageResult<Option<OcrLease<'_>>> {
        self.try_acquire()
    }
    pub fn run_owned(&self, image: ImageView<'_>, options: RunOptions) -> ImageResult<OwnedOcr> {
        self.acquire()?.run(image, options)?.try_to_owned()
    }
}
/// Controlled execution and immutable request inspection. Component replacement
/// and CPU-context mutation are available on standalone `OcrRequest`, rather than a
/// pooled lease. A borrowed view cannot outlive its lease or another mutable use.
///
/// ```compile_fail
/// use simd_paddle_ocr_rs::{engine::OcrEngine, image::ImageView, ocr::OcrView};
/// fn escape<'a>(engine: &'a OcrEngine, image: ImageView<'_>) -> OcrView<'a> {
///     let mut lease = engine.acquire().unwrap();
///     lease.run(image, simd_paddle_ocr_rs::RunOptions { capture_spans: false }).unwrap()
/// }
/// ```
///
/// ```compile_fail
/// use simd_paddle_ocr_rs::{engine::OcrEngine, image::ImageView};
/// fn release_while_borrowed(engine: &OcrEngine, image: ImageView<'_>) {
///     let mut lease = engine.acquire().unwrap();
///     let result = lease.run(image, simd_paddle_ocr_rs::RunOptions { capture_spans: false }).unwrap();
///     drop(lease);
///     println!("{}", result.text);
/// }
/// ```
///
/// ```compile_fail
/// use simd_paddle_ocr_rs::{engine::OcrEngine, cpu::CpuContext};
/// fn change_context(engine: &OcrEngine) {
///     let mut lease = engine.request().unwrap();
///     lease.set_cpu_context(CpuContext::serial()).unwrap();
/// }
/// ```
///
/// ```compile_fail
/// use simd_paddle_ocr_rs::engine::OcrEngine;
/// fn change_component(engine: &OcrEngine) {
///     let mut lease = engine.request().unwrap();
///     let _ = lease.recognizer_mut();
/// }
/// ```
///
/// ```compile_fail
/// use simd_paddle_ocr_rs::{engine::OcrEngine, ocr::OcrRequest};
/// fn replace_request(engine: &OcrEngine, replacement: OcrRequest) {
///     let mut lease = engine.request().unwrap();
///     *lease = replacement;
/// }
/// ```
pub struct OcrLease<'a> {
    lease: Lease<'a, OcrRequest>,
}
impl OcrLease<'_> {
    pub fn run(&mut self, image: ImageView<'_>, options: RunOptions) -> ImageResult<OcrView<'_>> {
        self.lease.run(image, options)
    }
    #[cfg(feature = "profiling")]
    pub fn run_profiled(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
    ) -> ImageResult<(OcrView<'_>, crate::profile::StageProfile)> {
        self.lease.run_profiled(image, capture)
    }
    pub fn run_with_trace<F>(
        &mut self,
        image: ImageView<'_>,
        capture: bool,
        observer: F,
    ) -> ImageResult<OcrView<'_>>
    where
        F: for<'a> FnMut(OcrTrace<'a>) -> ImageResult<()>,
    {
        self.lease.run_with_trace(image, capture, observer)
    }
}
impl Deref for OcrLease<'_> {
    type Target = OcrRequest;
    fn deref(&self) -> &OcrRequest {
        &self.lease
    }
}
impl Drop for OcrLease<'_> {
    fn drop(&mut self) {
        if self.lease.value.as_ref().is_some_and(OcrRequest::unwound) {
            let value = self.lease.value.take();
            self.lease.pool.lock().live -= 1;
            self.lease.pool.available.notify_one();
            drop(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn idle_accounting_separates_host_and_device_and_removes_both_on_acquire() {
        let pool = Pool::new(
            EngineLimits {
                max_requests: 1,
                max_idle_bytes: 100,
            },
            || Ok((40usize, 24usize)),
            |&(host_bytes, device_bytes)| crate::runtime::MemorySnapshot {
                host_bytes,
                device_bytes,
            },
        )
        .unwrap();
        let lease = pool.acquire(false).unwrap().unwrap();
        drop(lease);
        assert_eq!(pool.stats().idle_bytes, 64);
        assert_eq!(pool.stats().idle_host_bytes, 40);
        assert_eq!(pool.stats().idle_device_bytes, 24);
        let mut lease = pool.acquire(false).unwrap().unwrap();
        assert_eq!(pool.stats().idle_bytes, 0);
        assert_eq!(pool.stats().idle_device_bytes, 0);
        *lease = (80, 24);
        drop(lease);
        assert_eq!(pool.stats(), EngineStats::default());
    }
    #[test]
    fn leased_requests_cannot_change_the_engine_context_or_pollute_cached_requests() {
        let cpu = CpuContext::serial();
        let engine = OcrEngine::new(
            std::sync::Arc::new(
                (crate::ocr::tests::synthetic_ocr(cpu.clone()))
                    .plan()
                    .clone(),
            ),
            EngineLimits {
                max_requests: 1,
                max_idle_bytes: 16 << 20,
            },
        )
        .unwrap();
        {
            let mut request = engine.request().unwrap();
            let different = CpuContext::serial();
            assert!(!Arc::ptr_eq(&cpu, &different));
            assert!(request.lease.set_cpu_context(different.clone()).is_err());
            assert!(
                request
                    .lease
                    .set_execution_context(different, crate::cpu::CpuOptions::default())
                    .is_err()
            );
            request.lease.set_cpu_context(cpu.clone()).unwrap();
            assert!(Arc::ptr_eq(request.cpu_context(), &cpu));
        }
        let request = engine.request().unwrap();
        assert!(Arc::ptr_eq(request.cpu_context(), &cpu));
    }
    #[cfg(feature = "parallel")]
    #[test]
    fn caller_thread_trace_rejects_reentrant_wait_and_clears_admission_after_panics() {
        let cpu = Arc::new(CpuContext::new(2).unwrap());
        let engine = OcrEngine::new(
            std::sync::Arc::new(
                (crate::ocr::tests::synthetic_ocr(cpu.clone()))
                    .plan()
                    .clone(),
            ),
            EngineLimits {
                max_requests: 1,
                max_idle_bytes: 16 << 20,
            },
        )
        .unwrap();
        let pixels = crate::ocr::tests::two_lines();
        let image = ImageView::new(&pixels, 96, 96, 0, crate::image::PixelFormat::Bgr24).unwrap();
        let mut request = engine.request().unwrap();
        let caller = std::thread::current().id();
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        request
            .run_with_trace(image, true, |_| {
                assert_eq!(std::thread::current().id(), caller);
                assert!(!cpu.is_worker_thread());
                assert!(cpu.is_active_request());
                assert!(engine.request().is_err());
                assert!(engine.try_request()?.is_none());
                calls.set(calls.get() + 1);
                Ok(())
            })
            .unwrap();
        assert!(calls.get() > 0);
        assert!(!cpu.is_active_request());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = request.run_with_trace(image, true, |_| panic!("trace callback panic"));
            }))
            .is_err()
        );
        assert!(!cpu.is_active_request());
        drop(request);
        assert_eq!(engine.stats(), EngineStats::default());
        assert!(engine.try_request().unwrap().is_some());
    }
    #[test]
    fn caught_callback_panic_still_discards_the_request() {
        let engine = OcrEngine::new(
            std::sync::Arc::new(
                (crate::ocr::tests::synthetic_ocr(CpuContext::serial()))
                    .plan()
                    .clone(),
            ),
            EngineLimits {
                max_requests: 1,
                max_idle_bytes: 16 << 20,
            },
        )
        .unwrap();
        let pixels = crate::ocr::tests::two_lines();
        let image = ImageView::new(&pixels, 96, 96, 0, crate::image::PixelFormat::Bgr24).unwrap();
        let mut request = engine.request().unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = request.run_with_trace(image, true, |_| panic!("injected callback panic"));
            }))
            .is_err()
        );
        assert!(
            request
                .run(
                    image,
                    crate::RunOptions {
                        capture_spans: false
                    }
                )
                .is_err()
        );
        drop(request);
        assert_eq!(engine.stats(), EngineStats::default());
        let mut replacement = engine.try_request().unwrap().unwrap();
        assert_eq!(
            replacement
                .run(
                    image,
                    crate::RunOptions {
                        capture_spans: false
                    }
                )
                .unwrap()
                .detected_count,
            2
        );
    }
    #[test]
    fn traced_requests_are_evicted_when_diagnostics_exceed_the_idle_budget() {
        fn bytes(request: &OcrRequest) -> usize {
            request.retained_activation_bytes()
                + request.diagnostic_retained_activation_bytes()
                + request.retained_workspace_bytes()
        }
        let pixels = crate::ocr::tests::two_lines();
        let image = ImageView::new(&pixels, 96, 96, 0, crate::image::PixelFormat::Bgr24).unwrap();
        let mut template = crate::ocr::tests::synthetic_ocr(CpuContext::serial());
        template
            .run(
                image,
                crate::RunOptions {
                    capture_spans: true,
                },
            )
            .unwrap();
        let production_bytes = bytes(&template);
        let engine = OcrEngine::new(
            std::sync::Arc::new((template).plan().clone()),
            EngineLimits {
                max_requests: 1,
                max_idle_bytes: production_bytes,
            },
        )
        .unwrap();
        {
            let mut request = engine.acquire().unwrap();
            request
                .run(
                    image,
                    crate::RunOptions {
                        capture_spans: true,
                    },
                )
                .unwrap();
            assert!(bytes(&request) <= production_bytes);
        }
        assert_eq!(engine.stats().idle_requests, 1);
        {
            let mut request = engine.acquire().unwrap();
            let result = request.run_with_trace(image, true, |_| Ok(())).unwrap();
            assert_eq!(result.detected_count, 2);
            assert!(
                bytes(&request) > production_bytes,
                "trace diagnostic capacities must be included in the lease size"
            );
        }
        assert_eq!(engine.stats(), EngineStats::default());
        let mut replacement = engine.acquire().unwrap();
        assert_eq!(
            replacement
                .run(
                    image,
                    crate::RunOptions {
                        capture_spans: true
                    }
                )
                .unwrap()
                .detected_count,
            2
        );
        drop(replacement);
        assert_eq!(engine.stats().idle_requests, 1);
    }
    #[test]
    fn leases_reuse_state_enforce_admission_and_evict_large_capacity() {
        let calls = Arc::new(AtomicUsize::new(0));
        let count = calls.clone();
        let pool = Pool::new(
            EngineLimits {
                max_requests: 1,
                max_idle_bytes: 16,
            },
            move || {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(Vec::<u8>::new())
            },
            |v| crate::runtime::MemorySnapshot::host(v.capacity()),
        )
        .unwrap();
        let mut first = pool.acquire(false).unwrap().unwrap();
        assert!(pool.acquire(false).unwrap().is_none());
        first.extend_from_slice(&[1, 2, 3]);
        drop(first);
        let second = pool.acquire(false).unwrap().unwrap();
        assert_eq!(second.as_slice(), [1, 2, 3]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        drop(second);
        let mut large = pool.acquire(false).unwrap().unwrap();
        large.reserve(128);
        drop(large);
        assert_eq!(pool.stats(), EngineStats::default());
    }
    #[test]
    fn factory_failure_and_panic_return_admission_and_waiters_wake() {
        let calls = Arc::new(AtomicUsize::new(0));
        let count = calls.clone();
        let pool = Arc::new(
            Pool::new(
                EngineLimits {
                    max_requests: 1,
                    max_idle_bytes: 8,
                },
                move || match count.fetch_add(1, Ordering::SeqCst) {
                    0 => Err(ImageError::new("injected creation failure")),
                    1 => panic!("injected creation panic"),
                    _ => Ok(7usize),
                },
                |_| crate::runtime::MemorySnapshot::host(8),
            )
            .unwrap(),
        );
        assert!(pool.acquire(false).is_err());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.acquire(false))).is_err()
        );
        assert_eq!(pool.stats().live_requests, 0);
        let lease = pool.acquire(true).unwrap().unwrap();
        let other = pool.clone();
        let (sent, received) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let value = other.acquire(true).unwrap().unwrap();
            sent.send(*value).unwrap();
        });
        assert!(received.try_recv().is_err());
        drop(lease);
        assert_eq!(
            received
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap(),
            7
        );
        thread.join().unwrap();
        assert_eq!(pool.stats().idle_requests, 1);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _lease = pool.acquire(true).unwrap().unwrap();
                panic!("injected request panic");
            }))
            .is_err()
        );
        assert_eq!(pool.stats(), EngineStats::default());
    }
    #[test]
    fn simultaneous_requests_never_exceed_the_cap_or_share_mutable_values() {
        let pool = Arc::new(
            Pool::new(
                EngineLimits {
                    max_requests: 2,
                    max_idle_bytes: 16,
                },
                || Ok(0usize),
                |_| crate::runtime::MemorySnapshot::host(8),
            )
            .unwrap(),
        );
        std::thread::scope(|scope| {
            for index in 1..=8 {
                let pool = &pool;
                scope.spawn(move || {
                    for _ in 0..20 {
                        let mut lease = pool.acquire(true).unwrap().unwrap();
                        *lease = index;
                        std::thread::yield_now();
                        assert_eq!(*lease, index);
                        assert!(pool.stats().live_requests <= 2);
                    }
                });
            }
        });
        assert!(pool.stats().idle_bytes <= 16);
    }
    #[test]
    fn inline_engine_serializes_complete_requests_and_keeps_callbacks_on_the_caller() {
        use crate::{detector::DetectionTrace, ocr::OcrTrace};

        let cpu = CpuContext::serial();
        let engine = OcrEngine::new(
            std::sync::Arc::new(
                (crate::ocr::tests::synthetic_ocr(cpu.clone()))
                    .plan()
                    .clone(),
            ),
            EngineLimits {
                max_requests: 3,
                max_idle_bytes: 32 << 20,
            },
        )
        .unwrap();
        assert!(cpu.install(|| engine.request().is_err()));
        let pixels = crate::ocr::tests::two_lines();
        let ready = std::sync::Barrier::new(3);
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..3 {
                let engine = &engine;
                let pixels = &pixels;
                let ready = &ready;
                let active = &active;
                let peak = &peak;
                scope.spawn(move || {
                    let mut request = engine.request().unwrap();
                    let caller = std::thread::current().id();
                    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
                    ready.wait();
                    for _ in 0..3 {
                        let image =
                            ImageView::new(pixels, 96, 96, 0, crate::image::PixelFormat::Bgr24)
                                .unwrap();
                        let result = request
                            .run_with_trace(image, true, |event| {
                                assert_eq!(std::thread::current().id(), caller);
                                calls.set(calls.get() + 1);
                                match event {
                                    OcrTrace::Detection(DetectionTrace::Input { .. }) => {
                                        let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                                        peak.fetch_max(current, Ordering::SeqCst);
                                        std::thread::sleep(std::time::Duration::from_millis(2));
                                    }
                                    OcrTrace::Result(_) => {
                                        active.fetch_sub(1, Ordering::SeqCst);
                                    }
                                    _ => {}
                                }
                                Ok(())
                            })
                            .unwrap();
                        assert_eq!(result.detected_count, 2);
                    }
                    assert!(calls.get() > 0);
                });
            }
        });
        assert_eq!(peak.load(Ordering::SeqCst), 1);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert_eq!(engine.stats().idle_requests, 3);
    }
    #[cfg(feature = "parallel")]
    #[test]
    fn real_ocr_leases_share_cpu_and_preserve_borrowed_and_owned_results() {
        let cpu = Arc::new(CpuContext::new(2).unwrap());
        let template = crate::ocr::tests::synthetic_ocr(cpu.clone());
        let engine = Arc::new(
            OcrEngine::new(
                std::sync::Arc::new((template).plan().clone()),
                EngineLimits {
                    max_requests: 2,
                    max_idle_bytes: 16 << 20,
                },
            )
            .unwrap(),
        );
        assert!(cpu.install(|| engine.acquire().is_err()));
        let pixels = Arc::new(crate::ocr::tests::two_lines());
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let engine = &engine;
                let pixels = &pixels;
                let cpu = &cpu;
                scope.spawn(move || {
                    for _ in 0..3 {
                        let image =
                            ImageView::new(pixels, 96, 96, 0, crate::image::PixelFormat::Bgr24)
                                .unwrap();
                        let mut request = engine.acquire().unwrap();
                        assert!(Arc::ptr_eq(request.cpu_context(), cpu));
                        let borrowed = request
                            .run(
                                image,
                                crate::RunOptions {
                                    capture_spans: true,
                                },
                            )
                            .unwrap();
                        assert_eq!(borrowed.detected_count, 2);
                        let owned = borrowed.try_to_owned().unwrap();
                        drop(request);
                        assert_eq!(owned.lines.len(), 2);
                    }
                });
            }
        });
        assert!(engine.stats().live_requests <= 2);
        assert!(engine.stats().idle_bytes <= 16 << 20);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn concurrent_different_widths_match_serial_with_one_two_four_and_eight_workers() {
        use crate::{cpu::CpuOptions, image::PixelFormat};

        let original = crate::ocr::tests::two_lines();
        let mut baseline = crate::ocr::tests::synthetic_ocr(CpuContext::serial());
        let cases = [96, 128, 160].map(|width| {
            let stride = width * 3 + 5;
            let mut pixels = vec![0; stride * 96];
            for y in 0..96 {
                for x in 0..width {
                    let source = (y * 96 + x * 96 / width) * 3;
                    let destination = y * stride + x * 3;
                    pixels[destination..destination + 3]
                        .copy_from_slice(&original[source..source + 3]);
                }
            }
            let image = ImageView::new(&pixels, width, 96, stride, PixelFormat::Bgr24).unwrap();
            let expected = [false, true].map(|capture| {
                let result = baseline
                    .run(
                        image,
                        crate::RunOptions {
                            capture_spans: capture,
                        },
                    )
                    .unwrap();
                assert_eq!(result.detected_count, 2);
                serde_json::to_value(result.try_to_owned().unwrap()).unwrap()
            });
            (pixels, width, stride, expected)
        });
        for budget in [1, 2, 4, 8] {
            let cpu = Arc::new(CpuContext::new(budget).unwrap());
            let mut template = crate::ocr::tests::synthetic_ocr(cpu.clone());
            template
                .set_execution_context(
                    cpu.clone(),
                    CpuOptions {
                        line_workers: budget,
                        preprocess_workers: budget,
                        detector_threads: budget,
                    },
                )
                .unwrap();
            let engine = OcrEngine::new(
                std::sync::Arc::new((template).plan().clone()),
                EngineLimits {
                    max_requests: 3,
                    max_idle_bytes: 64 << 20,
                },
            )
            .unwrap();
            let admitted = std::sync::Barrier::new(3);
            std::thread::scope(|scope| {
                for caller in 0..3 {
                    let engine = &engine;
                    let cpu = &cpu;
                    let admitted = &admitted;
                    let cases = &cases;
                    scope.spawn(move || {
                        for iteration in 0..6 {
                            let (pixels, width, stride, expected) =
                                &cases[(caller + iteration) % cases.len()];
                            let capture = iteration % 2 != 0;
                            let image =
                                ImageView::new(pixels, *width, 96, *stride, PixelFormat::Bgr24)
                                    .unwrap();
                            let mut request = engine.request().unwrap();
                            // The first round guarantees three admitted requests;
                            // later rounds allow callers to reshape independently.
                            if iteration == 0 {
                                admitted.wait();
                            }
                            assert!(Arc::ptr_eq(request.cpu_context(), cpu));
                            assert_eq!(request.cpu_budgets().line_workers(), budget);
                            assert_eq!(request.cpu_budgets().detector_threads(), budget);
                            let owned = request
                                .run(
                                    image,
                                    crate::RunOptions {
                                        capture_spans: capture,
                                    },
                                )
                                .unwrap()
                                .try_to_owned()
                                .unwrap();
                            drop(request);
                            assert_eq!(
                                serde_json::to_value(owned).unwrap(),
                                expected[usize::from(capture)],
                                "CPU budget {budget}, caller {caller}, width {width}"
                            );
                            assert!(engine.stats().live_requests <= 3);
                        }
                    });
                }
            });
            assert_eq!(engine.stats().idle_requests, 3);
            assert!(engine.stats().idle_bytes <= 64 << 20);
        }
    }
}
