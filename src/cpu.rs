//! One bounded CPU executor shared by requests and their nested kernels.
use crate::image::{ImageError, ImageResult};
use crate::runtime::{Backend, RuntimeError, RuntimeResult};
use std::{
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicUsize, Ordering},
    },
};

thread_local! {
    // The guard borrows its CpuContext, so these identity-only addresses cannot
    // move or be reused while registered. No pointer is ever dereferenced.
    static INLINE_CONTEXTS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static ACTIVE_REQUESTS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

pub(crate) struct ActiveRequest<'a> {
    context: &'a CpuContext,
    _calling_thread: PhantomData<Rc<()>>,
}

/// Counts callers before pool admission, including requests waiting for workers.
/// Unlike ActiveRequest, this guard may cross the pool boundary: it owns no TLS.
pub(crate) struct RequestDemand<'a>(&'a CpuContext);
impl Drop for RequestDemand<'_> {
    fn drop(&mut self) {
        self.0.request_demand.fetch_sub(1, Ordering::AcqRel);
    }
}
impl Drop for ActiveRequest<'_> {
    fn drop(&mut self) {
        let identity = std::ptr::from_ref(self.context) as usize;
        ACTIVE_REQUESTS.with_borrow_mut(|contexts| {
            let index = contexts
                .iter()
                .rposition(|&context| context == identity)
                .expect("active OCR request context is registered");
            contexts.remove(index);
        });
    }
}

/// Keeps inline work on its calling thread while sharing one admission slot.
/// MutexGuard also prevents the owner registration from moving to a new thread.
pub(crate) struct InlineAdmission<'a> {
    owner: Option<usize>,
    _lock: Option<MutexGuard<'a, ()>>,
}
impl Drop for InlineAdmission<'_> {
    fn drop(&mut self) {
        if let Some(owner) = self.owner {
            INLINE_CONTEXTS.with_borrow_mut(|owners| {
                let index = owners
                    .iter()
                    .rposition(|&context| context == owner)
                    .expect("inline CPU admission owner is registered");
                owners.remove(index);
            });
        }
    }
}

/// CPU budgets. Zero selects each stage's automatic budget.
/// Positive values are maxima capped to the context capacity and 16.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuOptions {
    pub line_workers: usize,
    pub preprocess_workers: usize,
    pub detector_threads: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuBudgets {
    capacity: usize,
    line_workers: usize,
    preprocess_workers: usize,
    detector_threads: usize,
    recognizer_base_threads: usize,
}
impl CpuOptions {
    pub fn resolve(self, capacity: usize, channels_last: bool) -> ImageResult<CpuBudgets> {
        if !(1..=1024).contains(&capacity) {
            return Err(ImageError::new("CPU capacity must be within 1..=1024"));
        }
        let lines = if self.line_workers == 0 {
            capacity.min(4)
        } else {
            cap_stage_threads(self.line_workers, capacity)
        };
        let preprocess = if self.preprocess_workers == 0 {
            capacity.min(16)
        } else {
            cap_stage_threads(self.preprocess_workers, capacity)
        };
        let detector = if self.detector_threads == 0 {
            capacity.min(if channels_last { 16 } else { 8 })
        } else {
            cap_stage_threads(self.detector_threads, capacity)
        };
        Ok(CpuBudgets {
            capacity,
            line_workers: lines,
            preprocess_workers: preprocess,
            detector_threads: detector,
            recognizer_base_threads: (capacity / lines).clamp(1, 8),
        })
    }
}
pub(crate) fn cap_stage_threads(requested: usize, capacity: usize) -> usize {
    requested.min(capacity).min(16)
}
impl CpuBudgets {
    pub fn capacity(self) -> usize {
        self.capacity
    }
    pub fn line_workers(self) -> usize {
        self.line_workers
    }
    pub fn preprocess_workers(self) -> usize {
        self.preprocess_workers
    }
    pub fn detector_threads(self) -> usize {
        self.detector_threads
    }
    pub fn recognizer_base_threads(self) -> usize {
        self.recognizer_base_threads
    }

    /// AVX2/FMA and native ARM recognition benefit from independent lines
    /// instead of nested kernels when a complete eight-unit wave is available.
    /// On Windows AVX2, dense pages on 24–31-thread contexts use sixteen
    /// single-threaded owners to avoid three-way nested kernel scheduling.
    /// Depend only on page geometry and configuration, never on request timing:
    /// stable ownership keeps worker-private shape caches and arenas reusable.
    pub(crate) fn for_recognition_units(
        self,
        units: usize,
        automatic: bool,
        backend: Backend,
    ) -> Self {
        if !automatic
            || !matches!(backend, Backend::NeonFma | Backend::Avx2Fma)
            || self.capacity < 8
            || units < 8
        {
            return self;
        }
        let owners = if cfg!(all(target_os = "windows", target_arch = "x86_64"))
            && backend == Backend::Avx2Fma
            && (24..32).contains(&self.capacity)
            && units >= 32
        {
            16
        } else {
            8
        };
        Self {
            line_workers: owners,
            recognizer_base_threads: (self.capacity / owners).clamp(1, 8),
            ..self
        }
    }

    /// Static sparse-page boost; scheduling progress never changes arithmetic.
    pub fn recognizer_threads(self, line_count: usize) -> usize {
        let active = self.line_workers.min(line_count).max(1);
        (self.recognizer_base_threads * self.line_workers / active)
            .clamp(self.recognizer_base_threads, self.capacity.min(16))
    }

    /// Preserve line ownership and buffers; only reduce nested kernel workers.
    pub(crate) fn recognizer_threads_for_capacity(
        self,
        line_count: usize,
        request_capacity: usize,
    ) -> usize {
        let active = self.line_workers.min(line_count).max(1);
        self.recognizer_threads(line_count)
            .min((request_capacity / active).max(1))
    }

    pub(crate) fn extraction_options(
        self,
        request_capacity: usize,
        automatic_line_workers: bool,
    ) -> CpuOptions {
        let available = request_capacity.max(1);
        // Extraction has no recognition work. Automatic classifiers can use the
        // request's CPU share without creating extra recognition owners, while
        // positive line_workers still bounds both stages as configured.
        let classifiers = if automatic_line_workers {
            cap_stage_threads(available, self.capacity)
        } else {
            self.line_workers.min(available)
        };
        CpuOptions {
            line_workers: classifiers,
            preprocess_workers: self.preprocess_workers.min(available),
            detector_threads: self.detector_threads.min(available),
        }
    }
}

/// A private fixed-size pool. Capacity one executes on the calling thread.
/// Production stages sharing its Arc share the cap across simultaneous requests.
/// Trace orchestration and user callbacks remain on their calling thread.
pub struct CpuContext {
    capacity: usize,
    request_demand: AtomicUsize,
    inline_admission: Mutex<()>,
    #[cfg(feature = "parallel")]
    pool: Option<rayon::ThreadPool>,
}
impl std::fmt::Debug for CpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuContext")
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}
// Only successful construction is cached; serialize initializers to avoid
// temporarily creating multiple process-wide pools under concurrent callers.
fn shared_context(
    cache: &Mutex<Option<Arc<CpuContext>>>,
    initialize: impl FnOnce() -> RuntimeResult<CpuContext>,
) -> RuntimeResult<Arc<CpuContext>> {
    let mut cached = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(context) = cached.as_ref() {
        return Ok(context.clone());
    }
    let context = Arc::new(initialize()?);
    *cached = Some(context.clone());
    Ok(context)
}
impl CpuContext {
    pub fn new(capacity: usize) -> RuntimeResult<Self> {
        if capacity == 0 || capacity > 1024 {
            return Err(RuntimeError::new("CPU capacity must be within 1..=1024"));
        }
        #[cfg(feature = "parallel")]
        {
            let pool = if capacity > 1 {
                Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(capacity)
                        .thread_name(|i| format!("simd-ocr-{i}"))
                        .build()
                        .map_err(|e| {
                            RuntimeError::caused_by(
                                crate::error::ErrorKind::Allocation,
                                "CPU pool creation",
                                e,
                            )
                        })?,
                )
            } else {
                None
            };
            Ok(Self {
                capacity,
                request_demand: AtomicUsize::new(0),
                inline_admission: Mutex::new(()),
                pool,
            })
        }
        #[cfg(not(feature = "parallel"))]
        {
            if capacity != 1 {
                return Err(RuntimeError::new(
                    "multiple CPU threads require the parallel feature",
                ));
            }
            Ok(Self {
                capacity,
                request_demand: AtomicUsize::new(0),
                inline_admission: Mutex::new(()),
            })
        }
    }
    pub fn automatic() -> RuntimeResult<Self> {
        #[cfg(feature = "parallel")]
        let capacity = std::thread::available_parallelism().map_or(1, usize::from);
        #[cfg(not(feature = "parallel"))]
        let capacity = 1;
        Self::new(capacity)
    }
    /// Default engines and requests share one process-wide worker capacity.
    pub fn shared_default() -> RuntimeResult<Arc<Self>> {
        static DEFAULT: Mutex<Option<Arc<CpuContext>>> = Mutex::new(None);
        shared_context(&DEFAULT, Self::automatic)
    }
    pub fn serial() -> Arc<Self> {
        Arc::new(Self::new(1).expect("inline CPU context needs no resources"))
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub(crate) fn register_request(&self) -> RequestDemand<'_> {
        self.request_demand.fetch_add(1, Ordering::AcqRel);
        RequestDemand(self)
    }
    /// A stage-local scheduling hint, not a reservation. The fixed pool remains
    /// the hard cap even when requests arrive or finish after this snapshot.
    pub(crate) fn request_capacity(&self) -> usize {
        (self.capacity / self.request_demand.load(Ordering::Acquire).max(1)).max(1)
    }
    fn inline_owned(&self) -> bool {
        let identity = std::ptr::from_ref(self) as usize;
        INLINE_CONTEXTS.with_borrow(|owners| owners.contains(&identity))
    }
    pub(crate) fn enter_request(&self) -> ActiveRequest<'_> {
        let identity = std::ptr::from_ref(self) as usize;
        ACTIVE_REQUESTS.with_borrow_mut(|contexts| contexts.push(identity));
        ActiveRequest {
            context: self,
            _calling_thread: PhantomData,
        }
    }
    pub(crate) fn is_active_request(&self) -> bool {
        let identity = std::ptr::from_ref(self) as usize;
        ACTIVE_REQUESTS.with_borrow(|contexts| contexts.contains(&identity))
    }
    /// Capacity one has no worker thread. Its calling thread holds the shared
    /// admission mutex, and nested work on this same context reuses ownership.
    pub(crate) fn enter_inline(&self) -> InlineAdmission<'_> {
        if self.capacity != 1 || self.inline_owned() {
            return InlineAdmission {
                owner: None,
                _lock: None,
            };
        }
        let lock = self
            .inline_admission
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let identity = std::ptr::from_ref(self) as usize;
        INLINE_CONTEXTS.with_borrow_mut(|owners| owners.push(identity));
        InlineAdmission {
            owner: Some(identity),
            _lock: Some(lock),
        }
    }
    pub fn is_worker_thread(&self) -> bool {
        if self.capacity == 1 {
            return self.inline_owned();
        }
        #[cfg(feature = "parallel")]
        if let Some(pool) = &self.pool {
            return pool.current_thread_index().is_some();
        }
        false
    }
    #[cfg(test)]
    pub(crate) fn assert_production_thread(&self, stage: &str) {
        assert!(
            self.is_worker_thread(),
            "{stage} production work escaped its CPU context on {:?}",
            std::thread::current().id()
        );
    }
    pub fn install<T: Send>(&self, operation: impl FnOnce() -> T + Send) -> T {
        #[cfg(feature = "parallel")]
        if let Some(pool) = &self.pool {
            return pool.install(operation);
        }
        let _inline = self.enter_inline();
        operation()
    }
    /// Bounded scoped jobs; nested calls stay on this same fixed-size pool.
    pub fn for_each(&self, count: usize, budget: usize, body: impl Fn(usize) + Sync) {
        let workers = count.min(budget.max(1)).min(self.capacity);
        if workers <= 1 {
            self.install(|| {
                for i in 0..count {
                    body(i);
                }
            });
            #[cfg(feature = "parallel")]
            return;
        }
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            self.install(|| {
                (0..workers).into_par_iter().for_each(|worker| {
                    for i in (worker..count).step_by(workers) {
                        body(i);
                    }
                })
            });
        }
    }
    /// Each item is exclusively borrowed and all jobs join before returning.
    pub fn for_each_mut<T: Send>(
        &self,
        items: &mut [T],
        budget: usize,
        body: impl Fn(usize, &mut T) + Sync,
    ) {
        // Exact floor boundaries keep every allowed job nonempty. Ceiling
        // chunk sizes underfilled the pool: nine crop bands with a budget of
        // eight produced only five jobs. Reuse the allocation-free partition
        // used by kernels, including its serial and shared-pool admission.
        let count = items.len();
        crate::runtime::parallel::aligned_bands(
            items,
            1,
            count,
            budget.max(1),
            1,
            self,
            |band, first| {
                for (offset, item) in band.iter_mut().enumerate() {
                    body(first + offset, item);
                }
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_initialization_retries_failure_and_publishes_one_success() {
        let cache = Mutex::new(None);
        let error =
            shared_context(&cache, || Err(RuntimeError::new("transient failure"))).unwrap_err();
        assert!(error.to_string().contains("transient failure"));
        let calls = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let handles = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        shared_context(&cache, || {
                            calls.fetch_add(1, Ordering::SeqCst);
                            CpuContext::new(1)
                        })
                        .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            let contexts = handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>();
            assert!(contexts.iter().all(|c| Arc::ptr_eq(c, &contexts[0])));
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn automatic_recognition_expands_only_supported_complete_eight_unit_waves() {
        for capacity in [1, 4, 7, 8, 10, 16, 23, 32, 48] {
            let base = CpuOptions::default().resolve(capacity, true).unwrap();
            for units in [0, 1, 4, 7, 8, 9, 31, 32, 33, 69] {
                for backend in [Backend::NeonFma, Backend::Avx2Fma] {
                    assert_eq!(base.for_recognition_units(units, false, backend), base);
                    let selected = base.for_recognition_units(units, true, backend);
                    if capacity < 8 || units < 8 {
                        assert_eq!(selected, base);
                    } else {
                        assert_eq!(selected.line_workers, 8);
                        assert_eq!(selected.recognizer_base_threads, (capacity / 8).clamp(1, 8));
                        assert_eq!(selected.detector_threads, base.detector_threads);
                        assert_eq!(selected.preprocess_workers, base.preprocess_workers);
                        // Demand can reduce nested work without changing the
                        // stable owners or exceeding the shared pool's cap.
                        for share in [1, 2, 4, 8] {
                            assert_eq!(selected.recognizer_threads_for_capacity(units, share), 1);
                            assert_eq!(selected.line_workers, 8);
                        }
                    }
                }
                for backend in [
                    Backend::Scalar,
                    Backend::Portable4,
                    Backend::Avx,
                    Backend::AvxFma,
                    Backend::Avx512,
                ] {
                    assert_eq!(base.for_recognition_units(units, true, backend), base);
                }
            }
        }
        for line_workers in [1, 2, 4, 8, 12, 16] {
            let explicit = CpuOptions {
                line_workers,
                ..Default::default()
            }
            .resolve(24, true)
            .unwrap();
            for backend in [Backend::NeonFma, Backend::Avx2Fma] {
                assert_eq!(explicit.for_recognition_units(69, false, backend), explicit);
            }
        }
    }

    #[test]
    fn dense_windows_avx2_ownership_has_bounded_capacity_and_geometry() {
        let windows_avx2 = cfg!(all(target_os = "windows", target_arch = "x86_64"));
        for (capacity, units, windows_owners) in [
            (23, 32, 8),
            (24, 31, 8),
            (24, 32, 16),
            (24, 69, 16),
            (31, 32, 16),
            (32, 32, 8),
        ] {
            let base = CpuOptions::default().resolve(capacity, true).unwrap();
            let selected = base.for_recognition_units(units, true, Backend::Avx2Fma);
            let owners = if windows_avx2 { windows_owners } else { 8 };
            assert_eq!(selected.line_workers, owners);
            assert_eq!(selected.recognizer_base_threads, capacity / owners);
            assert_eq!(selected.preprocess_workers, base.preprocess_workers);
            assert_eq!(selected.detector_threads, base.detector_threads);
            for share in [1, 8, 16, capacity] {
                assert!(
                    selected.recognizer_threads_for_capacity(units, share) <= capacity / owners
                );
                assert_eq!(selected.line_workers, owners);
            }
            let neon = base.for_recognition_units(units, true, Backend::NeonFma);
            assert_eq!(neon.line_workers, 8);
            assert_eq!(
                base.for_recognition_units(units, false, Backend::Avx2Fma),
                base
            );
        }
    }

    #[test]
    fn request_shares_reduce_nested_work_without_reassigning_lines() {
        let budgets = CpuOptions::default().resolve(8, true).unwrap();
        for (share, lines, threads) in [
            (8, 1, 8),
            (8, 4, 2),
            (4, 1, 4),
            (4, 4, 1),
            (2, 1, 2),
            (2, 4, 1),
            (1, 69, 1),
        ] {
            assert_eq!(
                budgets.recognizer_threads_for_capacity(lines, share),
                threads
            );
            let extraction = budgets.extraction_options(share, true);
            assert_eq!(extraction.line_workers, share);
            assert_eq!(extraction.detector_threads, share);
            assert_eq!(extraction.preprocess_workers, share);
        }
        let explicit = CpuOptions {
            line_workers: 2,
            preprocess_workers: 1,
            detector_threads: 2,
        }
        .resolve(8, true)
        .unwrap();
        assert_eq!(explicit.extraction_options(4, false).preprocess_workers, 1);
        assert_eq!(explicit.extraction_options(4, false).detector_threads, 2);
        assert_eq!(explicit.recognizer_threads_for_capacity(2, 4), 2);
        assert_eq!(explicit.recognizer_threads_for_capacity(1, 4), 4);
    }

    #[test]
    fn automatic_classifiers_use_request_capacity_and_explicit_line_maxima_are_preserved() {
        for capacity in [1, 4, 8, 32] {
            for requested in [0, 1, 2, 17, usize::MAX] {
                let budgets = CpuOptions {
                    line_workers: requested,
                    ..Default::default()
                }
                .resolve(capacity, true)
                .unwrap();
                let recognition_workers = budgets.line_workers;
                for share in [0, 1, 2, 8, 32, usize::MAX] {
                    let extraction = budgets.extraction_options(share, requested == 0);
                    let expected = if requested == 0 {
                        capacity.min(share.max(1)).min(16)
                    } else {
                        recognition_workers.min(share.max(1))
                    };
                    assert_eq!(extraction.line_workers, expected);
                    assert_eq!(budgets.line_workers, recognition_workers);
                }
            }
        }
        let automatic = CpuOptions::default().resolve(8, true).unwrap();
        assert_eq!(automatic.line_workers, 4);
        assert_eq!(automatic.extraction_options(8, true).line_workers, 8);
        assert_eq!(automatic.extraction_options(8, false).line_workers, 4);
    }

    #[test]
    fn request_demand_includes_waiting_callers_and_recovers_after_unwind() {
        let capacity = if cfg!(feature = "parallel") { 8 } else { 1 };
        let cpu = CpuContext::new(capacity).unwrap();
        let other = CpuContext::serial();
        let registered = std::sync::Barrier::new(2);
        let release = std::sync::Barrier::new(2);
        let first = cpu.register_request();
        assert_eq!(cpu.request_capacity(), capacity);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                // Registration must be visible before install can acquire a
                // worker; this caller deliberately has not entered the pool.
                let _queued = cpu.register_request();
                registered.wait();
                release.wait();
            });
            registered.wait();
            assert_eq!(cpu.request_capacity(), (capacity / 2).max(1));
            assert_eq!(other.request_demand.load(Ordering::Acquire), 0);
            release.wait();
        });
        assert_eq!(cpu.request_capacity(), capacity);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _failing = cpu.register_request();
                assert_eq!(cpu.request_capacity(), (capacity / 2).max(1));
                panic!("request failed before or during admission");
            }))
            .is_err()
        );
        assert_eq!(cpu.request_demand.load(Ordering::Acquire), 1);
        drop(first);
        assert_eq!(cpu.request_demand.load(Ordering::Acquire), 0);
        assert_eq!(cpu.request_capacity(), capacity);
    }

    #[test]
    fn original_budgets_and_sparse_boost() {
        for (cpu, lines, rec) in [
            (1, 1, 1),
            (2, 2, 1),
            (3, 3, 1),
            (8, 4, 2),
            (16, 4, 4),
            (32, 4, 8),
        ] {
            let b = CpuOptions::default().resolve(cpu, true).unwrap();
            assert_eq!((b.line_workers, b.recognizer_base_threads), (lines, rec));
            assert_eq!(b.preprocess_workers, cpu.min(16));
            assert_eq!(b.detector_threads, cpu.min(16));
            assert!(b.recognizer_threads(1) <= cpu.min(16));
            assert_eq!(b.recognizer_threads(lines), rec);
        }
        assert_eq!(
            CpuOptions::default()
                .resolve(32, false)
                .unwrap()
                .detector_threads,
            8
        );
        assert!(CpuOptions::default().resolve(0, false).is_err());
    }
    #[test]
    fn explicit_stage_budget_caps_each_option_without_changing_automatic_siblings() {
        for capacity in [1, 4, 32] {
            for channels_last in [false, true] {
                let automatic = CpuOptions::default()
                    .resolve(capacity, channels_last)
                    .unwrap();
                assert_eq!(
                    automatic,
                    CpuBudgets {
                        capacity,
                        line_workers: capacity.min(4),
                        preprocess_workers: capacity.min(16),
                        detector_threads: capacity.min(if channels_last { 16 } else { 8 }),
                        recognizer_base_threads: (capacity / capacity.min(4)).clamp(1, 8),
                    }
                );
                for requested in [17, usize::MAX] {
                    for field in 0..3 {
                        let mut options = CpuOptions::default();
                        let mut expected = automatic;
                        let capped = requested.min(capacity).min(16);
                        match field {
                            0 => {
                                options.line_workers = requested;
                                expected.line_workers = capped;
                                expected.recognizer_base_threads = (capacity / capped).clamp(1, 8);
                            }
                            1 => {
                                options.preprocess_workers = requested;
                                expected.preprocess_workers = capped;
                            }
                            _ => {
                                options.detector_threads = requested;
                                expected.detector_threads = capped;
                            }
                        }
                        assert_eq!(options.resolve(capacity, channels_last).unwrap(), expected);
                    }
                }
            }
        }
        assert!(
            CpuOptions {
                line_workers: usize::MAX,
                preprocess_workers: usize::MAX,
                detector_threads: usize::MAX,
            }
            .resolve(0, true)
            .is_err()
        );
    }
    #[test]
    fn inline_executes_on_the_calling_thread_and_empty_jobs_are_safe() {
        let cpu = CpuContext::new(1).unwrap();
        let caller = std::thread::current().id();
        assert_eq!(cpu.install(|| std::thread::current().id()), caller);
        let mut values = [0; 17];
        cpu.for_each_mut(&mut values, 16, |i, v| *v = i);
        assert_eq!(values, std::array::from_fn::<_, 17, _>(|i| i));
        cpu.for_each(0, 0, |_| panic!("empty executor dispatched a task"));
        assert!(CpuContext::new(0).is_err());
        assert!(CpuContext::new(1025).is_err());
    }
    #[test]
    fn mutable_jobs_cover_irregular_lengths_and_obey_requested_parallel_budget() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let capacity = if cfg!(feature = "parallel") { 8 } else { 1 };
        let cpu = CpuContext::new(capacity).unwrap();
        for count in [0, 1, 7, 8, 9, 15, 16, 17, 33] {
            for budget in [0, 1, 2, 4, 8, 16] {
                let active = AtomicUsize::new(0);
                let peak = AtomicUsize::new(0);
                let visited: Vec<_> = (0..count).map(|_| AtomicUsize::new(0)).collect();
                let mut values = vec![usize::MAX; count];
                cpu.for_each_mut(&mut values, budget, |index, value| {
                    assert!(cpu.is_worker_thread());
                    let concurrent = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(concurrent, Ordering::SeqCst);
                    visited[index].fetch_add(1, Ordering::Relaxed);
                    *value = index;
                    active.fetch_sub(1, Ordering::SeqCst);
                });
                assert_eq!(values, (0..count).collect::<Vec<_>>());
                assert!(visited.iter().all(|n| n.load(Ordering::Relaxed) == 1));
                assert!(peak.load(Ordering::SeqCst) <= budget.max(1).min(capacity));
            }
        }
    }
    #[test]
    fn concurrent_inline_work_is_serialized_and_same_context_calls_are_reentrant() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let cpu = CpuContext::new(1).unwrap();
        let ready = std::sync::Barrier::new(4);
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let cpu = &cpu;
                let ready = &ready;
                let active = &active;
                let peak = &peak;
                scope.spawn(move || {
                    let caller = std::thread::current().id();
                    ready.wait();
                    for _ in 0..6 {
                        cpu.install(|| {
                            assert_eq!(std::thread::current().id(), caller);
                            assert!(cpu.is_worker_thread());
                            let simultaneous = active.fetch_add(1, Ordering::SeqCst) + 1;
                            peak.fetch_max(simultaneous, Ordering::SeqCst);
                            cpu.install(|| assert_eq!(active.load(Ordering::SeqCst), 1));
                            cpu.for_each(2, 1, |_| assert_eq!(active.load(Ordering::SeqCst), 1));
                            let mut values = [0, 0];
                            cpu.for_each_mut(&mut values, 1, |index, value| *value = index);
                            assert_eq!(values, [0, 1]);
                            std::thread::sleep(std::time::Duration::from_millis(1));
                            active.fetch_sub(1, Ordering::SeqCst);
                        });
                    }
                    assert!(!cpu.is_worker_thread());
                });
            }
        });
        assert_eq!(peak.load(Ordering::SeqCst), 1);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn inline_ownership_and_admission_recover_after_caught_panics() {
        let cpu = CpuContext::new(1).unwrap();
        cpu.install(|| {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    cpu.install(|| panic!("nested inline panic"));
                }))
                .is_err()
            );
            assert!(cpu.is_worker_thread());
            cpu.install(|| assert!(cpu.is_worker_thread()));
        });
        assert!(!cpu.is_worker_thread());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cpu.install(|| panic!("outer inline panic"));
            }))
            .is_err()
        );
        assert!(!cpu.is_worker_thread());
        assert_eq!(cpu.install(|| 7), 7);
    }
    #[test]
    fn request_markers_are_context_specific_nested_and_unwind_safe() {
        let first = CpuContext::new(1).unwrap();
        let second = CpuContext::new(1).unwrap();
        assert!(!first.is_active_request());
        {
            let _outer = first.enter_request();
            assert!(first.is_active_request());
            assert!(!first.is_worker_thread());
            assert!(!second.is_active_request());
            {
                let _inner = first.enter_request();
                assert!(first.is_active_request());
            }
            assert!(first.is_active_request());
            std::thread::scope(|scope| {
                scope.spawn(|| assert!(!first.is_active_request()));
            });
        }
        assert!(!first.is_active_request());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _active = first.enter_request();
                panic!("active request panic");
            }))
            .is_err()
        );
        assert!(!first.is_active_request());
    }
    #[cfg(feature = "parallel")]
    #[test]
    fn nested_jobs_share_the_fixed_pool() {
        use std::sync::Mutex;
        let cpu = CpuContext::new(2).unwrap();
        let ids = Mutex::new(std::collections::HashSet::new());
        cpu.for_each(16, 2, |_| {
            cpu.for_each(16, 2, |_| {
                ids.lock().unwrap().insert(std::thread::current().id());
            })
        });
        assert!(ids.lock().unwrap().len() <= 2);
    }
}
