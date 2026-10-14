//! Optional executable-only allocator accounting; the library remains safe Rust.
#[cfg(feature = "heap-profiler")]
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};
#[cfg(feature = "heap-profiler")]
struct Counting;
#[cfg(feature = "heap-profiler")]
static LIVE: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "heap-profiler")]
static PEAK: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "heap-profiler")]
static CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "heap-profiler")]
static BYTES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "heap-profiler")]
fn allocated(size: usize) {
    CALLS.fetch_add(1, Relaxed);
    BYTES.fetch_add(size, Relaxed);
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
}
#[cfg(feature = "heap-profiler")]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            allocated(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            allocated(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let p = unsafe { System.realloc(p, l, n) };
        if !p.is_null() {
            LIVE.fetch_sub(l.size(), Relaxed);
            allocated(n);
        }
        p
    }
}
#[cfg(feature = "heap-profiler")]
#[global_allocator]
static ALLOCATOR: Counting = Counting;
#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct Counts {
    pub warm_allocation_count: Option<usize>,
    pub warm_allocation_bytes: Option<usize>,
    pub warm_retained_heap_bytes: Option<usize>,
    pub warm_peak_heap_growth_bytes: Option<usize>,
}
pub fn begin() -> usize {
    #[cfg(feature = "heap-profiler")]
    {
        let n = LIVE.load(Relaxed);
        PEAK.store(n, Relaxed);
        CALLS.store(0, Relaxed);
        BYTES.store(0, Relaxed);
        n
    }
    #[cfg(not(feature = "heap-profiler"))]
    {
        0
    }
}
pub fn end(baseline: usize) -> Counts {
    #[cfg(feature = "heap-profiler")]
    {
        Counts {
            warm_allocation_count: Some(CALLS.load(Relaxed)),
            warm_allocation_bytes: Some(BYTES.load(Relaxed)),
            warm_retained_heap_bytes: Some(LIVE.load(Relaxed).saturating_sub(baseline)),
            warm_peak_heap_growth_bytes: Some(PEAK.load(Relaxed).saturating_sub(baseline)),
        }
    }
    #[cfg(not(feature = "heap-profiler"))]
    {
        let _ = baseline;
        Counts::default()
    }
}
