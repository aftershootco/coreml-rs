use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};

pub struct TrackingAllocator;
static POINTER: AtomicUsize = AtomicUsize::new(0);
static SIZE: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);
static WRONG_LAYOUT: AtomicBool = AtomicBool::new(false);

// Each test runs in its own process and watches one Vec<u8> at a time.
// Track realloc too: Vec::into_boxed_slice may shrink or move the allocation.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        System.alloc(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = System.realloc(ptr, layout, size);
        if POINTER.load(SeqCst) == ptr as usize && !result.is_null() {
            SIZE.store(size, SeqCst);
            POINTER.store(result as usize, SeqCst);
        }
        result
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if POINTER
            .compare_exchange(ptr as usize, 0, SeqCst, SeqCst)
            .is_ok()
        {
            let size = SIZE.load(SeqCst);
            WRONG_LAYOUT.store(layout.size() != size || layout.align() != 1, SeqCst);
            FREES.fetch_add(1, SeqCst);
            // Report a bad layout without passing it on to the real allocator.
            System.dealloc(ptr, Layout::from_size_align_unchecked(size, 1));
        } else {
            System.dealloc(ptr, layout);
        }
    }
}

pub fn watch(buffer: &Vec<u8>) {
    assert_eq!(POINTER.load(SeqCst), 0, "previous allocation is still live");
    assert!(buffer.capacity() > 0);
    FREES.store(0, SeqCst);
    WRONG_LAYOUT.store(false, SeqCst);
    SIZE.store(buffer.capacity(), SeqCst);
    POINTER.store(buffer.as_ptr() as usize, SeqCst);
}

pub fn assert_live() {
    assert_eq!(
        FREES.load(SeqCst),
        0,
        "specification freed before model drop"
    );
}

pub fn assert_freed() {
    assert_eq!(
        FREES.load(SeqCst),
        1,
        "specification must be freed exactly once"
    );
    assert!(
        !WRONG_LAYOUT.load(SeqCst),
        "allocation freed with the wrong layout"
    );
}
