use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

#[repr(align(32))]
pub struct AlignedF32 {
    data: NonNull<f32>,
    len: usize,
}

impl AlignedF32 {
    pub fn new(len: usize) -> Self {
        // Calculate the layout based on the number of elements and the required alignment
        let layout = Layout::from_size_align(len * std::mem::size_of::<f32>(), 32)
            .expect("Failed to create layout");
        // Allocate memory with the calculated layout
        let data = unsafe { alloc(layout) } as *mut f32;
        let data = NonNull::new(data).expect("Failed to allocate memory");
        AlignedF32 { data, len }
    }

    pub fn as_slice(&self) -> &[f32] {
        // Create a slice from the raw pointer
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // Create a mutable slice from the raw pointer
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }
}

impl Drop for AlignedF32 {
    fn drop(&mut self) {
        // Calculate the layout based on the number of elements and the alignment
        let layout = Layout::from_size_align(self.len * std::mem::size_of::<f32>(), 32)
            .expect("Failed to create layout");
        // Deallocate the memory
        unsafe { dealloc(self.data.as_ptr() as *mut u8, layout) };
    }
}