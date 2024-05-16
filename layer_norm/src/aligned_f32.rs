use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

#[repr(align(32))]
pub struct AlignedF32 {
    data: NonNull<f32>,
    len: usize,
}

impl AlignedF32 {
    pub fn new(len: usize) -> Self {
        let layout = Layout::array::<f32>(len).unwrap();
        let data = unsafe { alloc(layout) } as *mut f32;
        let data = NonNull::new(data).expect("Failed to allocate memory");
        AlignedF32 { data, len }
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }
}

impl Drop for AlignedF32 {
    fn drop(&mut self) {
        let layout = Layout::array::<f32>(self.len).unwrap();
        unsafe { dealloc(self.data.as_ptr() as *mut u8, layout) };
    }
}