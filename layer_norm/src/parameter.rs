#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct Parameter {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
}

impl Parameter {
    pub fn new(shape: &[usize], initializer: fn(&mut [f32])) -> Self {
        let size = shape.iter().product();
        let mut data = vec![0.0; size];
        initializer(&mut data);
        let grad = vec![0.0; size];
        Self { data, grad }
    }

    pub fn zero_grad(&mut self) {
        let zero = unsafe { _mm256_setzero_ps() };
        let mut i = 0;
        while i + 8 <= self.grad.len() {
            unsafe {
                _mm256_storeu_ps(self.grad.as_mut_ptr().add(i), zero);
            }
            i += 8;
        }
        for j in i..self.grad.len() {
            self.grad[j] = 0.0;
        }
    }

    pub fn add_grad(&mut self, grad: &[f32]) {
        assert_eq!(self.grad.len(), grad.len());
        let mut i = 0;
        while i + 8 <= self.grad.len() {
            let self_grad = unsafe { _mm256_loadu_ps(self.grad.as_ptr().add(i)) };
            let grad_wide = unsafe { _mm256_loadu_ps(grad.as_ptr().add(i)) };
            let result = unsafe { _mm256_add_ps(self_grad, grad_wide) };
            unsafe {
                _mm256_storeu_ps(self.grad.as_mut_ptr().add(i), result);
            }
            i += 8;
        }
        for j in i..self.grad.len() {
            self.grad[j] += grad[j];
        }
    }