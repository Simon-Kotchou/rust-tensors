#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn gelu(x: &mut [f32]) {
    const SQRT2_OVER_PI: f32 = 0.7978845608028654;
    const COEF0: f32 = 0.044715;

    for i in 0..x.len() {
        let x_i = x[i];
        let cdf = 0.5 * (1.0 + (x_i * SQRT2_OVER_PI * (1.0 + COEF0 * x_i * x_i)).tanh());
        x[i] = x_i * cdf;
    }
}

pub fn silu(x: &mut [f32]) {
    for i in 0..x.len() {
        x[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

pub fn relu(x: &mut [f32]) {
    let zero = unsafe { _mm256_setzero_ps() };
    let mut i = 0;
    while i + 8 <= x.len() {
        let x_wide = unsafe { _mm256_loadu_ps(&x[i]) };
        let relu_wide = unsafe { _mm256_max_ps(x_wide, zero) };
        unsafe { _mm256_storeu_ps(&mut x[i], relu_wide) };
        i += 8;
    }
    for j in i..x.len() {
        x[j] = x[j].max(0.0);
    }
}

pub fn softmax(x: &mut [f32]) {
    let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exp_sum = 0.0;

    for i in 0..x.len() {
        x[i] = (x[i] - max_val).exp();
        exp_sum += x[i];
    }

    let inv_exp_sum = 1.0 / exp_sum;
    for i in 0..x.len() {
        x[i] *= inv_exp_sum;
    }
}

pub fn leaky_relu(x: &mut [f32], alpha: f32) {
    let zero = unsafe { _mm256_setzero_ps() };
    let alpha_wide = unsafe { _mm256_set1_ps(alpha) };

    let mut i = 0;
    while i + 8 <= x.len() {
        let x_wide = unsafe { _mm256_loadu_ps(&x[i]) };
        let mask = unsafe { _mm256_cmp_ps(x_wide, zero, _CMP_LT_OQ) };
        let neg_part = unsafe { _mm256_mul_ps(x_wide, alpha_wide) };
        let pos_part = x_wide;
        let result = unsafe { _mm256_blendv_ps(pos_part, neg_part, mask) };
        unsafe { _mm256_storeu_ps(&mut x[i], result) };
        i += 8;
    }

    for j in i..x.len() {
        x[j] = if x[j] < 0.0 { x[j] * alpha } else { x[j] };
    }
}