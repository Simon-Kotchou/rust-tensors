#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use rand::Rng;

pub fn linear_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let n_stride = 16;
    let n_simd = n - (n % n_stride);
    inp.chunks_exact(k)
        .zip(out.chunks_exact_mut(n))
        .for_each(|(inp_m, out_m)| {
            for j in (0..n_simd).step_by(n_stride) {
                let mut sum0 = unsafe { _mm512_setzero_ps() };
                let mut sum1 = unsafe { _mm512_setzero_ps() };
                for i in (0..k).step_by(8) {
                    let inp_wide0 = unsafe { _mm512_set1_ps(inp_m[i]) };
                    let inp_wide1 = unsafe { _mm512_set1_ps(inp_m[i + 1]) };
                    let weight_wide0 = unsafe { _mm512_loadu_ps(&weight[j + i * n]) };
                    let weight_wide1 = unsafe { _mm512_loadu_ps(&weight[j + (i + 1) * n]) };
                    sum0 = unsafe { _mm512_fmadd_ps(inp_wide0, weight_wide0, sum0) };
                    sum1 = unsafe { _mm512_fmadd_ps(inp_wide1, weight_wide1, sum1) };
                    let inp_wide2 = unsafe { _mm512_set1_ps(inp_m[i + 2]) };
                    let inp_wide3 = unsafe { _mm512_set1_ps(inp_m[i + 3]) };
                    let weight_wide2 = unsafe { _mm512_loadu_ps(&weight[j + (i + 2) * n]) };
                    let weight_wide3 = unsafe { _mm512_loadu_ps(&weight[j + (i + 3) * n]) };
                    sum0 = unsafe { _mm512_fmadd_ps(inp_wide2, weight_wide2, sum0) };
                    sum1 = unsafe { _mm512_fmadd_ps(inp_wide3, weight_wide3, sum1) };
                    let inp_wide4 = unsafe { _mm512_set1_ps(inp_m[i + 4]) };
                    let inp_wide5 = unsafe { _mm512_set1_ps(inp_m[i + 5]) };
                    let weight_wide4 = unsafe { _mm512_loadu_ps(&weight[j + (i + 4) * n]) };
                    let weight_wide5 = unsafe { _mm512_loadu_ps(&weight[j + (i + 5) * n]) };
                    sum0 = unsafe { _mm512_fmadd_ps(inp_wide4, weight_wide4, sum0) };
                    sum1 = unsafe { _mm512_fmadd_ps(inp_wide5, weight_wide5, sum1) };
                    let inp_wide6 = unsafe { _mm512_set1_ps(inp_m[i + 6]) };
                    let inp_wide7 = unsafe { _mm512_set1_ps(inp_m[i + 7]) };
                    let weight_wide6 = unsafe { _mm512_loadu_ps(&weight[j + (i + 6) * n]) };
                    let weight_wide7 = unsafe { _mm512_loadu_ps(&weight[j + (i + 7) * n]) };
                    sum0 = unsafe { _mm512_fmadd_ps(inp_wide6, weight_wide6, sum0) };
                    sum1 = unsafe { _mm512_fmadd_ps(inp_wide7, weight_wide7, sum1) };
                }
                let bias_wide = unsafe { _mm512_loadu_ps(&bias[j]) };
                let out_wide0 = unsafe { _mm512_add_ps(sum0, bias_wide) };
                let out_wide1 = unsafe { _mm512_add_ps(sum1, bias_wide) };
                unsafe {
                    _mm512_storeu_ps(&mut out_m[j], out_wide0);
                    _mm512_storeu_ps(&mut out_m[j + 8], out_wide1);
                }
            }
            for j in n_simd..n {
                let mut sum = 0.0;
                for i in 0..k {
                    sum += inp_m[i] * weight[i * n + j];
                }
                out_m[j] = sum + bias[j];
            }
        });
}

pub fn dropout(x: &mut [f32], prob: f32, train: bool) {
    if !train {
        return;
    }
    let scale = 1.0 / (1.0 - prob);
    let mut rng = rand::thread_rng();
    let n_stride = 16;
    let n_simd = x.len() - (x.len() % n_stride);
    for i in (0..n_simd).step_by(n_stride) {
        let mut mask = unsafe { _mm512_setzero_ps() };
        for j in 0..n_stride {
            if rng.gen::<f32>() < prob {
                mask = unsafe { _mm512_mask_blend_ps(1 << j, mask, _mm512_set1_ps(0.0)) };
            } else {
                mask = unsafe { _mm512_mask_blend_ps(1 << j, mask, _mm512_set1_ps(scale)) };
            }
        }
        let x_wide = unsafe { _mm512_loadu_ps(&x[i]) };
        let result_wide = unsafe { _mm512_mul_ps(x_wide, mask) };
        unsafe {
            _mm512_storeu_ps(&mut x[i], result_wide);
        }
    }
    for i in n_simd..x.len() {
        if rng.gen::<f32>() < prob {
            x[i] = 0.0;
        } else {
            x[i] *= scale;
        }
    }
}