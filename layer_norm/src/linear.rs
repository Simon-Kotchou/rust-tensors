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
    let n_stride = 8;
    let n_simd = n - (n % n_stride);
    inp.chunks_exact(k)
        .zip(out.chunks_exact_mut(n))
        .for_each(|(inp_m, out_m)| {
            for j in (0..n_simd).step_by(n_stride) {
                let mut sum = unsafe { _mm256_setzero_ps() };
                for i in (0..k).step_by(4) {
                    let inp_wide = unsafe { _mm256_set1_ps(inp_m[i]) };
                    let weight_wide = unsafe { _mm256_loadu_ps(&weight[j + i * n]) };
                    sum = unsafe { _mm256_fmadd_ps(inp_wide, weight_wide, sum) };

                    let inp_wide = unsafe { _mm256_set1_ps(inp_m[i + 1]) };
                    let weight_wide = unsafe { _mm256_loadu_ps(&weight[j + (i + 1) * n]) };
                    sum = unsafe { _mm256_fmadd_ps(inp_wide, weight_wide, sum) };

                    let inp_wide = unsafe { _mm256_set1_ps(inp_m[i + 2]) };
                    let weight_wide = unsafe { _mm256_loadu_ps(&weight[j + (i + 2) * n]) };
                    sum = unsafe { _mm256_fmadd_ps(inp_wide, weight_wide, sum) };

                    let inp_wide = unsafe { _mm256_set1_ps(inp_m[i + 3]) };
                    let weight_wide = unsafe { _mm256_loadu_ps(&weight[j + (i + 3) * n]) };
                    sum = unsafe { _mm256_fmadd_ps(inp_wide, weight_wide, sum) };
                }
                let bias_wide = unsafe { _mm256_loadu_ps(&bias[j]) };
                let out_wide = unsafe { _mm256_add_ps(sum, bias_wide) };
                unsafe {
                    _mm256_storeu_ps(&mut out_m[j], out_wide);
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

    for i in 0..x.len() {
        if rng.gen::<f32>() < prob {
            x[i] = 0.0;
        } else {
            x[i] *= scale;
        }
    }
}