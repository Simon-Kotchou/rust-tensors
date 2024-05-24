#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

                inp_m
                    .iter()
                    .zip(weight.chunks_exact(n).skip(j))
                    .for_each(|(&inp_val, weight_k)| {
                        let inp_wide = unsafe { _mm256_set1_ps(inp_val) };
                        let weight_wide = unsafe { _mm256_loadu_ps(weight_k) };
                        sum = unsafe { _mm256_fmadd_ps(inp_wide, weight_wide, sum) };
                    });

                let bias_wide = unsafe { _mm256_loadu_ps(&bias[j]) };
                let out_wide = unsafe { _mm256_add_ps(sum, bias_wide) };
                unsafe {
                    _mm256_storeu_ps(&mut out_m[j], out_wide);
                }
            }

            for j in n_simd..n {
                let mut sum = 0.0;
                inp_m
                    .iter()
                    .zip(weight.chunks_exact(n).skip(j))
                    .for_each(|(&inp_val, weight_k)| {
                        sum += inp_val * weight_k[0];
                    });
                out_m[j] = sum + bias[j];
            }
        });
}