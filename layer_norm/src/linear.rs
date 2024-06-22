#![feature(stdsimd)]
use std::arch::x86_64::*;
use rand::prelude::*;
use rayon::prelude::*;

pub fn linear_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) {
    assert_eq!(inp.len(), m * k);
    assert_eq!(weight.len(), k * n);
    assert_eq!(bias.len(), n);
    assert_eq!(out.len(), m * n);

    const SIMD_WIDTH: usize = 8;
    let n_simd = n - (n % SIMD_WIDTH);

    out.par_chunks_exact_mut(n)
        .zip(inp.par_chunks_exact(k))
        .for_each(|(out_m, inp_m)| {
            for j in (0..n_simd).step_by(SIMD_WIDTH) {
                let mut sum = unsafe { _mm256_setzero_ps() };
                for i in (0..k).step_by(4) {
                    unsafe {
                        let inp_wide = _mm256_set1_ps(inp_m[i]);
                        let weight_wide = _mm256_loadu_ps(weight.as_ptr().add(j + i * n));
                        sum = _mm256_fmadd_ps(inp_wide, weight_wide, sum);

                        let inp_wide = _mm256_set1_ps(inp_m[i + 1]);
                        let weight_wide = _mm256_loadu_ps(weight.as_ptr().add(j + (i + 1) * n));
                        sum = _mm256_fmadd_ps(inp_wide, weight_wide, sum);

                        let inp_wide = _mm256_set1_ps(inp_m[i + 2]);
                        let weight_wide = _mm256_loadu_ps(weight.as_ptr().add(j + (i + 2) * n));
                        sum = _mm256_fmadd_ps(inp_wide, weight_wide, sum);

                        let inp_wide = _mm256_set1_ps(inp_m[i + 3]);
                        let weight_wide = _mm256_loadu_ps(weight.as_ptr().add(j + (i + 3) * n));
                        sum = _mm256_fmadd_ps(inp_wide, weight_wide, sum);
                    }
                }
                unsafe {
                    let bias_wide = _mm256_loadu_ps(bias.as_ptr().add(j));
                    let out_wide = _mm256_add_ps(sum, bias_wide);
                    _mm256_storeu_ps(out_m.as_mut_ptr().add(j), out_wide);
                }
            }

            // Handle remaining elements
            for j in n_simd..n {
                out_m[j] = inp_m.iter()
                    .zip(weight[j..].iter().step_by(n))
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>() + bias[j];
            }
        });
}

pub fn dropout(x: &mut [f32], prob: f32, train: bool) {
    if !train {
        return;
    }

    let scale = 1.0 / (1.0 - prob);
    let mut rng = thread_rng();

    x.par_iter_mut().for_each(|val| {
        if rng.gen::<f32>() >= prob {
            *val *= scale;
        } else {
            *val = 0.0;
        }
    });
}
