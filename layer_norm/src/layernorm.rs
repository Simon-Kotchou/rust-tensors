use std::arch::x86_64::*;

use crate::AlignedF32;

pub fn layernorm_forward(
    out: &mut AlignedF32,
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &AlignedF32,
    weight: &AlignedF32,
    bias: &AlignedF32,
    b: usize,
    t: usize,
    c: usize,
) {
    println!("Starting layernorm_forward");
    let eps: f32 = 1e-5;
    let c_simd = c - (c % 8);

    for bt in 0..(b * t) {
        println!("Processing batch {}", bt);
        let offset: usize = bt * c;
        let x: &[f32] = &inp.as_slice()[offset..offset + c];
        let mut m: __m256 = unsafe { _mm256_setzero_ps() };
        let mut v: __m256 = unsafe { _mm256_setzero_ps() };

        // Calculate mean
        let mut i = 0;
        while i + 24 <= c {
            let xi0: __m256 = unsafe { _mm256_load_ps(&x[i]) };
            let xi1: __m256 = unsafe { _mm256_load_ps(&x[i + 8]) };
            let xi2: __m256 = unsafe { _mm256_load_ps(&x[i + 16]) };
            m = unsafe { _mm256_add_ps(m, xi0) };
            m = unsafe { _mm256_add_ps(m, xi1) };
            m = unsafe { _mm256_add_ps(m, xi2) };
            i += 24;
        }
        while i < c_simd {
            let xi: __m256 = unsafe { _mm256_load_ps(&x[i]) };
            m = unsafe { _mm256_add_ps(m, xi) };
            i += 8;
        }

        let mut mean_val = hsum_ps_avx(m);
        for j in c_simd..c {
            mean_val += x[j];
        }
        mean_val /= c as f32;
        mean[bt] = mean_val;

        // Calculate variance
        let mean_val_simd = unsafe { _mm256_set1_ps(mean_val) };
        i = 0;
        while i + 24 <= c {
            let xi0: __m256 = unsafe { _mm256_load_ps(&x[i]) };
            let xi1: __m256 = unsafe { _mm256_load_ps(&x[i + 8]) };
            let xi2: __m256 = unsafe { _mm256_load_ps(&x[i + 16]) };
            let xi0_m: __m256 = unsafe { _mm256_sub_ps(xi0, mean_val_simd) };
            let xi1_m: __m256 = unsafe { _mm256_sub_ps(xi1, mean_val_simd) };
            let xi2_m: __m256 = unsafe { _mm256_sub_ps(xi2, mean_val_simd) };
            v = unsafe { _mm256_fmadd_ps(xi0_m, xi0_m, v) };
            v = unsafe { _mm256_fmadd_ps(xi1_m, xi1_m, v) };
            v = unsafe { _mm256_fmadd_ps(xi2_m, xi2_m, v) };
            i += 24;
        }
        while i < c_simd {
            let xi: __m256 = unsafe { _mm256_load_ps(&x[i]) };
            let xi_m: __m256 = unsafe { _mm256_sub_ps(xi, mean_val_simd) };
            v = unsafe { _mm256_fmadd_ps(xi_m, xi_m, v) };
            i += 8;
        }

        let mut var_val = hsum_ps_avx(v);
        for j in c_simd..c {
            let xi_m = x[j] - mean_val;
            var_val += xi_m * xi_m;
        }
        var_val /= c as f32;

        let rstd_val = (var_val + eps).sqrt().recip();
        rstd[bt] = rstd_val;

        // Normalize and apply weight and bias
        let s: __m256 = unsafe { _mm256_set1_ps(rstd_val) };
        let out_bt: &mut [f32] = &mut out.as_mut_slice()[offset..offset + c];

        i = 0;
        while i + 24 <= c {
            let xi0: __m256 = unsafe { _mm256_load_ps(&x[i]) };
            let xi1: __m256 = unsafe { _mm256_load_ps(&x[i + 8]) };
            let xi2: __m256 = unsafe { _mm256_load_ps(&x[i + 16]) };
            let wi0: __m256 = unsafe { _mm256_load_ps(&weight.as_slice()[i]) };
            let wi1: __m256 = unsafe { _mm256_load_ps(&weight.as_slice()[i + 8]) };
            let wi2: __m256 = unsafe { _mm256_load_ps(&weight.as_slice()[i + 16]) };
            let bi0: __m256 = unsafe { _mm256_load_ps(&bias.as_slice()[i]) };
            let bi1: __m256 = unsafe { _mm256_load_ps(&bias.as_slice()[i + 8]) };
            let bi2: __m256 = unsafe { _mm256_load_ps(&bias.as_slice()[i + 16]) };
            let o0: __m256 = unsafe {
                _mm256_fmadd_ps(
                    _mm256_mul_ps(s, _mm256_sub_ps(xi0, mean_val_simd)),
                    wi0,
                    bi0,
                )
            };
            let o1: __m256 = unsafe {
                _mm256_fmadd_ps(
                    _mm256_mul_ps(s, _mm256_sub_ps(xi1, mean_val_simd)),
                    wi1,
                    bi1,
                )
            };
            let o2: __m256 = unsafe {
                _mm256_fmadd_ps(
                    _mm256_mul_ps(s, _mm256_sub_ps(xi2, mean_val_simd)),
                    wi2,
                    bi2,
                )
            };
            unsafe { _mm256_store_ps(&mut out_bt[i], o0) };
            unsafe { _mm256_store_ps(&mut out_bt[i + 8], o1) };
            unsafe { _mm256_store_ps(&mut out_bt[i + 16], o2) };
            i += 24;
        }
        while i < c_simd {
            let xi: __m256 = unsafe { _mm256_load_ps(&x[i]) };
            let wi: __m256 = unsafe { _mm256_load_ps(&weight.as_slice()[i]) };
            let bi: __m256 = unsafe { _mm256_load_ps(&bias.as_slice()[i]) };
            let o: __m256 = unsafe {
                _mm256_fmadd_ps(
                    _mm256_mul_ps(s, _mm256_sub_ps(xi, mean_val_simd)),
                    wi,
                    bi,
                )
            };
            unsafe { _mm256_store_ps(&mut out_bt[i], o) };
            i += 8;
        }

        for j in c_simd..c {
            out_bt[j] = (x[j] - mean_val) * rstd_val * weight.as_slice()[j] + bias.as_slice()[j];
        }
    }
    println!("Completed layernorm_forward");
}

#[inline]
fn hsum_ps_avx(v: __m256) -> f32 {
    unsafe {
        let sum: __m256 = _mm256_hadd_ps(v, v);
        let sum: __m256 = _mm256_hadd_ps(sum, sum);
        _mm_cvtss_f32(_mm_add_ss(
            _mm256_castps256_ps128(sum),
            _mm256_extractf128_ps(sum, 1),
        ))
    }
}