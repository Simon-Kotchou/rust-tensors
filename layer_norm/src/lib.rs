// #[cfg(target_arch = "x86_64")]
// use std::arch::x86_64::*;

// pub fn layernorm_forward(
//     out: &mut [f32],
//     mean: &mut [f32],
//     rstd: &mut [f32],
//     inp: &[f32],
//     weight: &[f32],
//     bias: &[f32],
//     b: usize,
//     t: usize,
//     c: usize,
// ) {
//     let eps: f32 = 1e-5;
//     let c_simd = c - (c % 8);

//     for bt in 0..(b * t) {
//         let offset: usize = bt * c;
//         let x: &[f32] = &inp[offset..offset + c];
//         let mut m: __m256 = unsafe { _mm256_setzero_ps() };
//         let mut v: __m256 = unsafe { _mm256_setzero_ps() };

//         let mut i = 0;
//         while i + 24 <= c {
//             let xi0: __m256 = unsafe { _mm256_loadu_ps(&x[i]) };
//             let xi1: __m256 = unsafe { _mm256_loadu_ps(&x[i + 8]) };
//             let xi2: __m256 = unsafe { _mm256_loadu_ps(&x[i + 16]) };
//             m = unsafe { _mm256_add_ps(m, xi0) };
//             m = unsafe { _mm256_add_ps(m, xi1) };
//             m = unsafe { _mm256_add_ps(m, xi2) };
//             i += 24;
//         }
//         while i < c_simd {
//             let xi: __m256 = unsafe { _mm256_loadu_ps(&x[i]) };
//             m = unsafe { _mm256_add_ps(m, xi) };
//             i += 8;
//         }

//         let mut mean_val = hsum_ps_avx(m);
//         for j in c_simd..c {
//             mean_val += x[j];
//         }
//         mean_val /= c as f32;
//         mean[bt] = mean_val;

//         let mean_val_simd = unsafe { _mm256_set1_ps(mean_val) };
//         i = 0;
//         while i + 24 <= c {
//             let xi0: __m256 = unsafe { _mm256_loadu_ps(&x[i]) };
//             let xi1: __m256 = unsafe { _mm256_loadu_ps(&x[i + 8]) };
//             let xi2: __m256 = unsafe { _mm256_loadu_ps(&x[i + 16]) };
//             let xi0_m: __m256 = unsafe { _mm256_sub_ps(xi0, mean_val_simd) };
//             let xi1_m: __m256 = unsafe { _mm256_sub_ps(xi1, mean_val_simd) };
//             let xi2_m: __m256 = unsafe { _mm256_sub_ps(xi2, mean_val_simd) };
//             v = unsafe { _mm256_fmadd_ps(xi0_m, xi0_m, v) };
//             v = unsafe { _mm256_fmadd_ps(xi1_m, xi1_m, v) };
//             v = unsafe { _mm256_fmadd_ps(xi2_m, xi2_m, v) };
//             i += 24;
//         }
//         while i < c_simd {
//             let xi: __m256 = unsafe { _mm256_loadu_ps(&x[i]) };
//             let xi_m: __m256 = unsafe { _mm256_sub_ps(xi, mean_val_simd) };
//             v = unsafe { _mm256_fmadd_ps(xi_m, xi_m, v) };
//             i += 8;
//         }

//         let mut var_val = hsum_ps_avx(v);
//         for j in c_simd..c {
//             let xi_m = x[j] - mean_val;
//             var_val += xi_m * xi_m;
//         }
//         var_val /= c as f32;

//         let rstd_val = (var_val + eps).sqrt().recip();
//         rstd[bt] = rstd_val;

//         let s: __m256 = unsafe { _mm256_set1_ps(rstd_val) };
//         let out_bt: &mut [f32] = &mut out[offset..offset + c];

//         i = 0;
//         while i + 24 <= c {
//             let xi0: __m256 = unsafe { _mm256_loadu_ps(&x[i]) };
//             let xi1: __m256 = unsafe { _mm256_loadu_ps(&x[i + 8]) };
//             let xi2: __m256 = unsafe { _mm256_loadu_ps(&x[i + 16]) };
//             let wi0: __m256 = unsafe { _mm256_loadu_ps(&weight[i]) };
//             let wi1: __m256 = unsafe { _mm256_loadu_ps(&weight[i + 8]) };
//             let wi2: __m256 = unsafe { _mm256_loadu_ps(&weight[i + 16]) };
//             let bi0: __m256 = unsafe { _mm256_loadu_ps(&bias[i]) };
//             let bi1: __m256 = unsafe { _mm256_loadu_ps(&bias[i + 8]) };
//             let bi2: __m256 = unsafe { _mm256_loadu_ps(&bias[i + 16]) };
//             let o0: __m256 = unsafe {
//                 _mm256_fmadd_ps(
//                     _mm256_mul_ps(s, _mm256_sub_ps(xi0, mean_val_simd)),
//                     wi0,
//                     bi0,
//                 )
//             };
//             let o1: __m256 = unsafe {
//                 _mm256_fmadd_ps(
//                     _mm256_mul_ps(s, _mm256_sub_ps(xi1, mean_val_simd)),
//                     wi1,
//                     bi1,
//                 )
//             };
//             let o2: __m256 = unsafe {
//                 _mm256_fmadd_ps(
//                     _mm256_mul_ps(s, _mm256_sub_ps(xi2, mean_val_simd)),
//                     wi2,
//                     bi2,
//                 )
//             };
//             unsafe { _mm256_storeu_ps(&mut out_bt[i], o0) };
//             unsafe { _mm256_storeu_ps(&mut out_bt[i + 8], o1) };
//             unsafe { _mm256_storeu_ps(&mut out_bt[i + 16], o2) };
//             i += 24;
//         }
//         while i < c_simd {
//             let xi: __m256 = unsafe { _mm256_loadu_ps(&x[i]) };
//             let wi: __m256 = unsafe { _mm256_loadu_ps(&weight[i]) };
//             let bi: __m256 = unsafe { _mm256_loadu_ps(&bias[i]) };
//             let o: __m256 = unsafe {
//                 _mm256_fmadd_ps(
//                     _mm256_mul_ps(s, _mm256_sub_ps(xi, mean_val_simd)),
//                     wi,
//                     bi,
//                 )
//             };
//             unsafe { _mm256_storeu_ps(&mut out_bt[i], o) };
//             i += 8;
//         }

//         let mut j = c_simd;
//         while j + 8 <= c {
//             let xj = unsafe { _mm256_set_ps(
//                 x[j + 7], x[j + 6], x[j + 5], x[j + 4],
//                 x[j + 3], x[j + 2], x[j + 1], x[j],
//             ) };
//             let wj = unsafe { _mm256_loadu_ps(&weight[j]) };
//             let bj = unsafe { _mm256_loadu_ps(&bias[j]) };
//             let out_j = unsafe {
//                 _mm256_fmadd_ps(
//                     _mm256_mul_ps(s, _mm256_sub_ps(xj, mean_val_simd)),
//                     wj,
//                     bj,
//                 )
//             };
//             unsafe { _mm256_storeu_ps(&mut out_bt[j], out_j) };
//             j += 8;
//         }
//         for j in j..c {
//             out_bt[j] = (x[j] - mean_val) * rstd_val * weight[j] + bias[j];
//         }
//     }
// }
pub mod aligned_f32;
pub use aligned_f32::AlignedF32;

pub mod layernorm;
pub use layernorm::layernorm_forward;

// pub fn layernorm_backward(
//     inp_grad: &mut [f32],
//     weight_grad: &mut [f32],
//     bias_grad: &mut [f32],
//     out_grad: &[f32],
//     inp: &[f32],
//     mean: &[f32],
//     rstd: &[f32],
//     weight: &[f32],
//     b: usize,
//     t: usize,
//     c: usize,
// ) {
//     let c_simd = c - (c % 8);

//     for bt in 0..(b * t) {
//         let offset = bt * c;
//         let inp_bt = &inp[offset..offset + c];
//         let out_grad_bt = &out_grad[offset..offset + c];
//         let inp_grad_bt = &mut inp_grad[offset..offset + c];

//         let mean_val = mean[bt];
//         let rstd_val = rstd[bt];

//         let mut sum_grad: __m256 = unsafe { _mm256_setzero_ps() };
//         let mut sum_inp_grad: __m256 = unsafe { _mm256_setzero_ps() };

//         let mut i = 0;
//         while i < c_simd {
//             let out_grad_i: __m256 = unsafe { _mm256_loadu_ps(&out_grad_bt[i]) };
//             let inp_i: __m256 = unsafe { _mm256_loadu_ps(&inp_bt[i]) };
//             let weight_i: __m256 = unsafe { _mm256_loadu_ps(&weight[i]) };

//             sum_grad = unsafe { _mm256_add_ps(sum_grad, out_grad_i) };
//             sum_inp_grad = unsafe {
//                 _mm256_fmadd_ps(
//                     out_grad_i,
//                     _mm256_sub_ps(inp_i, _mm256_set1_ps(mean_val)),
//                     sum_inp_grad,
//                 )
//             };

//             let grad_i: __m256 = unsafe {
//                 _mm256_mul_ps(out_grad_i, _mm256_set1_ps(rstd_val * rstd_val))
//             };
//             let inp_grad_i: __m256 = unsafe {
//                 _mm256_fmadd_ps(grad_i, weight_i, _mm256_set1_ps(0.0))
//             };
//             unsafe { _mm256_storeu_ps(&mut inp_grad_bt[i], inp_grad_i) };

//             let weight_grad_i: __m256 = unsafe {
//                 _mm256_mul_ps(grad_i, _mm256_sub_ps(inp_i, _mm256_set1_ps(mean_val)))
//             };
//             unsafe { _mm256_storeu_ps(&mut weight_grad[i], weight_grad_i) };

//             unsafe { _mm256_storeu_ps(&mut bias_grad[i], out_grad_i) };

//             i += 8;
//         }

//         let sum_grad_scalar = hsum_ps_avx(sum_grad);
//         let sum_inp_grad_scalar = hsum_ps_avx(sum_inp_grad);

//         for j in c_simd..c {
//             let out_grad_j = out_grad_bt[j];
//             let inp_j = inp_bt[j];

//             sum_grad_scalar += out_grad_j;
//             sum_inp_grad_scalar += out_grad_j * (inp_j - mean_val);

//             let grad_j = out_grad_j * rstd_val * rstd_val;
//             inp_grad_bt[j] = grad_j * weight[j];
//             weight_grad[j] = grad_j * (inp_j - mean_val);
//             bias_grad[j] = out_grad_j;
//         }

//         let factor = rstd_val * rstd_val / c as f32;
//         let mean_grad = -sum_grad_scalar * rstd_val / c as f32;

//         i = 0;
//         while i < c_simd {
//             let inp_grad_i: __m256 = unsafe { _mm256_loadu_ps(&inp_grad_bt[i]) };
//             let inp_i: __m256 = unsafe { _mm256_loadu_ps(&inp_bt[i]) };

//             let term1: __m256 = unsafe { _mm256_set1_ps(factor * sum_inp_grad_scalar) };
//             let term2: __m256 = unsafe {
//                 _mm256_mul_ps(
//                     _mm256_set1_ps(factor),
//                     _mm256_sub_ps(inp_i, _mm256_set1_ps(mean_val)),
//                 )
//             };
//             let term3: __m256 = unsafe { _mm256_set1_ps(mean_grad) };

//             let inp_grad_i: __m256 = unsafe {
//                 _mm256_add_ps(inp_grad_i, _mm256_add_ps(term1, _mm256_add_ps(term2, term3)))
//             };
//             unsafe { _mm256_storeu_ps(&mut inp_grad_bt[i], inp_grad_i) };

//             i += 8;
//         }

//         for j in c_simd..c {
//             inp_grad_bt[j] += factor * sum_inp_grad_scalar
//                 + factor * (inp_bt[j] - mean_val)
//                 + mean_grad;
//         }
//     }
// }

// #[inline]
// fn hsum_ps_avx(v: __m256) -> f32 {
//     unsafe {
//         let mut sum: __m256 = _mm256_hadd_ps(v, v);
//         sum = _mm256_hadd_ps(sum, sum);
//         _mm_cvtss_f32(_mm_add_ss(
//             _mm256_castps256_ps128(sum),
//             _mm256_extractf128_ps(sum, 1),
//         ))
//     }
// }