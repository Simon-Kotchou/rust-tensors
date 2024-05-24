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
    let n_simd = n - (n % 8);

    for i in 0..m {
        let inp_offset = i * k;
        let out_offset = i * n;

        let mut j = 0;
        while j < n_simd {
            let mut sum: __m256 = unsafe { _mm256_setzero_ps() };

            let mut l = 0;
            while l < k {
                let inp_val = unsafe { _mm256_set1_ps(inp[inp_offset + l]) };
                let weight_val = unsafe { _mm256_loadu_ps(&weight[l * n + j]) };
                sum = unsafe { _mm256_fmadd_ps(inp_val, weight_val, sum) };
                l += 1;
            }

            let bias_val = unsafe { _mm256_loadu_ps(&bias[j]) };
            let out_val = unsafe { _mm256_add_ps(sum, bias_val) };
            unsafe { _mm256_storeu_ps(&mut out[out_offset + j], out_val) };

            j += 8;
        }

        for j in n_simd..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += inp[inp_offset + l] * weight[l * n + j];
            }
            out[out_offset + j] = sum + bias[j];
        }
    }
}

pub fn linear_backward(
    inp_grad: &mut [f32],
    weight_grad: &mut [f32],
    bias_grad: &mut [f32],
    out_grad: &[f32],
    inp: &[f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let n_simd = n - (n % 8);

    for i in 0..m {
        let inp_offset = i * k;
        let out_offset = i * n;

        let mut j = 0;
        while j < n_simd {
            let out_grad_val = unsafe { _mm256_loadu_ps(&out_grad[out_offset + j]) };

            let mut l = 0;
            while l < k {
                let inp_val = unsafe { _mm256_set1_ps(inp[inp_offset + l]) };
                let weight_grad_val = unsafe { _mm256_loadu_ps(&weight_grad[l * n + j]) };
                let weight_grad_val = unsafe { _mm256_fmadd_ps(inp_val, out_grad_val, weight_grad_val) };
                unsafe { _mm256_storeu_ps(&mut weight_grad[l * n + j], weight_grad_val) };

                let inp_grad_val = unsafe { _mm256_mul_ps(out_grad_val, _mm256_loadu_ps(&weight[l * n + j])) };
                inp_grad[inp_offset + l] += hsum_ps_avx(inp_grad_val);

                l += 1;
            }

            let bias_grad_val = unsafe { _mm256_loadu_ps(&bias_grad[j]) };
            let bias_grad_val = unsafe { _mm256_add_ps(bias_grad_val, out_grad_val) };
            unsafe { _mm256_storeu_ps(&mut bias_grad[j], bias_grad_val) };

            j += 8;
        }

        for j in n_simd..n {
            for l in 0..k {
                weight_grad[l * n + j] += inp[inp_offset + l] * out_grad[out_offset + j];
                inp_grad[inp_offset + l] += out_grad[out_offset + j] * weight[l * n + j];
            }
            bias_grad[j] += out_grad[out_offset + j];
        }
    }
}

#[inline]
fn hsum_ps_avx(v: __m256) -> f32 {
    unsafe {
        let mut sum: __m256 = _mm256_hadd_ps(v, v);
        sum = _mm256_hadd_ps(sum, sum);
        _mm_cvtss_f32(_mm_add_ss(
            _mm256_castps256_ps128(sum),
            _mm256_extractf128_ps(sum, 1),
        ))
    }
}