use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn matmul_avx2(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = _mm256_setzero_ps();

            let mut p = 0;
            while p + 8 <= k {
                let a_vec = _mm256_loadu_ps(&a[i * k + p]);
                let b_vec = _mm256_loadu_ps(&b[p * n + j]);
                sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
                p += 8;
            }

            let mut result = hsum256_ps_avx(sum);

            while p < k {
                result += a[i * k + p] * b[p * n + j];
                p += 1;
            }

            out[i * n + j] = result;
        }
    }
}

#[inline]
fn hsum256_ps_avx(x: __m256) -> f32 {
    unsafe {
        let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
        let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        _mm_cvtss_f32(x32)
    }
}