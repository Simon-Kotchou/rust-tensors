#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn layernorm_forward(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    b: usize,
    t: usize,
    c: usize,
) {
    let eps: f32 = 1e-5;
    for bt in 0..(b * t) {
        let offset: usize = bt * c;
        let x: &[f32] = &inp[offset..offset + c];
        let mut m: __m256 = unsafe { _mm256_setzero_ps() };
        let mut v: __m256 = unsafe { _mm256_setzero_ps() };
        for i in (0..c).step_by(8) {
            let xi: __m256 = unsafe { _mm256_loadu_ps(&x[i]) };
            m = unsafe { _mm256_add_ps(m, xi) };
            let xi_m: __m256 = unsafe { _mm256_sub_ps(xi, _mm256_set1_ps(hsum_ps_avx(m) / (c - 1) as f32)) };
            v = unsafe { _mm256_add_ps(v, _mm256_mul_ps(xi_m, xi_m)) };
        }
        let s: __m256 = unsafe { _mm256_set1_ps(1.0 / (hsum_ps_avx(v) / (c - 1) as f32 + eps).sqrt()) };
        let out_bt: &mut [f32] = &mut out[offset..offset + c];
        for i in (0..c).step_by(8) {
            let xi: __m256 = unsafe { _mm256_loadu_ps(&x[i]) };
            let wi: __m256 = unsafe { _mm256_loadu_ps(&weight[i]) };
            let bi: __m256 = unsafe { _mm256_loadu_ps(&bias[i]) };
            let o: __m256 = unsafe {
                _mm256_add_ps(
                    _mm256_mul_ps(
                        _mm256_mul_ps(s, _mm256_sub_ps(xi, _mm256_set1_ps(hsum_ps_avx(m) / (c - 1) as f32))),
                        wi,
                    ),
                    bi,
                )
            };
            unsafe { _mm256_storeu_ps(&mut out_bt[i], o) };
        }
        mean[bt] = hsum_ps_avx(m) / (c - 1) as f32;
        rstd[bt] = 1.0 / (hsum_ps_avx(v) / (c - 1) as f32 + eps).sqrt();
    }
}

#[inline]
fn hsum_ps_avx(v: __m256) -> f32 {
    unsafe {
        let mut sum: __m256 = _mm256_permute2f128_ps(v, v, 0x01);
        sum = _mm256_add_ps(sum, v);
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        _mm_cvtss_f32(_mm256_castps256_ps128(sum))
    }
}