#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn multihead_attention_forward(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    b: usize,
    t: usize,
    c: usize,
    num_heads: usize,
) {
    let head_dim = c / num_heads;
    let head_dim_simd = head_dim - (head_dim % 8);

    for bt in 0..(b * t) {
        let offset = bt * c;
        let q_bt = &q[offset..offset + c];
        let k_bt = &k[offset..offset + c];
        let v_bt = &v[offset..offset + c];
        let out_bt = &mut out[offset..offset + c];

        for h in 0..num_heads {
            let wq_h = &wq[h * c * head_dim..(h + 1) * c * head_dim];
            let wk_h = &wk[h * c * head_dim..(h + 1) * c * head_dim];
            let wv_h = &wv[h * c * head_dim..(h + 1) * c * head_dim];
            let wo_h = &wo[h * head_dim * c..(h + 1) * head_dim * c];

            let q_h = &mut [0.0; 1024];
            let k_h = &mut [0.0; 1024];
            let v_h = &mut [0.0; 1024];
            let out_h = &mut [0.0; 1024];

            linear_forward(q_h, q_bt, wq_h, &[0.0; 1024], 1, c, head_dim);
            linear_forward(k_h, k_bt, wk_h, &[0.0; 1024], 1, c, head_dim);
            linear_forward(v_h, v_bt, wv_h, &[0.0; 1024], 1, c, head_dim);

            let mut i = 0;
            while i < head_dim_simd {
                let q_wide = unsafe { _mm256_loadu_ps(&q_h[i]) };
                let k_wide = unsafe { _mm256_loadu_ps(&k_h[i]) };
                let mut sum = unsafe { _mm256_setzero_ps() };

                for j in 0..head_dim_simd {
                    let q_j = unsafe { _mm256_set1_ps(q_h[j]) };
                    let k_j = unsafe { _mm256_loadu_ps(&k_h[j]) };
                    let dot = unsafe { _mm256_dp_ps(q_j, k_j, 0xff) };
                    sum = unsafe { _mm256_add_ps(sum, dot) };
                }

                let v_wide = unsafe { _mm256_loadu_ps(&v_h[i]) };
                let out_wide = unsafe { _mm256_mul_ps(sum, v_wide) };
                unsafe { _mm256_storeu_ps(&mut out_h[i], out_wide) };

                i += 8;
            }

            for j in head_dim_simd..head_dim {
                let mut dot = 0.0;
                for k in 0..head_dim {
                    dot += q_h[k] * k_h[k];
                }
                out_h[j] = dot * v_h[j];
            }

            linear_forward(
                &mut out_bt[h * head_dim..(h + 1) * head_dim],
                out_h,
                wo_h,
                &[0.0; 1024],
                1,
                head_dim,
                head_dim,
            );
        }
    }
}