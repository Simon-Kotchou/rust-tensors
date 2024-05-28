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

    let mut q_h = vec![0.0; b * t * c];
    let mut k_h = vec![0.0; b * t * c];
    let mut v_h = vec![0.0; b * t * c];
    let mut out_h = vec![0.0; b * t * c];

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

            let q_h_offset = bt * c + h * head_dim;
            let k_h_offset = bt * c + h * head_dim;
            let v_h_offset = bt * c + h * head_dim;
            let out_h_offset = bt * c + h * head_dim;

            linear_forward(
                &mut q_h[q_h_offset..q_h_offset + head_dim],
                q_bt,
                wq_h,
                &[0.0; 1024],
                1,
                c,
                head_dim,
            );
            linear_forward(
                &mut k_h[k_h_offset..k_h_offset + head_dim],
                k_bt,
                wk_h,
                &[0.0; 1024],
                1,
                c,
                head_dim,
            );
            linear_forward(
                &mut v_h[v_h_offset..v_h_offset + head_dim],
                v_bt,
                wv_h,
                &[0.0; 1024],
                1,
                c,
                head_dim,
            );

            let mut attn_scores = vec![0.0; head_dim];
            for i in (0..head_dim_simd).step_by(8) {
                let q_wide = unsafe { _mm256_loadu_ps(&q_h[q_h_offset + i]) };
                let mut sum = unsafe { _mm256_setzero_ps() };
                for j in (0..head_dim_simd).step_by(8) {
                    let k_wide = unsafe { _mm256_loadu_ps(&k_h[k_h_offset + j]) };
                    let dot = unsafe { _mm256_dp_ps(q_wide, k_wide, 0xff) };
                    sum = unsafe { _mm256_add_ps(sum, dot) };
                }
                unsafe { _mm256_storeu_ps(&mut attn_scores[i], sum) };
            }
            for i in head_dim_simd..head_dim {
                let mut dot = 0.0;
                for j in 0..head_dim {
                    dot += q_h[q_h_offset + i] * k_h[k_h_offset + j];
                }
                attn_scores[i] = dot;
            }

            // Softmax
            let max_score = attn_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_scores = attn_scores.iter().map(|&s| (s - max_score).exp()).collect::<Vec<_>>();
            let sum_exp_scores = exp_scores.iter().sum::<f32>();
            exp_scores.iter_mut().for_each(|s| *s /= sum_exp_scores);

            // Matrix multiplication
            for i in 0..head_dim {
                let mut sum = 0.0;
                for j in 0..head_dim {
                    sum += exp_scores[j] * v_h[v_h_offset + j];
                }
                out_h[out_h_offset + i] = sum;
            }
            linear_forward(
                &mut out_bt[h * head_dim..(h + 1) * head_dim],
                &out_h[out_h_offset..out_h_offset + head_dim],
                wo_h,
                &[0.0; 1024],
                1,
                head_dim,
                head_dim,
            );
        }
    }
}