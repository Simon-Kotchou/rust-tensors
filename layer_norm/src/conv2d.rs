#![feature(stdsimd)]
use std::arch::x86_64::*;
use rayon::prelude::*;

pub fn conv2d_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    r: usize,
    s: usize,
    ph: usize,
    pw: usize,
    sh: usize,
    sw: usize,
) {
    let oh = (h + 2 * ph - r) / sh + 1;
    let ow = (w + 2 * pw - s) / sw + 1;
    let c_stride = h * w;
    let k_stride = oh * ow;
    let weight_stride = r * s * c;

    out.par_chunks_exact_mut(k * k_stride)
        .zip(inp.par_chunks_exact(c * c_stride))
        .for_each(|(out_n, inp_n)| {
            for (out_k, (&bias_k, weight_k)) in out_n.chunks_exact_mut(k_stride)
                .zip(bias.iter().zip(weight.chunks_exact(weight_stride)))
            {
                conv2d_kernel(out_k, inp_n, weight_k, bias_k, c, h, w, oh, ow, r, s, ph, pw, sh, sw);
            }
        });
}

#[inline(always)]
fn conv2d_kernel(
    out_k: &mut [f32],
    inp_n: &[f32],
    weight_k: &[f32],
    bias_k: f32,
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    r: usize,
    s: usize,
    ph: usize,
    pw: usize,
    sh: usize,
    sw: usize,
) {
    const SIMD_WIDTH: usize = 8;
    let c_simd = c - (c % SIMD_WIDTH);
    let bias_val = unsafe { _mm256_set1_ps(bias_k) };

    for y in 0..oh {
        for x in 0..ow {
            let mut sum = unsafe { _mm256_setzero_ps() };

            for c_idx in (0..c_simd).step_by(SIMD_WIDTH) {
                sum = process_conv2d_chunk(
                    inp_n, weight_k, c_idx, y, x, sh, sw, ph, pw, r, s, h, w, sum,
                );
            }

            for c_idx in c_simd..c {
                sum = process_conv2d_single(
                    inp_n, weight_k, c_idx, y, x, sh, sw, ph, pw, r, s, h, w, sum,
                );
            }

            let out_val = unsafe { _mm256_add_ps(sum, bias_val) };
            unsafe {
                _mm256_storeu_ps(&mut out_k[y * ow + x], out_val);
            }
        }
    }
}

#[inline(always)]
unsafe fn process_conv2d_chunk(
    inp_n: &[f32],
    weight_k: &[f32],
    c_idx: usize,
    y: usize,
    x: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    r: usize,
    s: usize,
    h: usize,
    w: usize,
    mut acc: __m256,
) -> __m256 {
    for fy_idx in 0..r {
        for fx_idx in 0..s {
            let iy = y.wrapping_mul(sh).wrapping_add(fy_idx).wrapping_sub(ph);
            let ix = x.wrapping_mul(sw).wrapping_add(fx_idx).wrapping_sub(pw);
            if iy < h && ix < w {
                let inp_offset = c_idx.wrapping_add(iy.wrapping_mul(w).wrapping_add(ix));
                let weight_offset = c_idx.wrapping_add(fy_idx.wrapping_mul(s).wrapping_add(fx_idx).wrapping_mul(8));
                let inp_val = _mm256_loadu_ps(inp_n.get_unchecked(inp_offset));
                let weight_val = _mm256_loadu_ps(weight_k.get_unchecked(weight_offset));
                acc = _mm256_fmadd_ps(inp_val, weight_val, acc);
            }
        }
    }
    acc
}

#[inline(always)]
unsafe fn process_conv2d_single(
    inp_n: &[f32],
    weight_k: &[f32],
    c_idx: usize,
    y: usize,
    x: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    r: usize,
    s: usize,
    h: usize,
    w: usize,
    mut acc: __m256,
) -> __m256 {
    for fy_idx in 0..r {
        for fx_idx in 0..s {
            let iy = y.wrapping_mul(sh).wrapping_add(fy_idx).wrapping_sub(ph);
            let ix = x.wrapping_mul(sw).wrapping_add(fx_idx).wrapping_sub(pw);
            if iy < h && ix < w {
                let inp_offset = c_idx.wrapping_add(iy.wrapping_mul(w).wrapping_add(ix));
                let weight_offset = c_idx.wrapping_mul(r.wrapping_mul(s)).wrapping_add(fy_idx.wrapping_mul(s).wrapping_add(fx_idx));
                let inp_val = *inp_n.get_unchecked(inp_offset);
                let weight_val = *weight_k.get_unchecked(weight_offset);
                acc = _mm256_add_ps(acc, _mm256_set1_ps(inp_val * weight_val));
            }
        }
    }
    acc
}