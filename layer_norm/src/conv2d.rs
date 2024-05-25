#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
    let weight_stride = r * s;
    let c_simd = c - (c % 8);

    inp.chunks_exact(c * c_stride)
        .zip(out.chunks_exact_mut(k * k_stride))
        .for_each(|(inp_n, out_n)| {
            weight
                .chunks_exact(c * weight_stride)
                .zip(bias.iter())
                .zip(out_n.chunks_exact_mut(k_stride))
                .for_each(|((weight_k, &bias_k), out_k)| {
                    (0..oh).zip(0..ow).for_each(|(y, x)| {
                        let sum = (0..c)
                            .fold(unsafe { _mm256_setzero_ps() }, |acc, c_idx| {
                                let acc = if c_idx < c_simd {
                                    process_conv2d_chunk(
                                        inp_n,
                                        weight_k,
                                        c_idx,
                                        weight_stride,
                                        y,
                                        x,
                                        sh,
                                        sw,
                                        ph,
                                        pw,
                                        r,
                                        s,
                                        h,
                                        w,
                                        acc,
                                    )
                                } else {
                                    process_conv2d_single(
                                        inp_n,
                                        weight_k,
                                        c_idx,
                                        weight_stride,
                                        y,
                                        x,
                                        sh,
                                        sw,
                                        ph,
                                        pw,
                                        r,
                                        s,
                                        h,
                                        w,
                                        acc,
                                    )
                                };
                                acc
                            });

                        let bias_val = unsafe { _mm256_set1_ps(bias_k) };
                        let out_val = unsafe { _mm256_add_ps(sum, bias_val) };
                        unsafe {
                            _mm256_storeu_ps(&mut out_k[y * ow + x], out_val);
                        }
                    });
                });
        });
}

#[inline]
fn process_conv2d_chunk(
    inp_n: &[f32],
    weight_k: &[f32],
    c_idx: usize,
    weight_stride: usize,
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
    acc: __m256,
) -> __m256 {
    let sum = (0..r)
        .flat_map(|fy_idx| (0..s).map(move |fx_idx| (fy_idx, fx_idx)))
        .fold(acc, |acc, (fy_idx, fx_idx)| {
            let iy = y * sh + fy_idx - ph;
            let ix = x * sw + fx_idx - pw;
            if is_valid_index(iy, ix, h, w) {
                let inp_offset = c_idx + iy * w + ix;
                let weight_offset = c_idx * weight_stride + fy_idx * s + fx_idx;
                let inp_val = unsafe { _mm256_loadu_ps(&inp_n[inp_offset]) };
                let weight_val = unsafe { _mm256_loadu_ps(&weight_k[weight_offset]) };
                unsafe { _mm256_fmadd_ps(inp_val, weight_val, acc) }
            } else {
                acc
            }
        });
    sum
}

#[inline]
fn process_conv2d_single(
    inp_n: &[f32],
    weight_k: &[f32],
    c_idx: usize,
    weight_stride: usize,
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
    acc: __m256,
) -> __m256 {
    let sum = (0..r)
        .flat_map(|fy_idx| (0..s).map(move |fx_idx| (fy_idx, fx_idx)))
        .fold(acc, |acc, (fy_idx, fx_idx)| {
            let iy = y * sh + fy_idx - ph;
            let ix = x * sw + fx_idx - pw;
            if is_valid_index(iy, ix, h, w) {
                let inp_offset = c_idx + iy * w + ix;
                let weight_offset = c_idx * weight_stride + fy_idx * s + fx_idx;
                let inp_val = inp_n[inp_offset];
                let weight_val = weight_k[weight_offset];
                unsafe { _mm256_add_ps(acc, _mm256_set1_ps(inp_val * weight_val)) }
            } else {
                acc
            }
        });
    sum
}

#[inline]
fn is_valid_index(iy: usize, ix: usize, h: usize, w: usize) -> bool {
    iy < h && ix < w
}