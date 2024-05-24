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

    inp.chunks_exact(c * c_stride)
        .zip(out.chunks_exact_mut(k * k_stride))
        .for_each(|(inp_n, out_n)| {
            weight
                .chunks_exact(c * weight_stride)
                .zip(bias.iter())
                .zip(out_n.chunks_exact_mut(k_stride))
                .for_each(|((weight_k, &bias_k), out_k)| {
                    for y in 0..oh {
                        for x in 0..ow {
                            let mut sum = unsafe { _mm256_setzero_ps() };

                            inp_n
                                .chunks_exact(c_stride)
                                .zip(weight_k.chunks_exact(weight_stride))
                                .for_each(|(inp_c, weight_c)| {
                                    for fy in 0..r {
                                        for fx in 0..s {
                                            let iy = y * sh + fy - ph;
                                            let ix = x * sw + fx - pw;

                                            if iy < 0 || iy >= h || ix < 0 || ix >= w {
                                                continue;
                                            }

                                            let inp_val = unsafe {
                                                _mm256_set1_ps(inp_c[iy * w + ix])
                                            };
                                            let weight_val = unsafe {
                                                _mm256_loadu_ps(&weight_c[fy * s + fx])
                                            };
                                            sum = unsafe { _mm256_fmadd_ps(inp_val, weight_val, sum) };
                                        }
                                    }
                                });

                            let bias_val = unsafe { _mm256_set1_ps(bias_k) };
                            let out_val = unsafe { _mm256_add_ps(sum, bias_val) };
                            unsafe {
                                _mm256_storeu_ps(&mut out_k[y * ow + x], out_val);
                            }
                        }
                    }
                });
        });
}