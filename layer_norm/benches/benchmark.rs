use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use layer_norm::{layernorm_forward, AlignedF32};

fn generate_random_data(b: usize, t: usize, c: usize) -> (AlignedF32, AlignedF32, AlignedF32, AlignedF32, Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut inp = AlignedF32::new(b * t * c);
    let mut weight = AlignedF32::new(c);
    let mut bias = AlignedF32::new(c);
    let out = AlignedF32::new(b * t * c);
    let mean = vec![0.0; b * t];
    let rstd = vec![0.0; b * t];

    for i in 0..b * t * c {
        inp.as_mut_slice()[i] = rng.gen();
    }
    for i in 0..c {
        weight.as_mut_slice()[i] = rng.gen();
        bias.as_mut_slice()[i] = rng.gen();
    }

    (inp, weight, bias, out, mean, rstd)
}

fn bench_layernorm_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("LayerNorm");

    for &(b, t, c) in &[(1, 128, 256), (4, 8, 512), (16, 32, 1024)] {
        let (inp, weight, bias, mut out, mut mean, mut rstd) = generate_random_data(b, t, c);

        group.bench_with_input(format!("b={}, t={}, c={}", b, t, c), &(b, t, c), |bencher, &(b, t, c)| {
            bencher.iter(|| {
                layernorm_forward(
                    black_box(&mut out),
                    black_box(&mut mean),
                    black_box(&mut rstd),
                    black_box(&inp),
                    black_box(&weight),
                    black_box(&bias),
                    b,
                    t,
                    c,
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_layernorm_forward);
criterion_main!(benches);