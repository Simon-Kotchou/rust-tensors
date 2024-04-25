use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use layer_norm::layernorm_forward;

fn generate_random_data(b: usize, t: usize, c: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let inp: Vec<f32> = (0..b * t * c).map(|_| rng.gen()).collect();
    let weight: Vec<f32> = (0..c).map(|_| rng.gen()).collect();
    let bias: Vec<f32> = (0..c).map(|_| rng.gen()).collect();
    let out: Vec<f32> = vec![0.0; b * t * c];
    let mean: Vec<f32> = vec![0.0; b * t];
    let rstd: Vec<f32> = vec![0.0; b * t];
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