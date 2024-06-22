use super::*;
use approx::assert_relative_eq;
use ndarray::{Array, Array2};

#[test]
fn test_linear_forward_small() {
    let inp = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let bias = vec![0.1, 0.2];
    let mut out = vec![0.0; 4];

    linear_forward(&mut out, &inp, &weight, &bias, 2, 2, 2);

    let expected = vec![
        1.0 * 0.1 + 2.0 * 0.3 + 0.1,
        1.0 * 0.2 + 2.0 * 0.4 + 0.2,
        3.0 * 0.5 + 4.0 * 0.7 + 0.1,
        3.0 * 0.6 + 4.0 * 0.8 + 0.2,
    ];

    assert_relative_eq!(Array::from(out), Array::from(expected), epsilon = 1e-5);
}

#[test]
fn test_linear_forward_large() {
    let m = 128;
    let k = 256;
    let n = 512;

    let inp: Vec<f32> = (0..m*k).map(|i| i as f32).collect();
    let weight: Vec<f32> = (0..k*n).map(|i| (i % 10) as f32 * 0.1).collect();
    let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let mut out = vec![0.0; m * n];

    linear_forward(&mut out, &inp, &weight, &bias, m, k, n);

    // Compute expected output using ndarray for verification
    let inp_arr = Array::from_shape_vec((m, k), inp.clone()).unwrap();
    let weight_arr = Array::from_shape_vec((k, n), weight.clone()).unwrap();
    let bias_arr = Array::from_shape_vec(n, bias.clone()).unwrap();

    let expected = inp_arr.dot(&weight_arr) + &bias_arr;

    assert_relative_eq!(
        Array::from_shape_vec((m, n), out).unwrap(),
        expected,
        epsilon = 1e-3
    );
}

#[test]
fn test_dropout() {
    let mut x = vec![1.0; 1000];
    let prob = 0.3;

    dropout(&mut x, prob, true);

    let zero_count = x.iter().filter(|&&v| v == 0.0).count();
    let nonzero_count = x.len() - zero_count;

    // Check if the number of zeroes is roughly 30% (with some tolerance)
    assert!((zero_count as f32 / x.len() as f32 - prob).abs() < 0.05);

    // Check if non-zero values are scaled correctly
    let expected_scale = 1.0 / (1.0 - prob);
    for &value in x.iter().filter(|&&v| v != 0.0) {
        assert_relative_eq!(value, expected_scale, epsilon = 1e-5);
    }

    // Test when train is false
    let mut x = vec![1.0; 1000];
    dropout(&mut x, prob, false);
    assert!(x.iter().all(|&v| v == 1.0));
}

#[test]
fn test_linear_forward_different_sizes() {
    let test_cases = vec![
        (1, 1, 1),
        (10, 5, 3),
        (100, 50, 25),
        (64, 128, 256),
    ];

    for (m, k, n) in test_cases {
        let inp: Vec<f32> = (0..m*k).map(|i| i as f32).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i % 10) as f32 * 0.1).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let mut out = vec![0.0; m * n];

        linear_forward(&mut out, &inp, &weight, &bias, m, k, n);

        let inp_arr = Array::from_shape_vec((m, k), inp.clone()).unwrap();
        let weight_arr = Array::from_shape_vec((k, n), weight.clone()).unwrap();
        let bias_arr = Array::from_shape_vec(n, bias.clone()).unwrap();

        let expected = inp_arr.dot(&weight_arr) + &bias_arr;

        assert_relative_eq!(
            Array::from_shape_vec((m, n), out).unwrap(),
            expected,
            epsilon = 1e-3
        );
    }
}