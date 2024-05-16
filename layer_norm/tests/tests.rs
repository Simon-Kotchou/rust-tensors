// use serde::Deserialize;
// use std::fs::File;
// use std::io::Read;
// use approx::assert_abs_diff_eq;
// use layer_norm::layernorm_forward;

// #[derive(Deserialize)]
// struct TestCase {
//     input: Vec<Vec<Vec<f32>>>,
//     weight: Vec<f32>,
//     bias: Vec<f32>,
//     output: Vec<Vec<Vec<f32>>>,
//     b: usize,
//     t: usize,
//     c: usize,
// }

// #[test]
// fn test_layernorm_forward() {
//     // Read the JSON file
//     let mut file = File::open("tests/layernorm_test_cases.json").unwrap();
//     let mut json_str = String::new();
//     file.read_to_string(&mut json_str).unwrap();

//     // Deserialize the JSON string into a vector of TestCase structs
//     let test_cases: Vec<TestCase> = serde_json::from_str(&json_str).unwrap();

//     // Iterate over the test cases
//     for test_case in test_cases {
//         let inp: Vec<f32> = test_case.input.into_iter().flatten().flatten().collect();
//         let weight: Vec<f32> = test_case.weight;
//         let bias: Vec<f32> = test_case.bias;
//         let expected_output: Vec<f32> = test_case.output.into_iter().flatten().flatten().collect();
//         let mut out: Vec<f32> = vec![0.0; test_case.b * test_case.t * test_case.c];
//         let mut mean: Vec<f32> = vec![0.0; test_case.b * test_case.t];
//         let mut rstd: Vec<f32> = vec![0.0; test_case.b * test_case.t];

//         layernorm_forward(
//             &mut out,
//             &mut mean,
//             &mut rstd,
//             &inp,
//             &weight,
//             &bias,
//             test_case.b,
//             test_case.t,
//             test_case.c,
//         );

//         // Compare the output with the expected output using approximate equality
//         for (x, y) in out.iter().zip(expected_output.iter()) {
//             assert_abs_diff_eq!(x, y, epsilon = 1e-5);
//         }
//     }
// }

// #[test]
// fn test_layernorm_forward_mse() {
//     // Read the JSON file
//     let mut file: File = File::open("tests/layernorm_test_cases.json").unwrap();
//     let mut json_str: String = String::new();
//     file.read_to_string(&mut json_str).unwrap();

//     // Deserialize the JSON string into a vector of TestCase structs
//     let test_cases: Vec<TestCase> = serde_json::from_str(&json_str).unwrap();

//     let mut total_mse: f32 = 0.0;
//     let mut num_elements: usize = 0;

//     // Iterate over the test cases
//     for test_case in test_cases {
//         let inp: Vec<f32> = test_case.input.into_iter().flatten().flatten().collect();
//         let weight: Vec<f32> = test_case.weight;
//         let bias: Vec<f32> = test_case.bias;
//         let expected_output: Vec<f32> = test_case.output.into_iter().flatten().flatten().collect();
//         let mut out: Vec<f32> = vec![0.0; test_case.b * test_case.t * test_case.c];
//         let mut mean: Vec<f32> = vec![0.0; test_case.b * test_case.t];
//         let mut rstd: Vec<f32> = vec![0.0; test_case.b * test_case.t];

//         layernorm_forward(
//             &mut out,
//             &mut mean,
//             &mut rstd,
//             &inp,
//             &weight,
//             &bias,
//             test_case.b,
//             test_case.t,
//             test_case.c,
//         );

//         // Compute the MSE for this test case
//         let mut mse: f32 = 0.0;
//         for (x, y) in out.iter().zip(expected_output.iter()) {
//             let diff = x - y;
//             mse += diff * diff;
//         }
//         mse /= test_case.b as f32 * test_case.t as f32 * test_case.c as f32;

//         total_mse += mse;
//         num_elements += test_case.b * test_case.t * test_case.c;
//     }

//     // Compute the average MSE across all test cases
//     let avg_mse: f32 = total_mse / num_elements as f32;
//     println!("Average MSE: {}", avg_mse);

//     // Assert that the average MSE is within an acceptable threshold
//     assert!(avg_mse < 1e-6, "Average MSE exceeds the acceptable threshold");
// }
use layer_norm::{layernorm_forward, AlignedF32};
use approx::assert_abs_diff_eq;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;

#[derive(Deserialize)]
struct TestCase {
    input: Vec<Vec<Vec<f32>>>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    output: Vec<Vec<Vec<f32>>>,
    b: usize,
    t: usize,
    c: usize,
}

#[test]
fn test_layernorm_forward() {
    // Read the JSON file
    let mut file = File::open("tests/layernorm_test_cases.json").unwrap();
    let mut json_str = String::new();
    file.read_to_string(&mut json_str).unwrap();

    // Deserialize the JSON string into a vector of TestCase structs
    let test_cases: Vec<TestCase> = serde_json::from_str(&json_str).unwrap();

    // Iterate over the test cases
    for test_case in test_cases {
        let inp_flat: Vec<f32> = test_case.input.into_iter().flatten().flatten().collect();
        let weight: Vec<f32> = test_case.weight;
        let bias: Vec<f32> = test_case.bias;
        let expected_output: Vec<f32> = test_case.output.into_iter().flatten().flatten().collect();

        let mut inp = AlignedF32::new(test_case.b * test_case.t * test_case.c);
        let mut weight_aligned = AlignedF32::new(test_case.c);
        let mut bias_aligned = AlignedF32::new(test_case.c);
        let mut out = AlignedF32::new(test_case.b * test_case.t * test_case.c);
        let mut mean = vec![0.0; test_case.b * test_case.t];
        let mut rstd = vec![0.0; test_case.b * test_case.t];

        inp.as_mut_slice().copy_from_slice(&inp_flat);
        weight_aligned.as_mut_slice().copy_from_slice(&weight);
        bias_aligned.as_mut_slice().copy_from_slice(&bias);

        layernorm_forward(
            &mut out,
            &mut mean,
            &mut rstd,
            &inp,
            &weight_aligned,
            &bias_aligned,
            test_case.b,
            test_case.t,
            test_case.c,
        );

        // Compare the output with the expected output using approximate equality
        for (x, y) in out.as_slice().iter().zip(expected_output.iter()) {
            assert_abs_diff_eq!(x, y, epsilon = 1e-5);
        }
    }
}

#[test]
fn test_layernorm_forward_mse() {
    // Read the JSON file
    let mut file = File::open("tests/layernorm_test_cases.json").unwrap();
    let mut json_str = String::new();
    file.read_to_string(&mut json_str).unwrap();

    // Deserialize the JSON string into a vector of TestCase structs
    let test_cases: Vec<TestCase> = serde_json::from_str(&json_str).unwrap();

    let mut total_mse: f32 = 0.0;
    let mut num_elements: usize = 0;

    // Iterate over the test cases
    for test_case in test_cases {
        let inp_flat: Vec<f32> = test_case.input.into_iter().flatten().flatten().collect();
        let weight: Vec<f32> = test_case.weight;
        let bias: Vec<f32> = test_case.bias;
        let expected_output: Vec<f32> = test_case.output.into_iter().flatten().flatten().collect();

        let mut inp = AlignedF32::new(test_case.b * test_case.t * test_case.c);
        let mut weight_aligned = AlignedF32::new(test_case.c);
        let mut bias_aligned = AlignedF32::new(test_case.c);
        let mut out = AlignedF32::new(test_case.b * test_case.t * test_case.c);
        let mut mean = vec![0.0; test_case.b * test_case.t];
        let mut rstd = vec![0.0; test_case.b * test_case.t];

        inp.as_mut_slice().copy_from_slice(&inp_flat);
        weight_aligned.as_mut_slice().copy_from_slice(&weight);
        bias_aligned.as_mut_slice().copy_from_slice(&bias);

        layernorm_forward(
            &mut out,
            &mut mean,
            &mut rstd,
            &inp,
            &weight_aligned,
            &bias_aligned,
            test_case.b,
            test_case.t,
            test_case.c,
        );

        // Compute the MSE for this test case
        let mut mse: f32 = 0.0;
        for (x, y) in out.as_slice().iter().zip(expected_output.iter()) {
            let diff = x - y;
            mse += diff * diff;
        }
        mse /= test_case.b as f32 * test_case.t as f32 * test_case.c as f32;

        total_mse += mse;
        num_elements += test_case.b * test_case.t * test_case.c;
    }

    // Compute the average MSE across all test cases
    let avg_mse: f32 = total_mse / num_elements as f32;
    println!("Average MSE: {}", avg_mse);

    // Assert that the average MSE is within an acceptable threshold
    assert!(avg_mse < 1e-6, "Average MSE exceeds the acceptable threshold");
}