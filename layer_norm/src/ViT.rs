#![feature(stdsimd)]

use ndarray::{Array, Array2, ArrayView2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use rayon::prelude::*;
use std::arch::x86_64::*;
use thiserror::Error;

/// Configuration for the Vision Transformer model.
#[derive(Clone, Debug)]
pub struct ViTConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub hidden_act: ActivationFunction,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub image_size: usize,
    pub patch_size: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum ActivationFunction {
    Gelu,
    Relu,
}

#[derive(Error, Debug)]
pub enum ViTError {
    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidInputShape { expected: Vec<usize>, actual: Vec<usize> },
    #[error("Computation error: {0}")]
    ComputationError(String),
}

type Result<T> = std::result::Result<T, ViTError>;

/// Vision Transformer Embeddings
pub struct ViTEmbeddings {
    patch_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,
    dropout: f32,
}

impl ViTEmbeddings {
    pub fn new(config: &ViTConfig) -> Self {
        let patch_size = config.patch_size;
        let num_patches = (config.image_size / patch_size).pow(2);
        let normal = Normal::new(0.0, (config.hidden_size as f64).recip()).unwrap();
        
        ViTEmbeddings {
            patch_embeddings: Array2::random((num_patches, config.hidden_size), normal),
            position_embeddings: Array2::random((1, num_patches + 1, config.hidden_size), normal),
            dropout: config.hidden_dropout_prob,
        }
    }

    pub fn forward(&self, pixel_values: &Array2<f32>) -> Result<Array2<f32>> {
        let (num_patches, hidden_size) = self.patch_embeddings.dim();
        let mut embeddings = Array2::zeros((1, num_patches + 1, hidden_size));
        
        // Compute patch embeddings
        embeddings
            .slice_mut(s![0, 1.., ..])
            .assign(&pixel_values.dot(&self.patch_embeddings));

        // Add position embeddings
        embeddings += &self.position_embeddings;

        // Apply dropout
        apply_dropout(&mut embeddings, self.dropout);

        Ok(embeddings)
    }
}

/// Vision Transformer Attention mechanism
pub struct ViTAttention {
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    query: Array2<f32>,
    key: Array2<f32>,
    value: Array2<f32>,
    dropout: f32,
}

impl ViTAttention {
    pub fn new(config: &ViTConfig) -> Self {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let normal = Normal::new(0.0, (config.hidden_size as f64 / all_head_size as f64).sqrt()).unwrap();
        
        ViTAttention {
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
            query: Array2::random((config.hidden_size, all_head_size), normal),
            key: Array2::random((config.hidden_size, all_head_size), normal),
            value: Array2::random((config.hidden_size, all_head_size), normal),
            dropout: config.attention_probs_dropout_prob,
        }
    }

    pub fn forward(&self, hidden_states: &Array2<f32>, output_attentions: bool) -> Result<(Array2<f32>, Option<Array2<f32>>)> {
        let (bs, seq_len, _) = hidden_states.dim();

        let q = self.project_and_transpose(hidden_states, &self.query)?;
        let k = self.project_and_transpose(hidden_states, &self.key)?;
        let v = self.project_and_transpose(hidden_states, &self.value)?;

        let attn_scores = self.compute_attention_scores(&q, &k)?;
        let attn_probs = self.compute_attention_probs(&attn_scores)?;

        let attn_output = self.compute_attention_output(&attn_probs, &v)?;
        let attn_output = attn_output.into_shape((bs, seq_len, self.all_head_size))?;

        let attn_weights = if output_attentions { Some(attn_probs) } else { None };

        Ok((attn_output, attn_weights))
    }

    fn project_and_transpose(&self, hidden_states: &Array2<f32>, weight: &Array2<f32>) -> Result<Array2<f32>> {
        let (bs, seq_len, _) = hidden_states.dim();
        hidden_states
            .dot(weight)
            .into_shape((bs, seq_len, self.num_attention_heads, self.attention_head_size))?
            .permuted_axes([0, 2, 1, 3])
            .into_shape((bs * self.num_attention_heads, seq_len, self.attention_head_size))
            .map_err(|e| ViTError::ComputationError(e.to_string()))
    }

    fn compute_attention_scores(&self, q: &Array2<f32>, k: &Array2<f32>) -> Result<Array2<f32>> {
        let (_, seq_len, _) = q.dim();
        q.dot(&k.t())
            .mapv(|x| x / (self.attention_head_size as f32).sqrt())
            .into_shape((_, seq_len, seq_len))
            .map_err(|e| ViTError::ComputationError(e.to_string()))
    }

    fn compute_attention_probs(&self, attn_scores: &Array2<f32>) -> Result<Array2<f32>> {
        let mut attn_probs = attn_scores.mapv(|x| x.exp());
        attn_probs /= &attn_probs.sum_axis(Axis(2)).insert_axis(Axis(2));
        apply_dropout(&mut attn_probs, self.dropout);
        Ok(attn_probs)
    }

    fn compute_attention_output(&self, attn_probs: &Array2<f32>, v: &Array2<f32>) -> Result<Array2<f32>> {
        attn_probs
            .dot(v)
            .into_shape((_, self.num_attention_heads, _, self.attention_head_size))
            .map_err(|e| ViTError::ComputationError(e.to_string()))
    }
}

/// Vision Transformer Intermediate layer
pub struct ViTIntermediate {
    dense: Array2<f32>,
    activation_fn: ActivationFunction,
}

impl ViTIntermediate {
    pub fn new(config: &ViTConfig) -> Self {
        let normal = Normal::new(0.0, (config.hidden_size as f64 / config.intermediate_size as f64).sqrt()).unwrap();
        ViTIntermediate {
            dense: Array2::random((config.hidden_size, config.intermediate_size), normal),
            activation_fn: config.hidden_act,
        }
    }

    pub fn forward(&self, hidden_states: &Array2<f32>) -> Result<Array2<f32>> {
        let mut intermediate_output = hidden_states.dot(&self.dense);
        apply_activation_function(&mut intermediate_output, self.activation_fn);
        Ok(intermediate_output)
    }
}

/// Vision Transformer Output layer
pub struct ViTOutput {
    dense: Array2<f32>,
    dropout: f32,
}

impl ViTOutput {
    pub fn new(config: &ViTConfig) -> Self {
        let normal = Normal::new(0.0, (config.intermediate_size as f64 / config.hidden_size as f64).sqrt()).unwrap();
        ViTOutput {
            dense: Array2::random((config.intermediate_size, config.hidden_size), normal),
            dropout: config.hidden_dropout_prob,
        }
    }

    pub fn forward(&self, hidden_states: &Array2<f32>, input_tensor: &Array2<f32>) -> Result<Array2<f32>> {
        let mut output = hidden_states.dot(&self.dense);
        apply_dropout(&mut output, self.dropout);
        Ok(output + input_tensor)
    }
}

/// Vision Transformer Layer
pub struct ViTLayer {
    attention: ViTAttention,
    intermediate: ViTIntermediate,
    output: ViTOutput,
}

impl ViTLayer {
    pub fn new(config: &ViTConfig) -> Self {
        ViTLayer {
            attention: ViTAttention::new(config),
            intermediate: ViTIntermediate::new(config),
            output: ViTOutput::new(config),
        }
    }

    pub fn forward(&self, hidden_states: &Array2<f32>, output_attentions: bool) -> Result<(Array2<f32>, Option<Array2<f32>>)> {
        let (attn_output, attn_weights) = self.attention.forward(hidden_states, output_attentions)?;
        let intermediate_output = self.intermediate.forward(&attn_output)?;
        let layer_output = self.output.forward(&intermediate_output, hidden_states)?;
        Ok((layer_output, attn_weights))
    }
}

/// Vision Transformer Encoder
pub struct ViTEncoder {
    layers: Vec<ViTLayer>,
}

impl ViTEncoder {
    pub fn new(config: &ViTConfig) -> Self {
        ViTEncoder {
            layers: (0..config.num_hidden_layers).map(|_| ViTLayer::new(config)).collect(),
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Array2<f32>,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> Result<(Array2<f32>, Option<Vec<Array2<f32>>>, Option<Vec<Array2<f32>>>)> {
        let mut all_hidden_states = if output_hidden_states { Some(vec![hidden_states.clone()]) } else { None };
        let mut all_attentions = if output_attentions { Some(Vec::new()) } else { None };

        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            if let Some(hidden_states_vec) = &mut all_hidden_states {
                hidden_states_vec.push(hidden_states.clone());
            }

            let (layer_output, attn_weights) = layer.forward(&hidden_states, output_attentions)?;
            hidden_states = layer_output;

            if let Some(attentions_vec) = &mut all_attentions {
                if let Some(attn_weights) = attn_weights {
                    attentions_vec.push(attn_weights);
                }
            }
        }

        if let Some(hidden_states_vec) = &mut all_hidden_states {
            hidden_states_vec.push(hidden_states.clone());
        }

        Ok((hidden_states, all_hidden_states, all_attentions))
    }
}

// Utility functions

fn apply_dropout(arr: &mut Array2<f32>, dropout_prob: f32) {
    arr.par_mapv_inplace(|x| {
        if rand::random::<f32>() < dropout_prob {
            0.0
        } else {
            x / (1.0 - dropout_prob)
        }
    });
}

fn apply_activation_function(arr: &mut Array2<f32>, activation_fn: ActivationFunction) {
    match activation_fn {
        ActivationFunction::Gelu => arr.mapv_inplace(gelu),
        ActivationFunction::Relu => arr.mapv_inplace(|x| x.max(0.0)),
    }
}

#[inline(always)]
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEFF: f32 = 0.044715;
    let x_cubed = x * x * x;
    x * 0.5 * (1.0 + (SQRT_2_OVER_PI * (x + COEFF * x_cubed)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vit_embeddings() {
        let config = ViTConfig {
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            hidden_act: ActivationFunction::Gelu,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            image_size: 224,
            patch_size: 16,
        };

        let embeddings = ViTEmbeddings::new(&config);
        let pixel_values = Array2::zeros((1, 196, 768));
        let result = embeddings.forward(&pixel_values).unwrap();
        
        assert_eq!(result.shape(), &[1, 197, 768]);
    }

}