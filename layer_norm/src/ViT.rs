use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use std::arch::x86_64::*;

#[derive(Clone)]
struct ViTConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    hidden_act: &'static str,
    hidden_dropout_prob: f32,
    attention_probs_dropout_prob: f32,
    image_size: usize,
    patch_size: usize,
}

struct ViTEmbeddings {
    patch_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,
    dropout: f32,
}

impl ViTEmbeddings {
    fn new(config: &ViTConfig) -> Self {
        let patch_size = config.patch_size;
        let num_patches = (config.image_size / patch_size) * (config.image_size / patch_size);
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, config.hidden_size as f64 / num_patches as f64).unwrap();
        let patch_embeddings = Array2::random((num_patches, config.hidden_size), normal);
        let position_embeddings = Array2::random((1, num_patches + 1), normal);
        ViTEmbeddings {
            patch_embeddings,
            position_embeddings,
            dropout: config.hidden_dropout_prob,
        }
    }

    fn forward(&self, pixel_values: &Array2<f32>) -> Array2<f32> {
        let num_patches = self.patch_embeddings.shape()[0];
        let mut embeddings = Array2::zeros((1, num_patches + 1, self.patch_embeddings.shape()[1]));
        let patch_embeddings = self.patch_embeddings.clone();
        let position_embeddings = self.position_embeddings.clone();
        embeddings
            .slice_mut(s![0, 1.., ..])
            .assign(&(&patch_embeddings * pixel_values).sum_axis(Axis(1)));
        embeddings += &position_embeddings;
        embeddings.map(|v| if rand::random::<f32>() < self.dropout { 0.0 } else { *v })
    }
}

struct ViTAttention {
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    query: Array2<f32>,
    key: Array2<f32>,
    value: Array2<f32>,
    dropout: f32,
}

impl ViTAttention {
    fn new(config: &ViTConfig) -> Self {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, config.hidden_size as f64 / all_head_size as f64).unwrap();
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

    fn forward(&self, hidden_states: &Array2<f32>, output_attentions: bool) -> (Array2<f32>, Option<Array2<f32>>) {
        let bs = hidden_states.shape()[0];
        let mut q = hidden_states.dot(&self.query).into_shape((bs, -1, self.num_attention_heads, self.attention_head_size)).unwrap();
        let mut k = hidden_states.dot(&self.key).into_shape((bs, -1, self.num_attention_heads, self.attention_head_size)).unwrap();
        let mut v = hidden_states.dot(&self.value).into_shape((bs, -1, self.num_attention_heads, self.attention_head_size)).unwrap();

        q.swap_axes(1, 2);
        k.swap_axes(1, 2);
        v.swap_axes(1, 2);

        let mut attn_scores = q.dot(&k.t()) / (self.attention_head_size as f32).sqrt();
        let attn_probs = attn_scores.map(|v| v.exp()) / attn_scores.map(|v| v.exp()).sum_axis(Axis(2));
        let attn_probs = attn_probs.map(|v| if rand::random::<f32>() < self.dropout { 0.0 } else { *v });

        let mut attn_output = attn_probs.dot(&v);
        attn_output.swap_axes(1, 2);
        attn_output.into_shape((bs, -1, self.all_head_size));

        let attn_weights = if output_attentions { Some(attn_probs) } else { None };

        (attn_output, attn_weights)
    }
}

struct ViTIntermediate {
    dense: Array2<f32>,
    intermediate_act_fn: fn(&mut Array2<f32>),
}

impl ViTIntermediate {
    fn new(config: &ViTConfig) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, config.hidden_size as f64 / config.intermediate_size as f64).unwrap();
        ViTIntermediate {
            dense: Array2::random((config.hidden_size, config.intermediate_size), normal),
            intermediate_act_fn: match config.hidden_act {
                "gelu" => Self::gelu,
                "relu" => Self::relu,
                _ => panic!("Unsupported activation function"),
            },
        }
    }

    fn forward(&self, hidden_states: &Array2<f32>) -> Array2<f32> {
        let mut intermediate_output = hidden_states.dot(&self.dense);
        (self.intermediate_act_fn)(&mut intermediate_output);
        intermediate_output
    }

    fn gelu(x: &mut Array2<f32>) {
        x.mapv_inplace(|v| {
            let cdf = 0.5 * (1.0 + ((v / (2.0 as f32).sqrt()) * (1.0 + 0.044715 * v * v)).tanh());
            v * cdf
        });
    }

    fn relu(x: &mut Array2<f32>) {
        x.mapv_inplace(|v| v    }
}

struct ViTOutput {
    dense:    dropout: f32,
}

impl ViTOutput {
    fn new(config: &ViTConfig) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, config.intermediate_size as f64 / config.hidden_size as f64).unwrap();
        ViTOutput {
            dense: Array2::random((config.intermediate_size, config.hidden_size), normal),
            dropout: config.hidden_dropout_prob,
        }
    }

    fn forward(&self, hidden_states: &Array2<f32>, input_tensor: &Array2<f32>) -> Array2<f32> {
        let mut output = hidden_states.dot(&self.dense);
        output.map(|v| if rand::random::<f32>() < self.dropout { 0.0 } else { *v });
        output + input_tensor
    }
}

struct ViTLayer {
    chunk_size_feed_forward: usize,
    attention: ViTAttention,
    intermediate: ViTIntermediate,
    output: ViTOutput,
}

impl ViTLayer {
    fn new(config: &ViTConfig) -> Self {
        ViTLayer {
            chunk_size_feed_forward: config.hidden_size,
            attention: ViTAttention::new(config),
            intermediate: ViTIntermediate::new(config),
            output: ViTOutput::new(config),
        }
    }

    fn forward(&self, hidden_states: &Array2<f32>, output_attentions: bool) -> (Array2<f32>, Option<Array2<f32>>) {
        let (attn_output, attn_weights) = self.attention.forward(hidden_states, output_attentions);
        let intermediate_output = self.intermediate.forward(&attn_output);
        let layer_output = self.output.forward(&intermediate_output, hidden_states);
        (layer_output, attn_weights)
    }
}

struct ViTEncoder {
    layer: Vec<ViTLayer>,
}

impl ViTEncoder {
    fn new(config: &ViTConfig) -> Self {
        let mut layer = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layer.push(ViTLayer::new(config));
        }
        ViTEncoder { layer }
    }

    fn forward(
        &self,
        hidden_states: &[f32],
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> (Vec<f32>, Option<Vec<Vec<f32>>>, Option<Vec<Vec<f32>>>) {
        let mut all_hidden_states = if output_hidden_states {
            Some(vec![hidden_states.to_vec()])
        } else {
            None
        };

        let mut all_attentions = if output_attentions {
            Some(Vec::new())
        } else {
            None
        };

        let mut hidden_states = hidden_states.to_vec();
        for layer_module in &self.layer {
            if let Some(hidden_states_vec) = all_hidden_states.as_mut() {
                hidden_states_vec.push(hidden_states.clone());
            }

            let (layer_output, attn_weights) = layer_module.forward(&hidden_states, output_attentions);
            hidden_states = layer_output;

            if let Some(attentions_vec) = all_attentions.as_mut() {
                if let Some(attn_weights) = attn_weights {
                    attentions_vec.push(attn_weights);
                }
            }
        }

        if let Some(hidden_states_vec) = all_hidden_states.as_mut() {
            hidden_states_vec.push(hidden_states.clone());
        }

        (hidden_states, all_hidden_states, all_attentions)
    }
}