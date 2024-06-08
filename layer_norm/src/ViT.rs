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
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    dropout: f32,
}

impl ViTAttention {
    fn new(config: &ViTConfig) -> Self {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        ViTAttention {
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
            query: vec![0.0; config.hidden_size * all_head_size],
            key: vec![0.0; config.hidden_size * all_head_size],
            value: vec![0.0; config.hidden_size * all_head_size],
            dropout: config.attention_probs_dropout_prob,
        }
    }

    fn forward(&self, hidden_states: &[f32], output_attentions: bool) -> (Vec<f32>, Option<Vec<f32>>) {
        let bs = hidden_states.len() / self.all_head_size;
        let mut q = vec![0.0; bs * self.all_head_size];
        let mut k = vec![0.0; bs * self.all_head_size];
        let mut v = vec![0.0; bs * self.all_head_size];

        linear_forward(
            &mut q,
            hidden_states,
            &self.query,
            &[0.0; 1024],
            bs,
            self.all_head_size,
            self.all_head_size,
        );
        linear_forward(
            &mut k,
            hidden_states,
            &self.key,
            &[0.0; 1024],
            bs,
            self.all_head_size,
            self.all_head_size,
        );
        linear_forward(
            &mut v,
            hidden_states,
            &self.value,
            &[0.0; 1024],
            bs,
            self.all_head_size,
            self.all_head_size,
        );

        let mut attn_output = vec![0.0; bs * self.all_head_size];
        let mut attn_probs = None;
        if output_attentions {
            attn_probs = Some(vec![0.0; bs * self.num_attention_heads * (hidden_states.len() / self.all_head_size)]);
        }

        multihead_attention_forward(
            &mut attn_output,
            &q,
            &k,
            &v,
            &self.query,
            &self.key,
            &self.value,
            &[0.0; 1024],
            bs,
            hidden_states.len() / self.all_head_size,
            self.all_head_size,
            self.num_attention_heads,
        );

        dropout(&mut attn_output, self.dropout, true);

        (attn_output, attn_probs)
    }
}


struct ViTIntermediate {
    dense: Vec<f32>,
    intermediate_act_fn: fn(&mut [f32]),
}

impl ViTIntermediate {
    fn new(config: &ViTConfig) -> Self {
        ViTIntermediate {
            dense: vec![0.0; config.hidden_size * config.intermediate_size],
            intermediate_act_fn: match config.hidden_act {
                "gelu" => gelu,
                "relu" => relu,
                _ => panic!("Unsupported activation function"),
            },
        }
    }

    fn forward(&self, hidden_states: &[f32]) -> Vec<f32> {
        let bs = hidden_states.len() / self.dense.len();
        let mut intermediate_output = vec![0.0; bs * self.dense.len()];
        linear_forward(
            &mut intermediate_output,
            hidden_states,
            &self.dense,
            &[0.0; 1024],
            bs,
            hidden_states.len(),
            self.dense.len(),
        );
        (self.intermediate_act_fn)(&mut intermediate_output);
        intermediate_output
    }
}

struct ViTOutput {
    dense: Vec<f32>,
    dropout: f32,
}

impl ViTOutput {
    fn new(config: &ViTConfig) -> Self {
        ViTOutput {
            dense: vec![0.0; config.intermediate_size * config.hidden_size],
            dropout: config.hidden_dropout_prob,
        }
    }

    fn forward(&self, hidden_states: &[f32], input_tensor: &[f32]) -> Vec<f32> {
        let bs = hidden_states.len() / self.dense.len();
        let mut output = vec![0.0; bs * self.dense.len()];
        linear_forward(
            &mut output,
            hidden_states,
            &self.dense,
            &[0.0; 1024],
            bs,
            hidden_states.len(),
            self.dense.len(),
        );
        dropout(&mut output, self.dropout, true);

        output
            .iter()
            .zip(input_tensor.iter())
            .map(|(&x, &y)| x + y)
            .collect()
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

    fn forward(
        &self,
        hidden_states: &[f32],
        output_attentions: bool,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
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