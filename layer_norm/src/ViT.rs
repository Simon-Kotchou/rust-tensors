use rand::Rng;
use std::arch::x86_64::*;

struct ViTConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    hidden_act: &'static str,
    hidden_dropout_prob: f32,
    attention_probs_dropout_prob: f32,
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