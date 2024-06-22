use std::sync::Arc;
use ndarray::{Array, Array2, ArrayView2, Axis};
use rayon::prelude::*;
use num_cpus;
use core_affinity;

// CPU-specific configuration
pub struct CPUConfig {
    pub num_physical_cores: usize,
    pub use_logical_cores: bool,
    pub numa_node: Option<usize>,
}

impl CPUConfig {
    pub fn new() -> Self {
        let num_physical_cores = num_cpus::get_physical();
        CPUConfig {
            num_physical_cores,
            use_logical_cores: false,
            numa_node: None,
        }
    }
}

// Modified ViTConfig to include CPU configuration
pub struct ViTConfig {
    // ... existing fields ...
    pub cpu_config: CPUConfig,
}

// CPU-optimized linear layer
pub struct OptimizedLinear {
    weight: Arc<Array2<f32>>,
    bias: Arc<Array1<f32>>,
    cpu_config: CPUConfig,
}

impl OptimizedLinear {
    pub fn new(in_features: usize, out_features: usize, cpu_config: CPUConfig) -> Self {
        let weight = Arc::new(Array2::zeros((out_features, in_features)));
        let bias = Arc::new(Array1::zeros(out_features));
        OptimizedLinear { weight, bias, cpu_config }
    }

    pub fn forward(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        let (m, k) = input.dim();
        let n = self.weight.shape()[0];
        let mut output = Array2::zeros((m, n));

        let chunk_size = m / self.cpu_config.num_physical_cores + 1;
        
        output.axis_iter_mut(Axis(0))
            .into_par_iter()
            .with_max_len(chunk_size)
            .for_each(|mut out_row| {
                let weight = self.weight.view();
                let input_row = input.slice(s![out_row.index(), ..]);
                for (o, &b) in out_row.iter_mut().zip(self.bias.iter()) {
                    *o = input_row.dot(&weight.index_axis(Axis(0), o.index())) + b;
                }
            });

        output
    }
}

// CPU-optimized attention mechanism
pub struct OptimizedAttention {
    query: OptimizedLinear,
    key: OptimizedLinear,
    value: OptimizedLinear,
    num_attention_heads: usize,
    attention_head_size: usize,
    cpu_config: CPUConfig,
}

impl OptimizedAttention {
    pub fn new(config: &ViTConfig) -> Self {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = hidden_size / num_attention_heads;

        OptimizedAttention {
            query: OptimizedLinear::new(hidden_size, hidden_size, config.cpu_config.clone()),
            key: OptimizedLinear::new(hidden_size, hidden_size, config.cpu_config.clone()),
            value: OptimizedLinear::new(hidden_size, hidden_size, config.cpu_config.clone()),
            num_attention_heads,
            attention_head_size,
            cpu_config: config.cpu_config.clone(),
        }
    }

    pub fn forward(&self, hidden_states: &ArrayView2<f32>) -> Array2<f32> {
        let (batch_size, seq_len, _) = hidden_states.dim();
        
        let q = self.query.forward(hidden_states);
        let k = self.key.forward(hidden_states);
        let v = self.value.forward(hidden_states);

        let q = q.into_shape((batch_size, seq_len, self.num_attention_heads, self.attention_head_size)).unwrap();
        let k = k.into_shape((batch_size, seq_len, self.num_attention_heads, self.attention_head_size)).unwrap();
        let v = v.into_shape((batch_size, seq_len, self.num_attention_heads, self.attention_head_size)).unwrap();

        let q = q.permuted_axes([0, 2, 1, 3]);
        let k = k.permuted_axes([0, 2, 3, 1]);
        let v = v.permuted_axes([0, 2, 1, 3]);

        let attention_scores = q.dot(&k) / (self.attention_head_size as f32).sqrt();
        let attention_probs = softmax(&attention_scores, Axis(3));

        let context = attention_probs.dot(&v)
            .permuted_axes([0, 2, 1, 3])
            .into_shape((batch_size, seq_len, self.num_attention_heads * self.attention_head_size))
            .unwrap();

        context
    }
}

// CPU-optimized ViT layer
pub struct OptimizedViTLayer {
    attention: OptimizedAttention,
    intermediate: OptimizedLinear,
    output: OptimizedLinear,
    cpu_config: CPUConfig,
}

impl OptimizedViTLayer {
    pub fn new(config: &ViTConfig) -> Self {
        OptimizedViTLayer {
            attention: OptimizedAttention::new(config),
            intermediate: OptimizedLinear::new(config.hidden_size, config.intermediate_size, config.cpu_config.clone()),
            output: OptimizedLinear::new(config.intermediate_size, config.hidden_size, config.cpu_config.clone()),
            cpu_config: config.cpu_config.clone(),
        }
    }

    pub fn forward(&self, hidden_states: &ArrayView2<f32>) -> Array2<f32> {
        let attention_output = self.attention.forward(hidden_states);
        let intermediate_output = self.intermediate.forward(&attention_output.view());
        let layer_output = self.output.forward(&intermediate_output.view());
        layer_output
    }
}

// CPU-optimized ViT encoder
pub struct OptimizedViTEncoder {
    layers: Vec<OptimizedViTLayer>,
    cpu_config: CPUConfig,
}

impl OptimizedViTEncoder {
    pub fn new(config: &ViTConfig) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| OptimizedViTLayer::new(config))
            .collect();

        OptimizedViTEncoder {
            layers,
            cpu_config: config.cpu_config.clone(),
        }
    }

    pub fn forward(&self, hidden_states: &ArrayView2<f32>) -> Array2<f32> {
        let mut current_hidden_states = hidden_states.to_owned();

        for layer in &self.layers {
            current_hidden_states = layer.forward(&current_hidden_states.view());
        }

        current_hidden_states
    }
}

// Helper function for softmax
fn softmax(input: &Array4<f32>, axis: Axis) -> Array4<f32> {
    let max = input.map_axis(axis, |view| view.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    let exp = (input - &max.insert_axis(axis)).mapv(|a| a.exp());
    let sum = exp.sum_axis(axis);
    exp / &sum.insert_axis(axis)
}

// Function to set thread affinity
fn set_thread_affinity(cpu_config: &CPUConfig) {
    let core_ids = core_affinity::get_core_ids().unwrap();
    let num_cores = if cpu_config.use_logical_cores {
        core_ids.len()
    } else {
        cpu_config.num_physical_cores
    };

    let start_core = if let Some(node) = cpu_config.numa_node {
        node * cpu_config.num_physical_cores
    } else {
        0
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cores)
        .start_handler(move |id| {
            let core_id = core_ids[(start_core + id) % core_ids.len()];
            core_affinity::set_for_current(core_id);
        })
        .build_global()
        .unwrap();
}

// Main function to run the optimized ViT
pub fn run_optimized_vit(config: ViTConfig, input: Array2<f32>) -> Array2<f32> {
    set_thread_affinity(&config.cpu_config);

    let encoder = OptimizedViTEncoder::new(&config);
    encoder.forward(&input.view())
}
