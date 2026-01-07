use crate::numeric::FloatingPoint;
use clap::Parser;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Dataset container for single-precision (f32) examples.
///
/// Holds training and test matrices as flattened vectors along with
/// their dimensions. This is the primary structure used by the
/// CLI runners and model loaders for f32-based datasets.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(bound = "")]
pub struct Data<D>
where
    D: FloatingPoint,
{
    pub m: u32,
    pub n: u32,
    pub x: Vec<D>,
    pub y: Vec<D>,
    pub m_test: u32,
    pub x_test: Vec<D>,
    pub y_test: Vec<D>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum ExampleMode {
    /// Linear Regression
    Linear,
    /// Logistic Regression
    Logistic,
    /// Neural Network - Generic
    NeuralNet,
    /// Neural Network - XOR
    XorNeuralNet,
    /// Neural Network - Image
    ImageNeuralNet,
    /// N Gram Generator
    NGram,
}

#[derive(Parser)]
#[command(name = "Iron Learn")]
#[command(name = "A Rust Machine Learning Library")]
pub struct IronLearnArgs {
    #[arg(long, short, default_value = "neural_net")]
    pub name: String,

    #[arg(long, short, default_value = "false")]
    pub cpu: bool,

    #[arg(long, short = 'x', default_value = "linear")]
    pub mode: ExampleMode,

    #[arg(long, short, default_value = "false")]
    pub restore: bool,

    #[arg(long, short, default_value = "0.01")]
    pub lr: f64,

    #[arg(long, short, default_value = "10001")]
    pub epochs: u32,

    #[arg(long, short, default_value = "data/neural_net.json")]
    pub data_file: String,

    #[arg(long, short, default_value = "false")]
    pub adjust_lr: bool,

    #[arg(long, short, default_value = "4")]
    pub internal_layers: u32,

    #[arg(long, short, default_value = "1000")]
    pub monitor_interval: usize,

    #[arg(long, short, default_value = "0")]
    pub sleep_time: u64,

    #[arg(long, short, default_value = "model.json")]
    pub parameters_path: String,

    #[arg(long, short = 'D', default_value = "Normal")]
    pub distribution: String,

    #[arg(long, default_value = "false")]
    pub predict_only: bool,

    #[arg(long, default_value = "0")]
    pub reproduce: u32,

    #[arg(long, short, default_value = "0")]
    pub temparature: f64,

    #[arg(long, default_value = "true")]
    pub no_repeat: bool,

    #[arg(long, default_value = "")]
    pub n_gram_seed: String,

    #[arg(long, default_value = "5")]
    pub n_gram_size: u8,
}
