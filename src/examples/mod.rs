mod contexts;
pub mod init;
mod read_file;
mod regression;
pub mod types;

pub mod n_gram;
mod neural_net_runner;

pub use n_gram::run_n_gram_generator;
pub use neural_net_runner::run_neural_net;
pub use regression::run_linear;
pub use regression::run_logistic;

use crate::nn::loss_functions::LossFunctionType;
use crate::nn::LayerType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct NetworkConfig {
    loss_function: LossFunctionType,
    layers: Vec<(u32, u32, LayerType, String)>,
}
