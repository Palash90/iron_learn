mod build_network;
pub mod contexts;
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
pub mod transformer;