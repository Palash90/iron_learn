mod contexts;
pub mod init;
mod read_file;
mod regression;
pub mod types;

pub mod bigram;
mod neural_net_runner;
pub mod trigram;

pub use bigram::run_bigram_generator;
pub use neural_net_runner::run_neural_net;
pub use regression::run_linear;
pub use regression::run_logistic;
pub use trigram::run_trigram_generator;
