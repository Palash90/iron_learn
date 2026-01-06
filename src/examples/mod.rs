mod contexts;
pub mod init;
mod read_file;
mod regression;
pub mod types;

pub mod bigram;
mod neural_net_runner;
pub mod trigram;
pub mod five_gram;

pub use bigram::run_bigram_generator;
pub use neural_net_runner::run_neural_net;
pub use regression::run_linear;
pub use regression::run_logistic;
pub use trigram::run_trigram_generator;
pub use five_gram::run_five_gram_generator;
