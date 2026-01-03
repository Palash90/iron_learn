mod contexts;
pub mod init;
mod read_file;
mod types;
mod regression;

mod neural_net_runner;

pub use neural_net_runner::run_neural_net;
pub use regression::run_linear;
pub use regression::run_logistic;



