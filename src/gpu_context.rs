#![cfg(feature = "cuda")]

use crate::cuda_tensor::CublasHandle;
use crate::cuda_tensor::CudaMemoryPool;
use cublas_sys::cublasHandle_t;
use cust::prelude::Function;
use cust::prelude::Module;
use cust::stream::Stream;
use std::ptr;
use std::sync::OnceLock;

/// Global GPU context singleton used throughout the crate.
///
/// This `OnceLock` holds the process-wide `GpuContext` and should be
/// initialized exactly once by calling `init_gpu(...)` prior to any GPU
/// operations. Consumers may read from this static to access the CUDA
/// `Module`, `Stream`, memory pool and cuBLAS handle.
pub static GPU_CONTEXT: OnceLock<GpuContext> = OnceLock::new();

#[derive(Debug)]
/// Process-wide GPU context containing CUDA resources used by the crate.
///
/// `GpuContext` bundles optional CUDA constructs (context, loaded module,
/// stream) and resource managers (memory pool and cuBLAS handle). Fields are
/// public for consumer access through the `GPU_CONTEXT` singleton.
pub struct GpuContext {
    pub context: Option<cust::context::Context>,
    pub module: Option<Module>,
    pub stream: Option<Stream>,
    pub pool: Option<CudaMemoryPool>,
    pub cublas_handle: CublasHandle,
}

impl GpuContext {
    /// Lookup a CUDA kernel/function by name from the loaded module.
    ///
    /// # Panics
    /// - If the `Module` has not been loaded into this context.
    pub fn get_function(&self, fn_name: &str) -> Function<'_> {
        let module = self.module.as_ref().expect("Module not found");

        match module.get_function(fn_name) {
            Ok(f) => f,
            Err(e) => {
                panic!("Error: {}, while getting function: {}", e, fn_name);
            }
        }
    }
}

/// Initialize the global GPU context singleton used by the crate.
///
/// - `context`: optionally provide an owned CUDA `Context`.
/// - `module`: optionally provide a loaded CUDA `Module` containing kernels.
/// - `stream`: optionally provide a CUDA `Stream` to use for kernel launches.
/// - `cublas_handle`: optionally provide a raw cuBLAS handle (`cublasHandle_t`).
///
/// Calling this function sets the `GPU_CONTEXT` `OnceLock` once; subsequent
/// calls will print a warning if initialization was already performed.
pub fn init_gpu(
    context: Option<cust::context::Context>,
    module: Option<Module>,
    stream: Option<Stream>,
    cublas_handle: Option<cublasHandle_t>,
) {
    let pool = match context {
        Some(_) => Some(CudaMemoryPool::get_mem_pool()),
        None => None,
    };

    let handle = match cublas_handle {
        Some(t) => t,
        _ => ptr::null_mut(),
    };

    let ctx = GpuContext {
        context,
        module,
        stream,
        pool,
        cublas_handle: CublasHandle { handle },
    };

    match GPU_CONTEXT.set(ctx) {
        Ok(_) => (),
        Err(_) => eprintln!("GpuContext has already been initialized!"),
    };
}
