use crate::cuda_tensor::CublasHandle;
use crate::cuda_tensor::CudaMemoryPool;
use cublas_sys::cublasCreate_v2;
use cublas_sys::cublasHandle_t;
use cublas_sys::cublasStatus_t;

use cust::prelude::Function;
use cust::prelude::Module;
use cust::stream::Stream;
use std::ptr;
use std::sync::OnceLock;

use cust::stream::StreamFlags;

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
/// Calling this function sets the `GPU_CONTEXT` `OnceLock` once; subsequent
/// calls will print a warning if initialization was already performed.
pub fn init_gpu() -> Result<(), String> {
    match cust::quick_init() {
        Ok(context) => {
            println!("✓ GPU initialization successful");

            let mut handle: cublasHandle_t = ptr::null_mut();
            unsafe {
                let status = cublasCreate_v2(&mut handle);
                if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err("Failed to create cuBLAS handle".to_string());
                }
            };

            println!("✓ CUBLAS initialization successful");

            let ptx = include_str!("../kernels/gpu_kernels.ptx");
            let module = Module::from_ptx(ptx, &[]).expect("CUDA module could not be initiated");

            let stream = match Stream::new(StreamFlags::NON_BLOCKING, None) {
                Ok(s) => s,
                Err(_) => {
                    return Err("Error creating stream".to_string());
                }
            };

            let pool = CudaMemoryPool::get_mem_pool();

            let ctx = GpuContext {
                context: Some(context),
                module: Some(module),
                stream: Some(stream),
                pool: Some(pool),
                cublas_handle: CublasHandle { handle },
            };

            match GPU_CONTEXT.set(ctx) {
                Ok(_) => (),
                Err(_) => eprintln!("GpuContext has already been initialized!"),
            };
        }
        Err(e) => {
            return Err(format!("Error: {}", e));
        }
    }

    Ok(())
}
