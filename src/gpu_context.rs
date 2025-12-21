use crate::cuda_tensor::CublasHandle;
use crate::cuda_tensor::CudaMemoryPool;
use cublas_sys::cublasHandle_t;
use cust::prelude::Function;
use cust::prelude::Module;
use cust::stream::Stream;
use std::collections::HashMap;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

pub static GPU_CONTEXT: OnceLock<GpuContext> = OnceLock::new();

static KERNEL_CONTEXT: OnceLock<Mutex<KernelMap>> = OnceLock::new();

#[derive(Debug)]
pub struct GpuContext {
    pub context: Option<cust::context::Context>,
    pub module: Option<Module>,
    pub stream: Option<Stream>,
    pub pool: Option<CudaMemoryPool>,
    pub cublas_handle: CublasHandle,
}

pub struct KernelMap {
    kernel_map: HashMap<String, Arc<Function<'static>>>,
}

impl GpuContext {
    pub fn get_function(&self, fn_name: &str) -> Arc<Function<'_>> {
        let mut guard = KERNEL_CONTEXT
            .get_or_init(|| {
                Mutex::new(KernelMap {
                    kernel_map: HashMap::new(),
                })
            })
            .lock()
            .unwrap();

        if let Some(f) = guard.kernel_map.get(fn_name) {
            return Arc::clone(f);
        }

        let module = self.module.as_ref().expect("Module not found");

        match module.get_function(fn_name) {
            Ok(f) => {
                let f_static = unsafe { std::mem::transmute::<Function<'_>, Function<'static>>(f) };
                let shared_fn = Arc::new(f_static);

                guard
                    .kernel_map
                    .insert(fn_name.to_string(), Arc::clone(&shared_fn));
                shared_fn
            }
            Err(e) => {
                panic!("Error: {}, while getting function: {}", e, fn_name);
            }
        }
    }
}

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
