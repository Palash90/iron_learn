#![cfg(feature = "cuda")]

use cublas_sys::*;
/// Wrapper around a raw cuBLAS handle used by the GPU context.
///
/// This type makes the raw `cublasHandle_t` available to GPU-backed
/// tensor operations and is marked `Send`/`Sync` via unsafe impls so it
/// can be stored in the global `GpuContext` safely.
#[derive(Debug)]
pub struct CublasHandle {
    pub handle: cublasHandle_t,
}

unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}
