use cublas_sys::*;
#[derive(Debug)]
pub struct CublasHandle {
    pub handle: cublasHandle_t,
}

unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}
