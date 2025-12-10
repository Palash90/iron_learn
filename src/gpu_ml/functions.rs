use cust::error::CudaResult;
use cust::memory::DevicePointer;
use cust::module::Module;
use cust::stream::Stream;

/// Lightweight wrappers around common CUDA kernel launches used in the crate.
/// Each wrapper fetches the Function from the Module and issues the launch
/// inside a single unsafe block so call sites remain concise and the unsafe
/// surface is localised.

pub fn gemv_row_major(
    module: &Module,
    stream: &Stream,
    d_x: DevicePointer<f64>,
    d_w: DevicePointer<f64>,
    d_out: DevicePointer<f64>,
    rows: i32,
    cols: i32,
) -> CudaResult<()> {
    let func = module.get_function("gemvRowMajor")?;
    let block = (1024, 1, 1);
    let grid_rows = (((rows as u32) + 2047) / 2048, 1, 1);
    unsafe {
        cust::launch!(
            func<<<grid_rows, block, 0, stream>>>(
                d_x,
                d_w,
                d_out,
                rows,
                cols
            )
        )?;
    }
    Ok(())
}

pub fn sigmoid_kernel(
    module: &Module,
    stream: &Stream,
    d_in: DevicePointer<f64>,
    d_out: DevicePointer<f64>,
    len: i32,
) -> CudaResult<()> {
    let func = module.get_function("sigmoidKernel")?;
    let block = (256, 1, 1);
    let grid = (((len as u32) + 255) / 256, 1, 1);
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(d_in, d_out, len))?;
    }
    Ok(())
}

pub fn threshold_kernel(
    module: &Module,
    stream: &Stream,
    d_in: DevicePointer<f64>,
    d_out: DevicePointer<f64>,
    len: i32,
) -> CudaResult<()> {
    let func = module.get_function("thresholdKernel")?;
    let block = (256, 1, 1);
    let grid = (((len as u32) + 255) / 256, 1, 1);
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(d_in, d_out, len))?;
    }
    Ok(())
}

pub fn vector_sub(
    module: &Module,
    stream: &Stream,
    a: DevicePointer<f64>,
    b: DevicePointer<f64>,
    out: DevicePointer<f64>,
    len: i32,
    stride: i32,
) -> CudaResult<()> {
    let func = module.get_function("vector_add")?; // implemented to perform a-b for this project
    let block = (256, 1, 1);
    let grid = (((len as u32) + 255) / 256, 1, 1);
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(a, b, out, len, stride))?;
    }
    Ok(())
}

pub fn grad_gemv_xt(
    module: &Module,
    stream: &Stream,
    x: DevicePointer<f64>,
    loss: DevicePointer<f64>,
    out_grad: DevicePointer<f64>,
    rows: i32,
    cols: i32,
) -> CudaResult<()> {
    let func = module.get_function("gradGemvXT")?;
    let block = (256, 1, 1);
    let grid = (((cols as u32) + 2047) / 2048, 1, 1);
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(x, loss, out_grad, rows, cols))?;
    }
    Ok(())
}

pub fn scale_vector(
    module: &Module,
    stream: &Stream,
    vec_ptr: DevicePointer<f64>,
    scale: f64,
    len: i32,
) -> CudaResult<()> {
    let func = module.get_function("scaleVector")?;
    let block = (256, 1, 1);
    let grid = (((len as u32) + 255) / 256, 1, 1);
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(vec_ptr, scale, len))?;
    }
    Ok(())
}

pub fn update_weights(
    module: &Module,
    stream: &Stream,
    w_ptr: DevicePointer<f64>,
    grad_ptr: DevicePointer<f64>,
    len: i32,
) -> CudaResult<()> {
    let func = module.get_function("updateWeights")?;
    let block = (256, 1, 1);
    let grid = (((len as u32) + 255) / 256, 1, 1);
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(w_ptr, grad_ptr, len))?;
    }
    Ok(())
}

pub fn matrix_mul(
    module: &Module,
    stream: &Stream,
    a: DevicePointer<f64>,
    b: DevicePointer<f64>,
    out: DevicePointer<f64>,
    m: i32,
    n: i32,
    k: i32,
) -> CudaResult<()> {
    let func = module.get_function("matrixMul")?;
    let block = (16, 16, 1);
    let grid_x = (((n as u32) + 15) / 16, ((m as u32) + 15) / 16, 1);
    unsafe {
        cust::launch!(func<<<grid_x, block, 0, stream>>>(a, b, out, m, n, k))?;
    }
    Ok(())
}

pub fn relu_kernel(
    module: &Module,
    stream: &Stream,
    in_ptr: DevicePointer<f64>,
    out_ptr: DevicePointer<f64>,
    len: i32,
) -> CudaResult<()> {
    let func = module.get_function("reluKernel")?;
    let block = (256, 1, 1);
    let grid = (((len as u32) + 255) / 256, 1, 1);
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(in_ptr, out_ptr, len))?;
    }
    Ok(())
}

pub fn relu_derivative(
    module: &Module,
    stream: &Stream,
    in_ptr: DevicePointer<f64>,
    out_ptr: DevicePointer<f64>,
    len: i32,
) -> CudaResult<()> {
    let func = module.get_function("reluDerivativeKernel")?;
    let block = (256, 1, 1);
    let grid = (((len as u32) + 255) / 256, 1, 1);
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(in_ptr, out_ptr, len))?;
    }
    Ok(())
}

pub fn transpose_naive(
    module: &Module,
    stream: &Stream,
    in_ptr: DevicePointer<f64>,
    out_ptr: DevicePointer<f64>,
    m: i32,
    n: i32,
) -> CudaResult<()> {
    let func = module.get_function("transpose_naive")?;
    let block = (16, 16, 1);
    let grid = (((n as u32) + 15) / 16, ((m as u32) + 15) / 16, 1);
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(in_ptr, out_ptr, m, n))?;
    }
    Ok(())
}

pub fn hadamard_prod(
    module: &Module,
    stream: &Stream,
    a: DevicePointer<f64>,
    b: DevicePointer<f64>,
    out: DevicePointer<f64>,
    len: i32,
) -> CudaResult<()> {
    let func = module.get_function("hadamardProd")?;
    let block = (256, 1, 1);
    let grid = (((len as u32) + 255) / 256, 1, 1);
    unsafe {
        cust::launch!(func<<<grid, block, 0, stream>>>(a, b, out, len))?;
    }
    Ok(())
}
