use super::functions::*;
use cust::error::CudaResult;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::Stream;

/// ReLU activation function
pub fn relu_activation(
    d_input: DevicePointer<f64>,
    d_output: DevicePointer<f64>,
    size: i32,
    module: &Module,
    stream: &Stream,
) -> CudaResult<()> {
    use super::functions::relu_kernel;
    relu_kernel(module, stream, d_input, d_output, size)
}

/// ReLU derivative
pub fn relu_derivative(
    d_z: DevicePointer<f64>,
    d_deriv: DevicePointer<f64>,
    size: i32,
    module: &Module,
    stream: &Stream,
) -> CudaResult<()> {
    use super::functions::relu_derivative as relu_deriv_helper;
    relu_deriv_helper(module, stream, d_z, d_deriv, size)
}

/// Sigmoid activation function
pub fn sigmoid_activation(
    d_input: DevicePointer<f64>,
    d_output: DevicePointer<f64>,
    size: i32,
    module: &Module,
    stream: &Stream,
) -> CudaResult<()> {
    use super::functions::sigmoid_kernel;
    sigmoid_kernel(module, stream, d_input, d_output, size)
}

/// Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
/// Since sigmoid is already computed, we use: deriv = sigmoid(z) * (1 - sigmoid(z))
/// For now, we approximate using a kernel or compute on device
pub fn sigmoid_derivative(
    d_z: DevicePointer<f64>,
    d_deriv: DevicePointer<f64>,
    size: i32,
    module: &Module,
    stream: &Stream,
) -> CudaResult<()> {
    // Placeholder: apply sigmoid then multiply by (1 - sigmoid)
    // This requires two passes or a specialized kernel
    sigmoid_activation(d_z, d_deriv, size, module, stream)?;
    // TODO: Implement proper sigmoid derivative kernel
    Ok(())
}

/// Linear (identity) activation - no-op
pub fn linear_activation(
    d_input: DevicePointer<f64>,
    d_output: DevicePointer<f64>,
    size: i32,
    _module: &Module,
    _stream: &Stream,
) -> CudaResult<()> {
    // Linear activation is identity, but GPU transfer is not ideal here
    // In practice, we'd skip this or use a copy kernel
    // For now, we assume the input buffer is already the output
    Ok(())
}

/// Linear derivative (always 1)
pub fn linear_derivative(
    _d_z: DevicePointer<f64>,
    _d_deriv: DevicePointer<f64>,
    _size: i32,
    _module: &Module,
    _stream: &Stream,
) -> CudaResult<()> {
    // Linear derivative is 1, so we skip computation
    // In practice, we'd fill d_deriv with 1.0
    Ok(())
}

/// Tanh activation function
pub fn tanh_activation(
    d_input: DevicePointer<f64>,
    d_output: DevicePointer<f64>,
    size: i32,
    module: &Module,
    stream: &Stream,
) -> CudaResult<()> {
    // Tanh kernel: apply element-wise tanh
    let func = module.get_function("tanhKernel")?;
    let block = (256, 1, 1);
    let grid_x = ((size as u32 + 255) / 256, 1, 1);

    unsafe {
        cust::launch!(func<<<grid_x, block, 0, stream>>>(d_input, d_output, size))?;
    }

    Ok(())
}

/// Tanh derivative: 1 - tanh^2(x)
pub fn tanh_derivative(
    d_z: DevicePointer<f64>,
    d_deriv: DevicePointer<f64>,
    size: i32,
    module: &Module,
    stream: &Stream,
) -> CudaResult<()> {
    // Tanh derivative kernel: compute 1 - tanh^2(z)
    let func = module.get_function("tanhDerivativeKernel")?;
    let block = (256, 1, 1);
    let grid_x = ((size as u32 + 255) / 256, 1, 1);

    unsafe {
        cust::launch!(func<<<grid_x, block, 0, stream>>>(d_z, d_deriv, size))?;
    }

    Ok(())
}
