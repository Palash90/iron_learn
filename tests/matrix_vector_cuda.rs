use cublas_sys::*;
use cust::prelude::Module;
use cust::stream::Stream;
use cust::stream::StreamFlags;
use iron_learn::init_gpu;
use iron_learn::tensor::math::TensorMath;
use iron_learn::Tensor;
use iron_learn::{init_context, GpuTensor};
use std::ptr;

type TensorType = f32;

fn init() {
    match cust::quick_init() {
        Ok(context) => {
            eprintln!("✓ GPU initialization successful");
            let ptx = include_str!("../kernels/gpu_kernels.ptx");
            let module = Module::from_ptx(ptx, &[]).expect("CUDA module could not be initiated");

            let stream = match Stream::new(StreamFlags::NON_BLOCKING, None) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error creating stream: {}", e);
                    return;
                }
            };

            let mut handle: cublasHandle_t = ptr::null_mut();
            unsafe {
                let status = cublasCreate_v2(&mut handle);
                if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    eprintln!("Failed to create cuBLAS handle");
                    return;
                }
            };

            init_context("Iron Learn", 5, String::new(), 0.0, 0, true);
            init_gpu(Some(context), Some(module), Some(stream), Some(handle));
        }
        Err(e) => {
            eprintln!("⚠ GPU initialization failed: {}. Using CPU mode.", e);
            init_context("Iron Learn", 5, "".to_string(), 0.01, 1, false);
        }
    }
}

#[cfg(test)]
#[test]
pub fn test_add() {
    init();

    let m1 = GpuTensor::<TensorType>::new(vec![1], vec![1.0]).unwrap();
    let m2 = GpuTensor::new(vec![1], vec![3.0]).unwrap();
    let result = GpuTensor::new(vec![1], vec![4.0]).unwrap();
    let m3 = (m1 + m2).unwrap();

    println!("Result");
    m3.print_matrix();

    println!("Expected");
    result.print_matrix();

    assert_eq!(result, m3);

    let m1 = GpuTensor::<TensorType>::new(vec![1, 2], vec![1.0, 2.0]).unwrap();
    let m2 = GpuTensor::new(vec![1, 2], vec![3.0, 4.0]).unwrap();
    let result = GpuTensor::new(vec![1, 2], vec![4.0, 6.0]).unwrap();
    let m3 = (m1 + m2).unwrap();

    GpuTensor::<TensorType>::synchronize();

    println!("Result");
    m3.print_matrix();

    println!("Expected");
    result.print_matrix();

    assert_eq!(result, m3);
}

#[test]
pub fn test_mul_float() {
    init();

    let m1 = GpuTensor::<TensorType>::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m2 = GpuTensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let m3 = (m1 * m2).unwrap();
    let result = GpuTensor::new(vec![2, 2], vec![19.0, 22.0, 43.0, 50.0]).unwrap();

    println!("Result");
    m3.print_matrix();

    println!("expected");
    result.print_matrix();

    assert_eq!(result, m3);
}

#[test]
pub fn test_hadamard_float() {
    init();

    let m1 = GpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m2 = GpuTensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let m3 = m1.multiply(&m2).unwrap();
    let result = GpuTensor::new(vec![2, 2], vec![5.0, 12.0, 21.0, 32.0]).unwrap();

    println!("Result");
    m3.print_matrix();
    result.print_matrix();
}

#[test]
pub fn test_neg_float() {
    init();

    let m1 = GpuTensor::<TensorType>::new(vec![2, 2], vec![1.0, 2.0, -3.0, 4.0]).unwrap();
    let m2 = (-m1).unwrap();
    let result = GpuTensor::new(vec![2, 2], vec![-1.0, -2.0, 3.0, -4.0]).unwrap();

    println!("Result");
    m2.print_matrix();
    result.print_matrix();

    let m1 = GpuTensor::<TensorType>::new(vec![1, 4], vec![1.0, 2.0, -3.0, -4.0]).unwrap();
    let m2 = (-m1).unwrap();
    let result = GpuTensor::new(vec![1, 4], vec![-1.0, -2.0, 3.0, 4.0]).unwrap();
}

#[test]
pub fn test_scale_float() {
    init();

    let m1 = GpuTensor::<TensorType>::new(vec![2, 2], vec![1.0, 2.0, -3.0, 4.0]).unwrap();
    let m2 = m1.scale(2.0).unwrap();
    let result = GpuTensor::new(vec![2, 2], vec![2.0, 4.0, -6.0, 8.0]).unwrap();

    println!("Result");
    m2.print_matrix();
    result.print_matrix();

    assert_eq!(result, m2);

    let m1 = GpuTensor::<TensorType>::new(vec![1, 4], vec![1.0, 2.0, -3.0, -4.0]).unwrap();
    let m2 = m1.scale(3.0).unwrap();
    let result = GpuTensor::new(vec![1, 4], vec![3.0, 6.0, -9.0, -12.0]).unwrap();

    m2.print_matrix();
    result.print_matrix();

    assert_eq!(result, m2);
}

fn sigmoid(x: TensorType) -> TensorType {
    TensorType::exp(x) / (1.0 + TensorType::exp(x))
}

#[test]
pub fn test_element_op_float() {
    init();

    let m1 = GpuTensor::new(vec![2, 2], vec![1.0, 2.0, -3.0, 4.0]).unwrap();

    let m2 = m1.sin().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![
            TensorType::sin(1.0),
            TensorType::sin(2.0),
            TensorType::sin(-3.0),
            TensorType::sin(4.0),
        ],
    )
    .unwrap();

    result.print_matrix();
    m2.print_matrix();
    assert_eq!(result, m2);

    let m2 = m1.cos().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![
            TensorType::cos(1.0),
            TensorType::cos(2.0),
            TensorType::cos(-3.0),
            TensorType::cos(4.0),
        ],
    )
    .unwrap();
    result.print_matrix();
    m2.print_matrix();
    assert_eq!(result, m2);

    let m2 = m1.tan().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![
            TensorType::tan(1.0),
            TensorType::tan(2.0),
            TensorType::tan(-3.0),
            TensorType::tan(4.0),
        ],
    )
    .unwrap();
    result.print_matrix();
    m2.print_matrix();
    assert_eq!(result, m2);

    let m2 = m1.tanh().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![
            TensorType::tanh(1.0),
            TensorType::tanh(2.0),
            TensorType::tanh(-3.0),
            TensorType::tanh(4.0),
        ],
    )
    .unwrap();
    result.print_matrix();
    m2.print_matrix();
    assert_eq!(result, m2);

    let m2 = m1.log().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![
            TensorType::log10(1.0),
            TensorType::log10(2.0),
            TensorType::log10(-3.0),
            TensorType::log10(4.0),
        ],
    )
    .unwrap();
    result.print_matrix();
    m2.print_matrix();
    assert_eq!(result, m2);

    let m2 = m1.ln().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![
            TensorType::ln(1.0),
            TensorType::ln(2.0),
            TensorType::ln(-3.0),
            TensorType::ln(4.0),
        ],
    )
    .unwrap();
    assert_eq!(result, m2);

    let m2 = m1.exp().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![
            TensorType::exp(1.0),
            TensorType::exp(2.0),
            TensorType::exp(-3.0),
            TensorType::exp(4.0),
        ],
    )
    .unwrap();
    assert_eq!(result, m2);

    let m2 = m1.sigmoid().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![sigmoid(1.0), sigmoid(2.0), sigmoid(-3.0), sigmoid(4.0)],
    )
    .unwrap();
    assert_eq!(result, m2);
}

#[test]
pub fn test_transpose() {
    init();

    let m = GpuTensor::<TensorType>::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = GpuTensor::new(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]).unwrap();

    println!("Original Matrix");
    m.print_matrix();
    let m_t = m.t().unwrap();
    println!("Transposed");
    m_t.print_matrix();

    println!("Expected");
    result.print_matrix();

     assert_eq!(result, m.t().unwrap());

    let m = GpuTensor::<TensorType>::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = GpuTensor::new(vec![3, 2], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();

     assert_eq!(result, m.t().unwrap());

    let m = GpuTensor::<TensorType>::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = GpuTensor::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

      assert_eq!(result, m.t().unwrap());

    let m = GpuTensor::<TensorType>::new(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let result = GpuTensor::new(
        vec![3, 3],
        vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
    )
    .unwrap();

        assert_eq!(result, m.t().unwrap());

    let m = GpuTensor::<TensorType>::new(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
    let result = GpuTensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();

      assert_eq!(result, m.t().unwrap());
}
