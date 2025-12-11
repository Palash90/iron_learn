use cust::prelude::Module;
use cust::stream::Stream;
use cust::stream::StreamFlags;
use iron_learn::{init_context, GpuTensor};
use iron_learn::Tensor;
use iron_learn::tensor::math::TensorMath;

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

            init_context(
                "Iron Learn",
                5,
                String::new(),
                0.0,
                0,
                true,
                Some(context),
                Some(module),
                Some(stream),
            );
        }
        Err(e) => {
            eprintln!("⚠ GPU initialization failed: {}. Using CPU mode.", e);
            init_context(
                "Iron Learn",
                5,
                "".to_string(),
                0.01,
                1,
                false,
                None,
                None,
                None,
            );
        }
    }
}

#[cfg(test)]
#[test]
pub fn add_f64() {
    init();

    let m1 = GpuTensor::<f64>::new(vec![1], vec![1.0]).unwrap();
    let m2 = GpuTensor::new(vec![1], vec![3.0]).unwrap();
    let result = GpuTensor::new(vec![1], vec![4.0]).unwrap();
    let m3 = (m1 + m2).unwrap();

    m3.print_matrix();
    result.print_matrix();

    assert_eq!(result, m3);

    let m1 = GpuTensor::new(vec![1, 2], vec![1.0, 2.0]).unwrap();
    let m2 = GpuTensor::new(vec![1, 2], vec![3.0, 4.0]).unwrap();
    let result = GpuTensor::new(vec![1, 2], vec![4.0, 6.0]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[cfg(test)]
#[test]
pub fn add_f64_2() {
    init();

    let m1 = GpuTensor::<f64>::new(vec![1], vec![1.0]).unwrap();
    let m2 = GpuTensor::new(vec![1], vec![3.0]).unwrap();
    let result = GpuTensor::new(vec![1], vec![4.0]).unwrap();
    let m3 = (m1 + m2).unwrap();

    m3.print_matrix();
    result.print_matrix();

    assert_eq!(result, m3);

    let m1 = GpuTensor::new(vec![1, 2], vec![1.0, 2.0]).unwrap();
    let m2 = GpuTensor::new(vec![1, 2], vec![3.0, 4.0]).unwrap();
    let result = GpuTensor::new(vec![1, 2], vec![4.0, 6.0]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
pub fn mul_float() {
    init();

    let m1 = GpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m2 = GpuTensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let m3 = (m1 * m2).unwrap();
    let result = GpuTensor::new(vec![2, 2], vec![19.0, 22.0, 43.0, 50.0]).unwrap();

    m3.print_matrix();
    result.print_matrix();

    assert_eq!(result, m3);
}

#[test]
pub fn hadamard_float() {
    init();

    let m1 = GpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m2 = GpuTensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let m3 = m1.multiply(&m2).unwrap();
    let result = GpuTensor::new(vec![2, 2], vec![5.0, 12.0, 21.0, 32.0]).unwrap();

    m3.print_matrix();
    result.print_matrix();

    assert_eq!(result, m3);
}

#[test]
pub fn neg_float() {
    init();

    let m1 = GpuTensor::new(vec![2, 2], vec![1.0, 2.0, -3.0, 4.0]).unwrap();
    let m2 = (-m1).unwrap();
    let result = GpuTensor::new(vec![2, 2], vec![-1.0, -2.0, 3.0, -4.0]).unwrap();

    m2.print_matrix();
    result.print_matrix();

    assert_eq!(result, m2);

    let m1 = GpuTensor::new(vec![1, 4], vec![1.0, 2.0, -3.0, -4.0]).unwrap();
    let m2 = (-m1).unwrap();
    let result = GpuTensor::new(vec![1, 4], vec![-1.0, -2.0, 3.0, 4.0]).unwrap();

    assert_eq!(result, m2);
}

#[test]
pub fn scale_float() {
    init();

    let m1 = GpuTensor::new(vec![2, 2], vec![1.0, 2.0, -3.0, 4.0]).unwrap();
    let m2 = m1.scale(2.0).unwrap();
    let result = GpuTensor::new(vec![2, 2], vec![2.0, 4.0, -6.0, 8.0]).unwrap();

    m2.print_matrix();
    result.print_matrix();

    assert_eq!(result, m2);

    let m1 = GpuTensor::new(vec![1, 4], vec![1.0, 2.0, -3.0, -4.0]).unwrap();
    let m2 = m1.scale(3.0).unwrap();
    let result = GpuTensor::new(vec![1, 4], vec![3.0, 6.0, -9.0, -12.0]).unwrap();

    m2.print_matrix();
    result.print_matrix();

    assert_eq!(result, m2);
}

fn sigmoid(x: f64) -> f64 {
    f64::exp(x) / (1.0 + f64::exp(x))
}

#[test]
pub fn sin_float() {
    init();

    let m1 = GpuTensor::new(vec![2, 2], vec![1.0, 2.0, -3.0, 4.0]).unwrap();

    let m2 = m1.sin().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![f64::sin(1.0), f64::sin(2.0), f64::sin(-3.0), f64::sin(4.0)],
    )
    .unwrap();
    assert_eq!(result, m2);

    let m2 = m1.cos().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![f64::cos(1.0), f64::cos(2.0), f64::cos(-3.0), f64::cos(4.0)],
    )
    .unwrap();
    assert_eq!(result, m2);

    let m2 = m1.tan().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![f64::tan(1.0), f64::tan(2.0), f64::tan(-3.0), f64::tan(4.0)],
    )
    .unwrap();
    assert_eq!(result, m2);

    let m2 = m1.tanh().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![
            f64::tanh(1.0),
            f64::tanh(2.0),
            f64::tanh(-3.0),
            f64::tanh(4.0),
        ],
    )
    .unwrap();
    assert_eq!(result, m2);

    let m2 = m1.log().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![
            f64::log10(1.0),
            f64::log10(2.0),
            f64::log10(-3.0),
            f64::log10(4.0),
        ],
    )
    .unwrap();
    assert_eq!(result, m2);

    let m2 = m1.ln().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![f64::ln(1.0), f64::ln(2.0), f64::ln(-3.0), f64::ln(4.0)],
    )
    .unwrap();
    assert_eq!(result, m2);

    let m2 = m1.exp().unwrap();
    let result = GpuTensor::new(
        vec![2, 2],
        vec![f64::exp(1.0), f64::exp(2.0), f64::exp(-3.0), f64::exp(4.0)],
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
pub fn transpose() {
    init();

    let m = GpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = GpuTensor::new(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]).unwrap();

    println!("Original Matrix");
    m.print_matrix();
    let m_t = m.t().unwrap();
    println!("Transposed");
    m_t.print_matrix();

    println!("Expected");
    result.print_matrix();

    assert_eq!(result, m.t().unwrap());

    let m = GpuTensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = GpuTensor::new(vec![3, 2], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();

    assert_eq!(result, m.t().unwrap());

    let m = GpuTensor::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = GpuTensor::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    assert_eq!(result, m.t().unwrap());

    let m = GpuTensor::new(
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

    let m = GpuTensor::new(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap();
    let result = GpuTensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();

    assert_eq!(result, m.t().unwrap());
}
