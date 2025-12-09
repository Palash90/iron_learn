use iron_learn::{init_context, GpuTensor};
fn init() {
    match cust::quick_init() {
        Ok(context) => {
            eprintln!("✓ GPU initialization successful");
            init_context(
                "Iron Learn",
                5,
                "".to_string(),
                0.01,
                1,
                true,
                Some(context),
            );
        }
        Err(e) => {
            eprintln!("⚠ GPU initialization failed: {}. Using CPU mode.", e);
            init_context("Iron Learn", 5, "".to_string(), 0.01, 1, false, None);
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
    let m3 = m1.hadamard(&m2).unwrap();
    let result = GpuTensor::new(vec![2, 2], vec![5.0, 12.0, 21.0, 32.0]).unwrap();

    m3.print_matrix();
    result.print_matrix();

    assert_eq!(result, m3);
}

#[test]
pub fn exp_float() {
    init();

    let m1 = GpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m2 = m1.exp().unwrap();
    let result = GpuTensor::new(vec![2, 2], vec![f64::exp(1.0), f64::exp(2.0), f64::exp(3.0), f64::exp(4.0)]).unwrap();

    m2.print_matrix();
    result.print_matrix();

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
