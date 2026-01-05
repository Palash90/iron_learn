#![cfg(feature = "cuda")]
#[cfg(test)]
mod cuda_tests {

    use iron_learn::tensor::math::TensorMath;
    use iron_learn::GpuTensor;
    use iron_learn::Tensor;

    use iron_learn::init_gpu;

    type TensorType = f32;

    #[test]
    pub fn test_cuda_add() {
        let _ = init_gpu();

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
    pub fn test_cuda_mul_float() {
        let _ = init_gpu();

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
    pub fn test_cuda_hadamard_float() {
        let _ = init_gpu();

        let m1 = GpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let m2 = GpuTensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let m3 = m1.matmul(&m2).unwrap();
        let result = GpuTensor::new(vec![2, 2], vec![5.0, 12.0, 21.0, 32.0]).unwrap();

        println!("Result");
        m3.print_matrix();
        result.print_matrix();
    }

    #[test]
    pub fn test_cuda_neg_float() {
        let _ = init_gpu();

        let m1 = GpuTensor::<TensorType>::new(vec![2, 2], vec![1.0, 2.0, -3.0, 4.0]).unwrap();
        let m2 = (-m1).unwrap();
        let result = GpuTensor::new(vec![2, 2], vec![-1.0, -2.0, 3.0, -4.0]).unwrap();

        println!("Result");
        m2.print_matrix();
        result.print_matrix();

        let m1 = GpuTensor::<TensorType>::new(vec![1, 4], vec![1.0, 2.0, -3.0, -4.0]).unwrap();
        let m2 = (-m1).unwrap();
        let result = GpuTensor::new(vec![1, 4], vec![-1.0, -2.0, 3.0, 4.0]).unwrap();

        println!("Result");
        m2.print_matrix();
        result.print_matrix();

        assert_eq!(result, m2);
    }

    #[test]
    fn test_cuda_matmul_identity() {
        let _ = init_gpu();

        let a = GpuTensor::<TensorType>::new(vec![2, 2], vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32])
            .unwrap();
        let identity =
            GpuTensor::<TensorType>::new(vec![2, 2], vec![1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32])
                .unwrap();

        let result = a.matmul(&identity).unwrap();
        assert_eq!(result.get_data(), a.get_data());
    }

    #[test]
    fn test_cuda_matmul_vector_dot_product() {
        let _ = init_gpu();

        // (1x3) * (3x1) = (1x1)
        let a = GpuTensor::<TensorType>::new(vec![1, 3], vec![1.0_f32, 2.0_f32, 3.0_f32]).unwrap();
        let b = GpuTensor::<TensorType>::new(vec![3, 1], vec![4.0_f32, 5.0_f32, 6.0_f32]).unwrap();

        let result = a.matmul(&b).unwrap();
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result.get_data(), vec![32.0_f32]);
        assert_eq!(result.get_shape(), &vec![1, 1]);
    }

    #[test]
    pub fn test_cuda_scale_float() {
        let _ = init_gpu();

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
    pub fn test_cuda_element_op_float() {
        let _ = init_gpu();

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

        println!("Sin check");
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

        println!("Cos check");
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

        println!("Tan check");
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

        println!("Tanh check");
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

        println!("Log10 check");
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

        println!("ln check");
        result.print_matrix();
        m2.print_matrix();
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
        println!("exp check");
        result.print_matrix();
        m2.print_matrix();
        assert_eq!(result, m2);

        let m2 = m1.sigmoid().unwrap();
        let result = GpuTensor::new(
            vec![2, 2],
            vec![sigmoid(1.0), sigmoid(2.0), sigmoid(-3.0), sigmoid(4.0)],
        )
        .unwrap();

        println!("Sigmoid check");
        result.print_matrix();
        m2.print_matrix();
        assert_eq!(result, m2);
    }

    #[test]
    pub fn test_cuda_transpose() {
        let _ = init_gpu();

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

        let m =
            GpuTensor::<TensorType>::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = GpuTensor::new(vec![3, 2], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        let m_t = m.t().unwrap();

        println!("Transposed");
        result.print_matrix();
        m_t.print_matrix();

        assert_eq!(result, m_t);

        let m = GpuTensor::<TensorType>::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = GpuTensor::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let m_t = m.t().unwrap();

        assert_eq!(result, m_t);

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
}
