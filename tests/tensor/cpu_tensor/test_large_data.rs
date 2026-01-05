#[cfg(test)]
mod tests {
    use iron_learn::CpuTensor;
    use iron_learn::Tensor;

    // Helper to simplify tensor creation in tests
    fn new_cpu_tensor(shape: Vec<u32>, data: Vec<f32>) -> CpuTensor<f32> {
        CpuTensor::<f32>::new(shape, data).unwrap()
    }

    #[test]
    fn test_div_large_data() {
        // Verifies there are no iterator bottlenecks or index out of bounds
        let size = 10_000;
        let data1 = vec![1.0; size];
        let data2 = vec![2.0; size];
        let t1 = new_cpu_tensor(vec![size as u32], data1);
        let t2 = new_cpu_tensor(vec![size as u32], data2);

        let result = t1.div(&t2).unwrap();
        let data = result.get_data();

        println!(
            "Count of first tensor 1s: {}",
            t1.get_data().iter().filter(|x| **x == 1.0).count()
        );
        println!(
            "Count of second tensor 2s: {}",
            t2.get_data().iter().filter(|x| **x == 2.0).count()
        );
        println!(
            "Count of result with 0.5: {}",
            data.iter().filter(|x| **x == 0.5).count()
        );

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x == 0.5));
    }

    #[test]
    fn test_hadamard_large_data() {
        // Verifies there are no iterator bottlenecks or index out of bounds
        let size = 10_000;
        let data1 = vec![1.0; size];
        let data2 = vec![2.0; size];
        let t1 = new_cpu_tensor(vec![size as u32], data1);
        let t2 = new_cpu_tensor(vec![size as u32], data2);

        let result = t1.mul(&t2).unwrap();
        let data = result.get_data();

        println!(
            "Count of first tensor 1s: {}",
            t1.get_data().iter().filter(|x| **x == 1.0).count()
        );
        println!(
            "Count of second tensor 2s: {}",
            t2.get_data().iter().filter(|x| **x == 2.0).count()
        );
        println!(
            "Count of result with 2.0: {}",
            data.iter().filter(|x| **x == 2.0).count()
        );

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_add_large_data() {
        // Verifies there are no iterator bottlenecks or index out of bounds
        let size = 10_000_000;
        let data1 = vec![1.0; size];
        let data2 = vec![2.0; size];
        let t1 = new_cpu_tensor(vec![size as u32], data1);
        let t2 = new_cpu_tensor(vec![size as u32], data2);

        let result = t1.add(&t2).unwrap();
        let data = result.get_data();

        println!(
            "Count of first tensor 1s: {}",
            t1.get_data().iter().filter(|x| **x == 1.0).count()
        );
        println!(
            "Count of second tensor 2s: {}",
            t2.get_data().iter().filter(|x| **x == 2.0).count()
        );
        println!(
            "Count of result with 3.0: {}",
            data.iter().filter(|x| **x == 3.0).count()
        );

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn test_sub_large_data() {
        // Verifies there are no iterator bottlenecks or index out of bounds
        let size = 10_000_000;
        let data1 = vec![1.0; size];
        let data2 = vec![2.0; size];
        let t1 = new_cpu_tensor(vec![size as u32], data1);
        let t2 = new_cpu_tensor(vec![size as u32], data2);

        let result = t1.sub(&t2).unwrap();
        let data = result.get_data();

        println!(
            "Count of first tensor 1s: {}",
            t1.get_data().iter().filter(|x| **x == 1.0).count()
        );
        println!(
            "Count of second tensor 2s: {}",
            t2.get_data().iter().filter(|x| **x == 2.0).count()
        );
        println!(
            "Count of result with -1.0: {}",
            data.iter().filter(|x| **x == -1.0).count()
        );

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x == -1.0));
    }

    #[test]
    fn test_matmul_large_cpu_data() {
        let rows_a = 999;
        let inner_k = 999;
        let cols_b = 999;

        let all_a = 1.0_f32;
        let all_b = 2.0_f32;

        let data1 = vec![all_a; rows_a * inner_k];
        let data2 = vec![all_b; inner_k * cols_b];

        let t1 = new_cpu_tensor(vec![rows_a as u32, inner_k as u32], data1);
        let t2 = new_cpu_tensor(vec![inner_k as u32, cols_b as u32], data2);

        let result = t1.matmul(&t2).expect("GPU Matmul failed");
        let result_data = result.get_data();

        assert_eq!(result.get_shape(), &vec![rows_a as u32, cols_b as u32]);
        assert_eq!(result_data.len(), rows_a * cols_b);

        assert!(result_data
            .iter()
            .all(|&x| x == (all_a * all_b * inner_k as f32)));

        println!(
            "Matmul verification successful. All sampled elements equal {}",
            inner_k
        );
    }
}
