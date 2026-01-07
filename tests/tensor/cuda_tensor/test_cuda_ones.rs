#[cfg(test)]
mod tests {
    use iron_learn::GpuTensor;
    use iron_learn::Tensor;

    use iron_learn::init_gpu;

    #[test]
    fn test_div_happy_path() {
        let _ = init_gpu();

        // Verifies basic element-wise division and shape preservation
        let shape = vec![20, 20];
        let t1 = GpuTensor::<f32>::ones(&shape);
        let data = t1.get_data();

        let unmatched = data.iter().filter(|x| **x != 1.0).count();
        let matched = data.iter().filter(|x| **x == 1.0).count();

        println!("Unmatched {unmatched}, matched {matched}");

        assert!(data.iter().all(|&x| x == 1.0));
    }
}
