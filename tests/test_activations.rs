#[cfg(test)]
mod activations_tests {
    use iron_learn::nn::{cos, sigmoid, sigmoid_prime, sin, tanh, tanh_prime};
    use iron_learn::CpuTensor;
    use iron_learn::Tensor;

    #[test]
    fn test_sigmoid_and_prime_cpu() {
        let vals = vec![0.0_f32, 1.0_f32, -1.0_f32, 2.0_f32];
        let x = CpuTensor::new(vec![4], vals.clone()).unwrap();

        let s = sigmoid(&x).expect("sigmoid failed");
        let sdata = s.get_data();

        let eps = 1e-6_f32;
        for (i, v) in vals.iter().enumerate() {
            let expected = 1.0_f32 / (1.0_f32 + (-v).exp());
            assert!(
                (sdata[i] - expected).abs() < eps,
                "sigmoid mismatch at {}",
                i
            );
        }

        let sp = sigmoid_prime(&s).expect("sigmoid_prime failed");
        let spd = sp.get_data();
        for (i, sd) in sdata.iter().enumerate() {
            let expected = sd * (1.0_f32 - sd);
            assert!(
                (spd[i] - expected).abs() < eps,
                "sigmoid_prime mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_tanh_and_prime_cpu() {
        let vals = vec![0.0_f32, 1.0_f32, -1.0_f32, 0.5_f32];
        let x = CpuTensor::new(vec![4], vals.clone()).unwrap();

        let t = tanh(&x).expect("tanh failed");
        let tdata = t.get_data();

        let eps = 1e-6_f32;
        for (i, v) in vals.iter().enumerate() {
            let expected = v.tanh();
            assert!((tdata[i] - expected).abs() < eps, "tanh mismatch at {}", i);
        }

        let tp = tanh_prime(&t).expect("tanh_prime failed");
        let tpd = tp.get_data();
        for (i, td) in tdata.iter().enumerate() {
            let expected = 1.0_f32 - td * td;
            assert!(
                (tpd[i] - expected).abs() < eps,
                "tanh_prime mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_sin_and_cos_cpu() {
        let vals = vec![0.0_f32, std::f32::consts::FRAC_PI_2, -1.0_f32, 2.0_f32];
        let x = CpuTensor::new(vec![4], vals.clone()).unwrap();

        let s = sin(&x).expect("sin failed");
        let sdata = s.get_data();
        let c = cos(&x).expect("cos failed");
        let cdata = c.get_data();

        let eps = 1e-6_f32;
        for (i, v) in vals.iter().enumerate() {
            let expected_s = v.sin();
            let expected_c = v.cos();
            assert!((sdata[i] - expected_s).abs() < eps, "sin mismatch at {}", i);
            assert!((cdata[i] - expected_c).abs() < eps, "cos mismatch at {}", i);
        }
    }
}
