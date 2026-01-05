use iron_learn::gradient_descent::gradient_descent;
use iron_learn::CpuTensor;
 use iron_learn::Tensor;

#[cfg(test)]
mod tests {
    use super::*;

    // Helper for generating test data
    fn get_cpu_tensor(shape: Vec<u32>, data: Vec<f32>) -> Result<CpuTensor<f32>, String> {
        CpuTensor::new(shape, data)
    }

    #[test]
    fn test_gradient_descent_linear_convergence() -> Result<(), Box<dyn std::error::Error>> {
        // Setup: Simple linear relationship y = 2x
        let x = get_cpu_tensor(vec![2, 1], vec![1.0, 2.0])?;
        let y = get_cpu_tensor(vec![2, 1], vec![2.0, 4.0])?;
        let w = get_cpu_tensor(vec![1, 1], vec![0.0])?; // Initial weight 0
        let learning_rate = 0.1;

        // Run gradient descent (linear mode)
        let updated_w = gradient_descent(&x, &y, &w, learning_rate, false)?;

        // Calculation check:
        // Prediction = [1*0, 2*0] = [0, 0]
        // Loss = [0-2, 0-4] = [-2, -4]
        // Grad_raw = [1, 2] * [-2, -4]^T = (1*-2) + (2*-4) = -10
        // Scale = -10 * (0.1 / 2) = -0.5
        // New W = 0 - (-0.5) = 0.5
        assert_eq!(updated_w.get_data()[0], 0.5);
        Ok(())
    }

    #[test]
    fn test_gradient_descent_logistic_activation() -> Result<(), Box<dyn std::error::Error>> {
        let x = get_cpu_tensor(vec![1, 1], vec![0.0])?;
        let y = get_cpu_tensor(vec![1, 1], vec![1.0])?;
        let w = get_cpu_tensor(vec![1, 1], vec![0.0])?;
        
        // sigmoid(0) is 0.5. 
        // With logistic = true, it uses sigmoid. With false, it uses raw 'lines' (0.0).
        let result = gradient_descent(&x, &y, &w, 1.0, true)?;

        // If sigmoid worked: loss is (0.5 - 1.0) = -0.5. 
        // Gradient update will be 0 - (1.0 * -0.5 / 1) = 0.5
        assert_eq!(result.get_data()[0], 0.5);
        Ok(())
    }

    #[test]
    fn test_gradient_descent_invalid_shapes() {
        let x = get_cpu_tensor(vec![2, 1], vec![1.0, 2.0]).unwrap();
        let y = get_cpu_tensor(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap(); // Mismatched N
        let w = get_cpu_tensor(vec![1, 1], vec![0.0]).unwrap();

        let result = gradient_descent(&x, &y, &w, 0.1, false);
        
        // Should return a String error because subtraction/multiplication 
        // of mismatched shapes should fail.
        assert!(result.is_err());
    }
}
