//! The `gradient_descent` module provides gradient descent optimization

use crate::Tensor;

/// The `gradient_descent` function performs a single step of the gradient descent optimization algorithm.
///
/// # Arguments
///
/// * `x` - All values of all features arranged in a m * n matrix, where m is nuber of data points and n is number of features.
/// * `w` - The current value of the weights for each feature, arranged in n * 1 tensor, where n is number of features.
/// * `y` - The collected target values arranged in a m * 1 matrix, where m is the number of data points.
/// * `l` - The learning rate, which controls the size of the update step.
///
/// # Returns
///
/// The updated weight matrix
///
/// # Example
///
/// ```
/// use iron_learn::Tensor;
/// use iron_learn::gradient_descent::gradient_descent;
///
/// let learning_rate: f64 = 0.01;
/// let w = Tensor::new(vec![2, 1], vec![3.0, 4.0]).unwrap();
/// let x = Tensor::new(vec![1, 2], vec![3.0, 4.0]).unwrap();
/// let y = Tensor::new(vec![1, 1], vec![5.0]).unwrap();
///
/// let w = gradient_descent(&x, &y, &w, learning_rate, false); // For linear regression
/// ```
///
pub fn gradient_descent(x: &Tensor<f64>, y: &Tensor<f64>, w: &Tensor<f64>, l: f64, logistic: bool) -> Tensor<f64> {
    let data_size = *(x.get_shape().first().unwrap()) as f64;
    let prediction = match logistic {
        false => x.mul(w).unwrap(),
        true => {
            let result = x.mul(w).unwrap();
            let result = -result;
            let result = Tensor::exp(&result);
            let shape = result.get_shape();

            let result = result.get_data().iter().map(|t|1.0/(1.0 + t)).collect();
            Tensor::new(shape, result).unwrap()
        }
    };
    
    let loss = y.sub(&prediction).unwrap();
    let d = x
        .t()
        .unwrap()
        .mul(&loss)
        .unwrap()
        .scale(-1.0 * l / data_size);
    w.sub(&d).unwrap()
}
