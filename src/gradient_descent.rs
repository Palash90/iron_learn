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
/// let w = gradient_descent(&x, &y, &w, learning_rate);
/// ```
///
pub fn gradient_descent(x: &Tensor<f64>, y: &Tensor<f64>, w: &Tensor<f64>, l: f64) -> Tensor<f64> {
    let data_size = *(x.get_shape().first().unwrap()) as f64;
    let prediction = x.mul(w).unwrap();
    let loss = y.sub(&prediction).unwrap();
    let d = x
        .t()
        .unwrap()
        .mul(&loss)
        .unwrap()
        .scale(-1.0 * l / data_size);
    w.sub(&d).unwrap()
}
