use crate::{Numeric, Tensor};

/// Helper to create a One-Hot Encoded Tensor from categorical indices.
///
/// # Arguments
/// * `labels` - A slice of indices (0-indexed).
/// * `num_classes` - Total number of unique categories.
pub fn one_hot_encode<T: Numeric, U: Tensor<T>>(
    labels: &[u32],
    num_classes: u32,
) -> Result<U, String> {
    let num_samples = labels.len() as u32;

    // 1. Create a flat vector for the data
    // Total size will be rows (samples) * columns (classes)
    let total_elements = (num_samples * num_classes) as usize;
    let mut data = vec![T::zero(); total_elements];

    // 2. Fill the "1"s at the correct positions
    for (row, &label_idx) in labels.iter().enumerate() {
        if label_idx >= num_classes {
            return Err(format!(
                "Label index {} exceeds number of classes {}",
                label_idx, num_classes
            ));
        }

        // Row-major index calculation: (current_row * total_cols) + current_col
        let index = (row * num_classes as usize) + label_idx as usize;
        data[index] = T::one();
    }

    // 3. Construct the Tensor using the trait's new() method
    U::new(vec![num_samples, num_classes], data)
}

/// Decodes a One-Hot encoded Tensor back into a vector of indices.
///
/// # Arguments
/// * `tensor` - The tensor to decode (assumed to be 2D).
pub fn one_hot_decode<T: Numeric, U: Tensor<T>>(tensor: &U) -> Result<Vec<u32>, String> {
    let shape = tensor.get_shape();

    // Ensure the tensor is 2D
    if shape.len() != 2 {
        return Err("Decoding is only supported for 2D Tensors".to_string());
    }

    let num_samples = shape[0] as usize;
    let num_classes = shape[1] as usize;
    let data = tensor.get_data();

    let mut indices = Vec::with_capacity(num_samples);

    for r in 0..num_samples {
        let mut max_val = T::zero();
        let mut max_idx = 0u32;
        let mut found = false;

        for c in 0..num_classes {
            let current_val = data[r * num_classes + c];

            // In a strict one-hot, we look for the 1.
            // In a soft-max output, we look for the highest probability.
            if !found || current_val > max_val {
                max_val = current_val;
                max_idx = c as u32;
                found = true;
            }
        }
        indices.push(max_idx);
    }

    Ok(indices)
}
