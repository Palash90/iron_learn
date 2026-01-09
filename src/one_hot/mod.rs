use crate::{Numeric, Tensor};

pub fn one_hot_encode<T: Tensor<D>, D: Numeric>(
    labels: &[u32],
    num_classes: u32,
    labels_per_sample: u32,
) -> Result<T, String> {
    let num_samples = (labels.len() as u32) / labels_per_sample;
    let row_width = (num_classes * labels_per_sample) as usize;
    let mut data = vec![D::zero(); (num_samples as usize) * row_width];

    for (i, sample_labels) in labels.chunks(labels_per_sample as usize).enumerate() {
        for (j, &label_idx) in sample_labels.iter().enumerate() {
            if label_idx >= num_classes {
                return Err(format!(
                    "Label index {} exceeds num_classes {}",
                    label_idx, num_classes
                ));
            }
            let index = (i * row_width) + (j * num_classes as usize) + label_idx as usize;
            data[index] = D::one();
        }
    }

    T::new(vec![num_samples, row_width as u32], data)
}

pub fn one_hot_decode<T: Numeric, U: Tensor<T>>(
    tensor: &U,
    labels_per_sample: u32,
) -> Result<Vec<u32>, String> {
    let shape = tensor.get_shape();
    let num_samples = shape[0] as usize;
    let total_cols = shape[1] as usize;
    let num_classes = (total_cols as u32 / labels_per_sample) as usize;
    let data = tensor.get_data();

    let mut indices = Vec::with_capacity(num_samples * labels_per_sample as usize);

    for r in 0..num_samples {
        for slot in 0..labels_per_sample as usize {
            let mut max_val = T::zero();
            let mut max_idx = 0u32;
            let mut found = false;

            let start_col = slot * num_classes;
            for c in 0..num_classes {
                let current_val = data[r * total_cols + start_col + c];
                if !found || current_val > max_val {
                    max_val = current_val;
                    max_idx = c as u32;
                    found = true;
                }
            }
            indices.push(max_idx);
        }
    }
    Ok(indices)
}
