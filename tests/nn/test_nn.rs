#[cfg(test)]
mod tests {
    use iron_learn::nn::DistributionType;
    use iron_learn::nn::LayerType;
    use iron_learn::ActivationLayer;
    use iron_learn::Layer;
    use iron_learn::LinearLayer;
    use iron_learn::Tensor;
    use iron_learn::{CpuTensor, MeanSquaredErrorLoss, NeuralNet};
    use std::fs;
    use tempfile::tempdir;

    // A helper function to create a network for testing
    fn setup_test_net() -> NeuralNet<CpuTensor<f32>, f32> {
        let mut layers: Vec<Box<dyn Layer<CpuTensor<f32>, f32>>> = Vec::new();

        let layer = LinearLayer::new(1, 2, "test_layer", &DistributionType::Xavier).unwrap();
        layers.push(Box::new(layer));

        let layer = ActivationLayer::new("test_activation_tanh", LayerType::Tanh);
        layers.push(Box::new(layer));

        let layer = ActivationLayer::new("test_activation_sigmoid", LayerType::Sigmoid);
        layers.push(Box::new(layer));

        NeuralNet::new(
            layers,
            Box::new(MeanSquaredErrorLoss),
            2,               // param count
            "2".to_string(), // label
            "TestNet".to_string(),
            0,    // epoch
            0.01, // lr
        )
    }

    #[test]
    fn test_predict_flow() {
        let mut net = setup_test_net();
        let input = CpuTensor::new(vec![1, 2], vec![1.0, 2.0]).unwrap();

        let result = net.predict(&input);

        assert!(result.is_ok());
        // Verify output shape matches expectations
    }

    #[test]
    fn test_save_model_io() {
        let net = setup_test_net();
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("subdir/model.json");
        let path_str = file_path.to_str().unwrap();

        net.save_model(path_str);

        // Check if file exists and is not empty
        let metadata = fs::metadata(path_str).unwrap();
        assert!(metadata.len() > 0);

        // Verify content contains the model name
        let content = fs::read_to_string(path_str).unwrap();
        assert!(content.contains(&net.name));
    }
}
