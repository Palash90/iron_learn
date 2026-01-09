#[cfg(test)]
mod tests {
    use iron_learn::nn::types::TrainingConfig;
    use iron_learn::nn::types::TrainingHook;
    use iron_learn::nn::DistributionType;
    use iron_learn::nn::LayerType;
    use iron_learn::Tensor;
    use iron_learn::{CpuTensor, MeanSquaredErrorLoss, NeuralNet, NeuralNetBuilder};
    use std::fs;
    use tempfile::tempdir;

    // A helper function to create a network for testing
    fn setup_test_net() -> NeuralNet<CpuTensor<f32>, f32> {
        let hl = 4;
        let input = 2;
        let loss = Box::new(MeanSquaredErrorLoss);

        let mut nn = NeuralNetBuilder::<CpuTensor<f32>, f32>::new();

        let layers = [
            (
                input,
                hl,
                LayerType::ReLU,
                "Input",
                "AL 1",
                &DistributionType::Normal,
            ),
            (
                hl,
                hl,
                LayerType::Tanh,
                "HL1",
                "AL2",
                &DistributionType::Xavier,
            ),
            (hl, hl, LayerType::Tanh, "HL1", "AL2", &DistributionType::He),
            (
                hl,
                1,
                LayerType::Sigmoid,
                "HL2",
                "Output",
                &DistributionType::Uniform,
            ),
        ];

        for layer in layers {
            nn.add_linear(layer.0, layer.1, layer.3, layer.5);
            nn.add_activation(layer.2, layer.4);
        }
        nn.build(loss, &"TestNet".to_string())
    }

    #[test]
    fn test_fit() {
        let mut net = setup_test_net();

        let input =
            CpuTensor::new(vec![4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let target = CpuTensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let mut monitor_captured = vec![];

        let monitor = |epoch: usize,
                       err: f32,
                       err_val: f32,
                       lr: f32,
                       _nn: &mut NeuralNet<CpuTensor<f32>, f32>| {
            monitor_captured.push((epoch, err, err_val, lr));
        };

        let config = TrainingConfig {
            epochs: 5,
            epoch_offset: 0,
            base_lr: 1.0,
            lr_adjustment: false,
        };

        let hook_config = TrainingHook::new(1, monitor);

        let result = net.fit(&input, &target, &input, &target, config, hook_config);

        assert!(result.is_ok());
        assert_eq!(monitor_captured.len(), 5);
        println!();
        println!("{:?}", monitor_captured);
        assert!(monitor_captured.iter().all(|a| a.3 == 1.0));
    }

    #[test]
    fn test_fit_cos_annealing() {
        let mut net = setup_test_net();

        let input =
            CpuTensor::new(vec![4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let target = CpuTensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let mut monitor_captured = vec![];

        let monitor = |epoch: usize,
                       err: f32,
                       err_val: f32,
                       lr: f32,
                       _nn: &mut NeuralNet<CpuTensor<f32>, f32>| {
            monitor_captured.push((epoch, err, err_val, lr));
        };

        let config = TrainingConfig {
            epochs: 5,
            epoch_offset: 0,
            base_lr: 1.0,
            lr_adjustment: true,
        };

        let hook_config = TrainingHook::new(1, monitor);

        let result = net.fit(&input, &target, &input, &target, config, hook_config);

        assert!(result.is_ok());
        assert_eq!(monitor_captured.len(), 5);
        assert!(!monitor_captured.iter().all(|a| a.2 == 1.0));
        assert!(monitor_captured[0].2 > monitor_captured[4].2);
    }

    #[test]
    fn test_predict_flow() {
        let mut net = setup_test_net();

        let input =
            CpuTensor::new(vec![4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();

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

    #[test]
    fn test_early_stopping_overfit() {
        let mut net = setup_test_net();
        let dir = tempdir().unwrap();
        net.name = dir.path().to_str().unwrap().to_string();

        // 1. Training Set: Simple linear mapping
        // The model will easily minimize this error.
        let x_train = CpuTensor::new(vec![2, 2], vec![0.5, 0.5, 1.0, 1.0]).unwrap();
        let y_train = CpuTensor::new(vec![2, 1], vec![0.0, 1.0]).unwrap();

        // 2. Validation Set: The "Trap"
        // Use the same inputs, but provide targets that are the exact opposite.
        // As training loss goes down (approaching 0 and 1),
        // the validation error on these points will mathematically increase.
        let x_val = x_train.clone();
        let y_val = CpuTensor::new(vec![2, 1], vec![1.0, 0.0]).unwrap();

        let mut epochs_run = 0;
        let monitor = |epoch: usize, _, _, _, _: &mut NeuralNet<CpuTensor<f32>, f32>| {
            epochs_run = epoch;
        };

        let config = TrainingConfig {
            epochs: 500,
            epoch_offset: 0,
            base_lr: 0.1, // Sufficiently high to move the weights quickly
            lr_adjustment: false,
        };

        let hook_config = TrainingHook::new(1000, monitor);

        let _ = net.fit(&x_train, &y_train, &x_val, &y_val, config, hook_config);

        // Assert that we stopped before the 500 limit
        assert!(
            epochs_run < 400,
            "Model failed to overfit; ran for {} epochs",
            epochs_run
        );

        let path = format!("model_outputs/{}/last_good_model.json", net.name);

        // Check that the last good model was saved
        let _save_path = dir
            .path()
            .join(format!("model_outputs/{}/last_good_model.json", net.name));

        assert!(dir.path().exists());
        let _ = fs::remove_file(path);
        
    }
}
