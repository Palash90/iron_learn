#[cfg(test)]
mod tests {
    use iron_learn::nn::loss_functions::LossFunctionType;
    use iron_learn::nn::DistributionType;
    use iron_learn::CpuTensor;
    use iron_learn::NeuralNetBuilder;

    #[test]
    fn test_build_parameter_labels() {
        let name = "TestNet".to_string();

        // 1. Test Small Count (< 1000)
        let mut builder_small = NeuralNetBuilder::<CpuTensor<f32>, f32>::new();
        builder_small.add_linear(2, 2, "small", &DistributionType::Xavier); // 4 params
        let net_small = builder_small.build(LossFunctionType::MeanSquaredError, &name);
        assert_eq!(net_small.label, "4");

        // 2. Test Kilo Count (>= 1000)
        let mut builder_kilo = NeuralNetBuilder::<CpuTensor<f32>, f32>::new();
        builder_kilo.add_linear(100, 10, "kilo", &DistributionType::Xavier); // 1000 params
        let net_kilo = builder_kilo.build(LossFunctionType::MeanSquaredError, &name);
        assert_eq!(net_kilo.label, "1k");

        // 3. Test Mega Count (>= 1,000,000)
        let mut builder_mega = NeuralNetBuilder::<CpuTensor<f32>, f32>::new();
        builder_mega.add_linear(1000, 1000, "mega", &DistributionType::Xavier); // 1,000,000 params
        let net_mega = builder_mega.build(LossFunctionType::MeanSquaredError, &name);
        assert_eq!(net_mega.label, "1.0M");
    }

    #[test]
    fn test_build_from_config() {
        use iron_learn::nn::LayerData;
        use iron_learn::nn::LayerType;
        use iron_learn::nn::ModelData;

        let linear = LayerData {
            name: "restored_fc".to_string(),
            layer_type: LayerType::Linear,
            shape: vec![2, 2],
            weights: vec![],
            index: 0,
        };

        let tanh = LayerData {
            name: "restored_tanh".to_string(),
            layer_type: LayerType::Tanh,
            shape: vec![2, 4],
            weights: vec![],
            index: 1,
        };

        let model_data = ModelData {
            name: "RestoredModel".to_string(),
            layers: vec![linear, tanh],
            parameter_count: 4,
            epoch: 1,
            loss_fn_type: LossFunctionType::MeanSquaredError,
            saved_lr: 1.0,
            epoch_error: vec![],
        };

        let nn = NeuralNetBuilder::<CpuTensor<f32>, f32>::build_from_config(
            model_data,
            &DistributionType::Xavier,
        );

        assert_eq!(nn.name, "RestoredModel");
        assert_eq!(nn.layers.len(), 2);
    }

    #[test]
    fn test_build_from_model() {
        use iron_learn::nn::LayerData;
        use iron_learn::nn::LayerType;
        use iron_learn::nn::ModelData;

        let layer_data = LayerData {
            name: "restored_fc".to_string(),
            layer_type: LayerType::Linear,
            shape: vec![2, 2],
            weights: vec![0.1, 0.2, 0.3, 0.4],
            index: 0,
        };

        let model_data = ModelData {
            name: "RestoredModel".to_string(),
            layers: vec![layer_data],
            parameter_count: 4,
            epoch: 10,
            saved_lr: 0.01,
            loss_fn_type: LossFunctionType::MeanSquaredError,
            epoch_error: vec![],
        };

        let net = NeuralNetBuilder::<CpuTensor<f32>, f32>::build_from_model(model_data);

        assert_eq!(net.name, "RestoredModel");
        assert_eq!(net.layers.len(), 1);
    }
}
