#[cfg(test)]
mod tests {
    use iron_learn::nn::DistributionType;
    use iron_learn::CpuTensor;
    use iron_learn::MeanSquaredErrorLoss;
    use iron_learn::NeuralNetBuilder;

    #[test]
    fn test_build_parameter_labels() {
        let loss_fn = Box::new(MeanSquaredErrorLoss);
        let name = "TestNet".to_string();

        // 1. Test Small Count (< 1000)
        let mut builder_small = NeuralNetBuilder::<CpuTensor<f32>, f32>::new();
        builder_small.add_linear(2, 2, "small", &DistributionType::Xavier); // 4 params
        let net_small = builder_small.build(loss_fn, &name);
        assert_eq!(net_small.label, "4");

        // 2. Test Kilo Count (>= 1000)
        let mut builder_kilo = NeuralNetBuilder::<CpuTensor<f32>, f32>::new();
        let loss_fn = Box::new(MeanSquaredErrorLoss);
        builder_kilo.add_linear(100, 10, "kilo", &DistributionType::Xavier); // 1000 params
        let net_kilo = builder_kilo.build(loss_fn, &name);
        assert_eq!(net_kilo.label, "1k");

        // 3. Test Mega Count (>= 1,000,000)
        let mut builder_mega = NeuralNetBuilder::<CpuTensor<f32>, f32>::new();
        let loss_fn = Box::new(MeanSquaredErrorLoss);
        builder_mega.add_linear(1000, 1000, "mega", &DistributionType::Xavier); // 1,000,000 params
        let net_mega = builder_mega.build(loss_fn, &name);
        assert_eq!(net_mega.label, "1.0M");
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
        };

        let loss_fn = Box::new(MeanSquaredErrorLoss);
        let net = NeuralNetBuilder::<CpuTensor<f32>, f32>::build_from_model(model_data, loss_fn);

        assert_eq!(net.name, "RestoredModel");
        assert_eq!(net.layers.len(), 1);
    }
}
