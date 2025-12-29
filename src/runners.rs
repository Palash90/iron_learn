use crate::MeanSquaredErrorLoss;
// use crate::commons::add_bias_term;
use crate::commons::add_bias_term;
use crate::commons::denormalize_features;
use crate::commons::normalize_features_mean_std;
use crate::neural_network::LayerType;
use crate::neural_network::NeuralNetDataType;
use crate::normalize_features;
use crate::read_file::deserialize_data;
use crate::read_file::deserialize_data_double_precision;
use crate::read_file::deserialize_model;
use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;
use crate::DataDoublePrecision;
use crate::NeuralNet;
use crate::NeuralNetBuilder;
use crate::GLOBAL_CONTEXT;
use crate::{linear_regression, logistic_regression, predict_linear, predict_logistic};
use std::time::Instant;

use image::{ImageBuffer, Luma};

use std::thread;
use std::time::Duration;

use crate::neural_network::DistributionType;

pub fn run_logistic<T>() -> Result<(), String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    let l = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let xy: DataDoublePrecision = deserialize_data_double_precision(&data_path)
        .map_err(|e| format!("Data deserialization error: {}", e))?;

    print!("\nLogistic Regression\n");
    print!("Number of examples (m): {}\n", xy.m);
    print!("Number of features (n): {}\n", xy.n);
    print!("Total X values length: {}\n", xy.x.len());
    print!("Total Y values length: {}\n", xy.y.len());

    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    let mut w = T::new(vec![xy.n + 1, 1], vec![0.0; (xy.n + 1) as usize])?;

    let now = Instant::now();

    w = logistic_regression(&x, &y, w, l, e).unwrap();
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    let x_test = T::new(vec![xy.m_test, xy.n], xy.x_test.clone())?;
    let y_test = T::new(vec![xy.m_test, 1], xy.y_test.clone())?;

    let predictions = predict_logistic(&x_test, &w)?;

    let mut correct = 0;
    let total = xy.m_test as usize;
    let predictions_data = predictions.get_data();
    let y_test_data = y_test.get_data();

    for i in 0..total {
        let pred = predictions_data[i];
        let actual = y_test_data[i];
        if (pred - actual).abs() < 1e-10 {
            correct += 1;
        }
    }

    let accuracy = (correct as f64) / (total as f64) * 100.0;
    println!("\nResults:");
    println!("Total samples: {}", total);
    println!("Correct predictions: {}", correct);
    println!("Accuracy: {:.2?}%", accuracy);

    Ok(())
}

pub fn run_linear<T>() -> Result<(), String>
where
    T: Tensor<f64> + TensorMath<f64, MathOutput = T>,
{
    let l = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;

    let xy = deserialize_data_double_precision(data_path)
        .map_err(|e| format!("Data deserialization error: {}", e))?;

    print!("\nLinear Regression\n");
    print!("Number of examples (m): {}\n", xy.m);
    print!("Number of features (n): {}\n", xy.n);
    print!("Total X values length: {}\n", xy.x.len());
    print!("Total Y values length: {}\n", xy.y.len());

    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    let (x, x_mean, x_std) = normalize_features_mean_std(&x);
    let (y, y_mean, y_std) = normalize_features_mean_std(&y);

    let mut w = T::new(vec![xy.n + 1, 1], vec![0.0; (xy.n + 1) as usize])?;

    let now = Instant::now();

    w = linear_regression(&x, &y, w, l, e).unwrap();

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    if xy.m_test == 0 {
        println!("\nNo test data available for prediction.");
        return Ok(());
    }

    let x_test = T::new(vec![xy.m_test, xy.n], xy.x_test.clone())?;
    let y_test = T::new(vec![xy.m_test, 1], xy.y_test.clone())?;

    let x_test = normalize_features(&x_test, &x_mean, &x_std);

    let predictions = predict_linear(&x_test, &w)?;

    let predictions = denormalize_features(&predictions, &y_mean, &y_std);

    let mut total_squared_error = 0.0;
    let total = xy.m_test as usize;
    let predictions_data = predictions.get_data();
    let y_test_data = y_test.get_data();

    for i in 0..total {
        let pred = predictions_data[i];
        let actual = y_test_data[i];
        let error = pred - actual;
        total_squared_error += error * error;
    }

    let mse = total_squared_error / (total as f64);
    println!("\nResults:");
    println!("Total test samples: {}", total);
    println!("Mean Squared Error: {:.4}", mse);
    println!("Root MSE: {:.4}", mse.sqrt());

    Ok(())
}

pub fn run_neural_net<T>() -> Result<(), String>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    let lr = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate as NeuralNetDataType;
    let e = GLOBAL_CONTEXT.get().unwrap().epochs;
    let data_path = &GLOBAL_CONTEXT.get().unwrap().data_path;
    let hidden_length = GLOBAL_CONTEXT.get().unwrap().hidden_layer_length;
    let weights_path = &GLOBAL_CONTEXT.get().unwrap().weights_path;
    let monitor_interval = GLOBAL_CONTEXT.get().unwrap().monitor_interval;
    let sleep_time = GLOBAL_CONTEXT.get().unwrap().sleep_time;
    let name = &GLOBAL_CONTEXT.get().unwrap().name;
    let restore = GLOBAL_CONTEXT.get().unwrap().restore;
    let lr_adjustment = GLOBAL_CONTEXT.get().unwrap().lr_adjust;
    let distribution = &GLOBAL_CONTEXT.get().unwrap().distribution;

    let xy =
        deserialize_data(data_path).map_err(|e| format!("Data deserialization error: {}", e))?;

    let x = T::new(vec![xy.m, xy.n], xy.x.clone()).unwrap();
    // let x = x.scale(1.0 / 199.0).unwrap(); // Normalization
    let y = T::new(vec![xy.m, 1], xy.y.clone()).unwrap();

    //let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    //let x_test = T::new(vec![xy.m_test, xy.n], xy.x_test.clone())?;

    //let (x, x_mean, x_std) = normalize_features_mean_std(&x);
    //let (y, y_mean, y_std) = normalize_features_mean_std(&y);

    // let x_with_bias = add_bias_term(&x).unwrap(); // Bias Trick

    let loss_function_instance = Box::new(MeanSquaredErrorLoss);
    let input_length = xy.n;

    let input_length = input_length; //  + 1; // To compensate for bias

    let nn = define_neural_net::<T>(hidden_length, input_length, distribution);

    let weights_path = name.to_owned() + "/" + &weights_path;

    let (l, epoch_offset, mut nn) = match !weights_path.is_empty() {
        true => match deserialize_model(&weights_path) {
            Some(model) => match restore {
                true => (
                    model.saved_lr.clone(),
                    model.epoch.clone(),
                    NeuralNetBuilder::build_from_model(model, loss_function_instance),
                ),
                false => (lr, 0, nn.build(loss_function_instance, name)),
            },
            None => (lr, 0, nn.build(loss_function_instance, name)),
        },
        false => (lr, 0, nn.build(loss_function_instance, name)),
    };

    let mut start_time = Instant::now();
    let mut last_epoch = 0;

    let monitor = |epoch: usize,
                   err: NeuralNetDataType,
                   current_lr: NeuralNetDataType,
                   nn: &mut NeuralNet<T>| {
        let elapsed = start_time.elapsed();
        start_time = Instant::now();

        println!("\tEpoch {epoch}: Loss (MSE) = {err:.8}, Current LR : {current_lr:.8}, {last_epoch} - {epoch} time elapsed: {elapsed:.2?}");

        last_epoch = epoch;

        if epoch % monitor_interval == 0 {
            let y_pred = nn.predict(&x).unwrap();

            if epoch % (monitor_interval) == 0 {
                if name.contains(&"image") {
                    draw_image(epoch as i32, &x, &y_pred, 200, 200, name);
                }

                nn.save_model(&weights_path);
            }

            // Rest for a few seconds before starting again
            if sleep_time > 0 && epoch != 0 {
                println!("Taking a nap");
                thread::sleep(Duration::from_secs(sleep_time));
                println!("Awake again");
            }
        }
    };

    if !restore {
        if name.contains(&"image") {
            draw_image(-1, &x, &y, 200, 200, name);
        }
    }

    let _ = nn.fit(
        &x,
        &y,
        e as usize,
        epoch_offset,
        l,
        lr_adjustment,
        monitor,
        monitor_interval,
    );

    if !name.contains(&"image") {
        let predictions = nn.predict(&x).unwrap();
        println!();
        println!("Input:");
        x.print_matrix();

        println!("Predictions:");
        predictions.print_matrix();
    }

    Ok(())
}

fn define_neural_net<T>(hl: u32, input: u32, distribution: &DistributionType) -> NeuralNetBuilder<T>
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    let mut nn = NeuralNetBuilder::<T>::new();

    let _image_layers = [
        (input, hl, LayerType::Tanh, "Input", "AL 1"),
        (hl, hl, LayerType::Tanh, "HL1", "AL2"),
        (hl, 2 * hl, LayerType::Tanh, "HL2", "AL3"),
        (2 * hl, hl, LayerType::Tanh, "HL3", "AL4"),
        (hl, hl / 2, LayerType::Tanh, "HL4", "AL5"),
        (hl / 2, hl / 2, LayerType::Tanh, "HL10", "AL11"),
        (hl / 2, hl / 2, LayerType::Tanh, "HL11", "AL12"),
        (hl / 2, 1, LayerType::Sigmoid, "HL12", "Output"),
    ];

    let xor_layers = [
        (input, hl, LayerType::Tanh, "Input", "AL 1"),
        (hl, hl, LayerType::Tanh, "HL4", "AL5"),
        (hl, 1, LayerType::Sigmoid, "HL12", "Output"),
    ];

    for layer in xor_layers {
        nn.add_linear(layer.0, layer.1, layer.3, distribution);
        nn.add_activation(layer.2, layer.4);
    }
    nn
}

fn draw_image<T>(epoch: i32, x_test: &T, y: &T, height: u32, width: u32, name: &String)
where
    T: Tensor<NeuralNetDataType> + TensorMath<NeuralNetDataType, MathOutput = T> + 'static,
{
    let mut image_data: Vec<(u32, u32, u8)> = vec![];

    let x_data = x_test.get_data();
    let y_data = y.get_data();

    for i in 0..y_data.len() {
        let x_co = (x_data[2 * i] * (width - 1) as f32).round() as u32;
        let y_co = (x_data[2 * i + 1] * (height - 1) as f32).round() as u32;
        let pixel = 255 - (y_data[i] * 255.0) as u8;

        image_data.push((x_co, y_co, pixel));
    }

    draw_grid(image_data, epoch, height, width, name);
}

fn draw_grid(points: Vec<(u32, u32, u8)>, epoch: i32, height: u32, width: u32, name: &String) {
    let mut imgbuf = ImageBuffer::from_pixel(width, height, Luma([255u8]));

    for (x, y, pixel) in points {
        if x < width && y < height {
            imgbuf.put_pixel(x, y, Luma([pixel]));
        }
    }

    let image_file = name.to_owned() + "/images/output" + &epoch.to_string() + ".png";

    match imgbuf.save(&image_file) {
        Ok(_) => println!("Image successfully rendered to {}", image_file),
        Err(e) => eprintln!("Error saving image {}: {}", image_file, e),
    }
}
