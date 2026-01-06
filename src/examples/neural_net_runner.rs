use crate::examples::contexts::GLOBAL_CONTEXT;
use crate::examples::init::ExampleMode;
use crate::examples::read_file::deserialize_data;
use crate::examples::read_file::deserialize_model;
use crate::nn::LayerType;
use crate::numeric::FloatingPoint;
use crate::tensor::math::TensorMath;
use crate::tensor::Tensor;
use crate::NeuralNet;
use crate::NeuralNetBuilder;
use std::time::Instant;

use image::{ImageBuffer, Luma};

use std::thread;
use std::time::Duration;

use crate::nn::DistributionType;

use crate::commons::add_bias_term;
use crate::MeanSquaredErrorLoss;
use std::fs;
use std::path::Path;

/// Train and evaluate a neural network using configuration from the global context.
///
/// This function builds a neural network according to global settings,
/// loads optional saved weights, runs training with a monitoring callback,
/// and prints predictions for non-image tasks. Returns `Ok(())` on success or
/// an error string.
pub fn run_neural_net<T, D>() -> Result<(), String>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    let lr = GLOBAL_CONTEXT
        .get()
        .ok_or("GLOBAL_CONTEXT not initialized")?
        .learning_rate;
    let lr = D::from_f64(lr);
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
    let predict_only = GLOBAL_CONTEXT.get().unwrap().predict_only;
    let example = GLOBAL_CONTEXT.get().unwrap().example_mode;
    let weights_path = name.to_owned() + "/" + weights_path;
    let resize = GLOBAL_CONTEXT.get().unwrap().resize;

    let xy =
        deserialize_data(data_path).map_err(|e| format!("Data deserialization error: {}", e))?;

    let x = T::new(vec![xy.m, xy.n], xy.x.clone())?;
    let y = T::new(vec![xy.m, 1], xy.y.clone())?;

    let x_test = match example {
        ExampleMode::ImageNeuralNet => match resize {
            0 => T::new(vec![xy.m, xy.n], xy.x.clone())?,
            n_pixels => {
                println!("Creating canvas for {n_pixels} x {n_pixels} pixels");
                let mut coords = vec![];
                for i in 0..n_pixels {
                    for j in 0..n_pixels {
                        coords.push(D::from_u32(j));
                        coords.push(D::from_u32(i));
                    }
                }
                T::new(vec![n_pixels * n_pixels, xy.n], coords)?
            }
        },
        _ => T::new(vec![xy.m_test, xy.n], xy.x_test.clone())?,
    };

    let y_test = match example {
        ExampleMode::ImageNeuralNet => T::new(vec![xy.m, 1], xy.y.clone())?,
        _ => T::new(vec![xy.m_test, 1], xy.y_test.clone())?,
    };

    let (x_with_bias, input_length, x_test_with_bias) =
        prepare_network_input(&x, &x_test, example)?;

    let loss_function_instance = Box::new(MeanSquaredErrorLoss);

    let (l, epoch_offset, mut nn) = match !weights_path.is_empty() && restore {
        true => match deserialize_model::<D>(&weights_path) {
            Some(model) => (
                model.saved_lr,
                model.epoch,
                NeuralNetBuilder::build_from_model(model, loss_function_instance),
            ),
            None => (
                lr,
                0,
                define_neural_net::<T, D>(hidden_length, input_length, distribution)
                    .build(loss_function_instance, name),
            ),
        },
        false => (
            lr,
            0,
            define_neural_net::<T, D>(hidden_length, input_length, distribution)
                .build(loss_function_instance, name),
        ),
    };

    let mut start_time = Instant::now();
    let mut last_epoch = 0;

    let monitor = |epoch: usize, err: D, current_lr: D, nn: &mut NeuralNet<T, D>| {
        let elapsed = start_time.elapsed();
        start_time = Instant::now();

        println!("\tEpoch {epoch}: Loss (MSE) = {err:.8}, Current LR : {current_lr:.8}, {last_epoch} - {epoch} time elapsed: {elapsed:.2?}");

        last_epoch = epoch;

        if epoch.is_multiple_of(monitor_interval) {
            let y_pred = nn.predict(&x_with_bias).unwrap();

            if epoch.is_multiple_of(monitor_interval) {
                if example == ExampleMode::ImageNeuralNet {
                    let size = (xy.m as f64).sqrt() as u32;

                    let pixels: Vec<u8> = y_pred
                        .get_data()
                        .clone()
                        .iter()
                        .map(|x| (x.f64() * 255.0) as u8)
                        .collect();
                    let coordinates: Vec<u32> =
                        xy.x.clone().iter().map(|x| x.f64() as u32).collect();

                    draw_image(epoch as i32, &coordinates, &pixels, size, size, name);
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

    if !predict_only {
        let _ = nn.fit(
            &x_with_bias,
            &y,
            e as usize,
            epoch_offset,
            l,
            lr_adjustment,
            monitor,
            monitor_interval,
        );
    } else {
        println!("Skipped Fitting as in Predict Only Mode");
    }

    let predictions = nn.predict(&x_test_with_bias).unwrap();

    if example == ExampleMode::ImageNeuralNet {
        let size = (x_test_with_bias.get_shape()[0] as f64).sqrt() as u32;

        let pixels: Vec<u8> = predictions
            .get_data()
            .clone()
            .iter()
            .map(|x| (x.f64() * 255.0) as u8)
            .collect();
        let coordinates: Vec<u32> = x_test
            .get_data()
            .clone()
            .iter()
            .map(|x| x.f64() as u32)
            .collect();

        draw_image(-1, &coordinates, &pixels, size, size, name);
    } else {
        let error = predictions.sub(&y_test).unwrap();
        let error = error.sum().unwrap();
        println!("Test Error:");
        error.print_matrix();

        println!("X Test:");
        x_test.print_matrix();
        println!("Y Test:");
        y_test.print_matrix();
        println!("Predictions:");
        predictions.print_matrix();
    }

    Ok(())
}

fn prepare_network_input<T, D>(x: &T, x_test: &T, mode: ExampleMode) -> Result<(T, u32, T), String>
where
    T: Tensor<D>,
    D: FloatingPoint,
{
    let (x_normalized, x_test_normalized) = match mode {
        ExampleMode::ImageNeuralNet => {
            let x_size = (x.get_shape()[0] as f64).sqrt() as u32;
            let x_norm = x.scale(D::one() / D::from_u32(x_size))?;

            let size = (x_test.get_shape()[0] as f64).sqrt() as u32;
            let x_test_norm = x_test.scale(D::one() / D::from_u32(size))?;

            (x_norm, x_test_norm)
        }
        _ => (
            T::zeroes(x.get_shape()).add(x)?,
            T::zeroes(x_test.get_shape()).add(x_test)?,
        ),
    };

    let x_with_bias = add_bias_term(&x_normalized)?;

    let x_test_with_bias = add_bias_term(&x_test_normalized)?;

    let input_length = x.get_shape()[1] + 1;

    Ok((x_with_bias, input_length, x_test_with_bias))
}

fn define_neural_net<T, D>(
    hl: u32,
    input: u32,
    distribution: &DistributionType,
) -> NeuralNetBuilder<T, D>
where
    T: Tensor<D> + TensorMath<D, MathOutput = T> + 'static,
    D: FloatingPoint + 'static,
{
    let mut nn = NeuralNetBuilder::<T, D>::new();

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

fn draw_image(epoch: i32, x_data: &[u32], y_data: &[u8], height: u32, width: u32, name: &String) {
    println!("Drawing {height} x {width} images");
    let mut image_data: Vec<(u32, u32, u8)> = vec![];

    for i in 0..y_data.len() {
        let x_co = x_data[2 * i];
        let y_co = x_data[2 * i + 1];
        let pixel = 255 - y_data[i];

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

    let path = Path::new(&image_file);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap(); // Creates all directories if they don't exist
    }

    match imgbuf.save(&image_file) {
        Ok(_) => println!("Image successfully rendered to {}", image_file),
        Err(e) => eprintln!("Error saving image {}: {}", image_file, e),
    }
}
