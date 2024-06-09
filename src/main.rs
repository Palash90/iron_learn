use iron_learn::{logistic_regression, linear_regression};
use iron_learn::Tensor;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::fs;

#[derive(Debug, Deserialize, Serialize)]
struct XY {
    m: u32, // Number of data points
    n: u32, // Number of features
    x: Vec<f64>, // A matrix of data length m * n
    y: Vec<f64> // A matrix of data length m
}

#[derive(Debug, Deserialize, Serialize)]
struct Data {
    linear: XY,
    logistic: XY
}
fn main() {
    let l = 0.0015;

    let e = 100_000_00;
    let e = 100_000_0;

    let contents = fs::read_to_string("data.json")
        .expect("Should have been able to read the file");

    let xy: Data = serde_json::from_str(&contents).unwrap();
    let xy = xy.linear;

    let x = Tensor::new(vec![xy.m, xy.n], xy.x).unwrap();
    let y = Tensor::new(vec![xy.m, 1], xy.y).unwrap();
    let mut w = Tensor::new(vec![xy.n, 1], vec![0.0; xy.n as usize]).unwrap();

    for _ in 0..e {
        w = linear_regression(&x, &y, &w, l)
    }

    println!("{}", w);

    let xy: Data = serde_json::from_str(&contents).unwrap();
    let xy = xy.logistic;

    let x = Tensor::new(vec![xy.m, xy.n], xy.x).unwrap();
    let y = Tensor::new(vec![xy.m, 1], xy.y).unwrap();
    let mut w = Tensor::new(vec![xy.n, 1], vec![0.0; xy.n as usize]).unwrap();

    let now = Instant::now();

    for _ in 0..e {
        w = logistic_regression(&x, &y, &w, l)
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    println!("{}", w);
}
