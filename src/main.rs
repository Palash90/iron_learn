use iron_learn::gradient_descent::gradient_descent;
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

fn main() {

    let contents = fs::read_to_string("data.json")
        .expect("Should have been able to read the file");

    let xy: XY = serde_json::from_str(&contents).unwrap();

    let x = Tensor::new(vec![xy.m, xy.n], xy.x).unwrap();
    let y = Tensor::new(vec![xy.m, 1], xy.y).unwrap();
    let mut w = Tensor::new(vec![xy.n, 1], vec![0.0; xy.n as usize]).unwrap();

    let l = 0.0001;

    let e = 10000;
    
    let now = Instant::now();

    for i in 0..e {
        w = gradient_descent(&x, &y, &w, l)

    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    println!("{}", w); // Should return something close to [5, 12, 13, 50]
}
