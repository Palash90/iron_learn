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

    let e = 100000;
    
    for _ in 0..e {
       // println!("{:?}", w.get_data());
        w = gradient_descent(&x, &y, &w, l, false)
    }

    println!("{}", w); // Should return something close to [5, 12, 13, 50]

    let x = Tensor::new(vec![6, 2], vec![0.5, 1.5, 1.0, 1.0, 1.5, 0.5, 3.0, 0.5, 2.0, 2.0, 1.0, 2.5]).unwrap();
    let y = Tensor::new(vec![6, 1], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let mut w = Tensor::new(vec![2, 1], vec![8.0, 8.0]).unwrap();

    let l = 0.0001;

    let e = 100000;
    
    let now = Instant::now();

    for _ in 0..e {
       // println!("{:?}", w.get_data());
        w = gradient_descent(&x, &y, &w, l, true)
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    println!("{}", w); // Should return something close to [5, 12, 13, 50]
}
