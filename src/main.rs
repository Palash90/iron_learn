use iron_learn::Matrix;

pub fn main() {
    let m = Matrix { row: 1, column: 5 };
    println!("Matrix add from main.rs {}", m.add());
}
