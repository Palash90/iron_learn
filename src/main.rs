use iron_learn::Matrix;

fn main() {
    let m1 = Matrix::new(vec![2, 2], vec![1,2,3,4]).unwrap();

    let m2 = Matrix::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();

    println!("Result {:?}", (m1 * m2).unwrap().get_data())
}