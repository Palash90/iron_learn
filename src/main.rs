use iron_learn::Tensor;

fn main() {
    let mut m1 = Tensor::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();

    let mut m2 = Tensor::new(vec![3, 2], vec![ 7, 8, 9, 10, 11, 12]).unwrap();

    println!("{} ", m1 );

    m1.reshape(vec![3,2]);

    m2.reshape(vec![2,3]);

    println!("{} ", m1 );

}
