use iron_learn::Tensor;

pub fn main() {
    let mut m = Tensor::<u32>::new(vec![3u32, 4u32]);
    m.data.push(54u32);
    m.data.push(5444u32);

    let mut m1 = Tensor::new(vec![3u32, 4u32]);
    m1.data.push(4u32);
    m1.data.push(46u32);

    println!("{:?}", m + m1)
}
