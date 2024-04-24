use iron_learn::Matrix;

pub fn main() {
    let mut m = Matrix::<i32>::new(vec![3u32, 4u32]);
    m.data.push(54);
}
