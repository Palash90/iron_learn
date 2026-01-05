use iron_learn::Tensor;
use iron_learn::{Complex, CpuTensor};

#[cfg(test)]
#[test]
pub fn test_add_i8() {
    let m1 = CpuTensor::<i8>::new(vec![1], vec![1i8]).unwrap();
    let m2 = CpuTensor::new(vec![1], vec![3i8]).unwrap();
    let result = CpuTensor::new(vec![1], vec![4i8]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());

    let m1 = CpuTensor::<i8>::new(vec![1, 2], vec![1i8, 2i8]).unwrap();
    let m2 = CpuTensor::new(vec![1, 2], vec![3i8, 4i8]).unwrap();
    let result = CpuTensor::new(vec![1, 2], vec![4i8, 6i8]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
pub fn test_add_i16() {
    let m1 = CpuTensor::new(vec![1], vec![1i16]).unwrap();
    let m2 = CpuTensor::new(vec![1], vec![1i16]).unwrap();
    let result = CpuTensor::new(vec![1], vec![2i16]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());

    let m1 = CpuTensor::new(vec![1, 2], vec![1i16, 2i16]).unwrap();
    let m2 = CpuTensor::new(vec![1, 2], vec![3i16, 4i16]).unwrap();
    let result = CpuTensor::new(vec![1, 2], vec![4i16, 6i16]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
pub fn test_add_i32() {
    let m1 = CpuTensor::new(vec![1], vec![1]).unwrap();
    let m2 = CpuTensor::new(vec![1], vec![3]).unwrap();
    let result = CpuTensor::new(vec![1], vec![4]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());

    let m1 = CpuTensor::new(vec![1, 2], vec![1, 2]).unwrap();
    let m2 = CpuTensor::new(vec![1, 2], vec![3, 4]).unwrap();
    let result = CpuTensor::new(vec![1, 2], vec![4, 6]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
pub fn test_mul_i32() {
    let m1 = CpuTensor::new(vec![1, 1], vec![1]).unwrap();
    let m2 = CpuTensor::new(vec![1, 1], vec![3]).unwrap();

    assert_eq!(3, (m1 * m2).unwrap().get_data()[0]);

    let m1 = CpuTensor::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    let m2 = CpuTensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    let result = CpuTensor::new(vec![2, 2], vec![19, 22, 43, 50]).unwrap();

    assert_eq!(result, (m1 * m2).unwrap());
}

#[test]
pub fn test_mul_2_cols() {
    let m1 = CpuTensor::new(vec![1, 2], vec![1, 2]).unwrap();
    let m2 = CpuTensor::new(vec![1, 2], vec![3, 4]).unwrap().t().unwrap();

    assert_eq!(11, (m1 * m2).unwrap().get_data()[0]);
}

#[test]
#[should_panic(expected = "ShapeMismatch")]
pub fn test_add_with_shape_mismatch() {
    let m1 = CpuTensor::new(vec![1, 3], vec![1, 2, 5]).unwrap();
    let m2 = CpuTensor::new(vec![1, 2], vec![3, 4]).unwrap();
    let result = CpuTensor::new(vec![1, 2], vec![4, 6]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
#[should_panic(expected = "ShapeMismatch")]
pub fn test_mul_i32_panics() {
    let m1 = CpuTensor::new(vec![2, 1], vec![1, 2]).unwrap();
    let m2 = CpuTensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    let result = CpuTensor::new(vec![2, 2], vec![19, 22, 43, 50]).unwrap();

    assert_eq!(result, (m1 * m2).unwrap());
}

#[test]
pub fn test_mul_i32_reverse() {
    let m1 = CpuTensor::new(vec![1, 2], vec![1, 2]).unwrap();
    let m2 = CpuTensor::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    let result = CpuTensor::new(vec![1, 2], vec![19, 22]).unwrap();

    assert_eq!(result, (m1 * m2).unwrap());
}

#[test]
pub fn test_mul_random() {
    let m1 = CpuTensor::new(
        vec![6, 6],
        vec![
            693, 246, 267, 431, 327, 507, 990, 244, 93, 470, 604, 597, 543, 201, 429, 453, 975,
            101, 993, 319, 440, 617, 172, 669, 224, 98, 622, 826, 752, 855, 946, 105, 613, 352,
            756, 221,
        ],
    )
    .unwrap();
    let m2 = CpuTensor::new(
        vec![6, 6],
        vec![
            738, 55, 22, 872, 611, 303, 365, 543, 283, 374, 951, 798, 150, 866, 398, 272, 583, 470,
            92, 342, 83, 307, 882, 731, 769, 944, 548, 567, 601, 258, 143, 572, 527, 58, 446, 734,
        ],
    )
    .unwrap();
    let result = CpuTensor::new(
        vec![6, 6],
        vec![
            1004890, 1149009, 673288, 1116056, 1615821, 1303342, 1426717, 1339880, 812467, 1501216,
            1934959, 1475992, 1344343, 1643620, 864697, 1363112, 1803598, 1183384, 1199968,
            1364922, 785273, 1430627, 2112552, 1748690, 1070927, 2085626, 1211457, 1130720,
            2154502, 1863808, 1473774, 1600363, 854472, 1580452, 1898626, 1273112,
        ],
    )
    .unwrap();

    assert_eq!(result, (m1 * m2).unwrap());
}

#[test]
pub fn test_mul_random_diff() {
    let m1 = CpuTensor::new(
        vec![6, 2],
        vec![
            9864, 607, 1349, 7440, 593, 8303, 2657, 2637, 6391, 3961, 8372, 7166,
        ],
    )
    .unwrap();
    let m2 = CpuTensor::new(
        vec![2, 6],
        vec![
            6058, 1323, 1540, 5300, 1780, 2944, 9042, 4634, 2527, 8117, 1412, 1906,
        ],
    )
    .unwrap();
    let result = CpuTensor::new(
        vec![6, 6],
        vec![
            65244606, 15862910, 16724449, 57206219, 18415004, 30196558, 75444722, 36261687,
            20878340, 67540180, 12906500, 18152096, 78668120, 39260641, 21894901, 70538351,
            12779376, 17571310, 39939860, 15735069, 10755479, 35486629, 8452904, 12848330,
            74532040, 26810567, 19851587, 66023737, 16968912, 26364770, 115512548, 44283400,
            31001362, 102538022, 25020552, 38305564,
        ],
    )
    .unwrap();

    assert_eq!(result, (m1 * m2).unwrap());
}

#[test]
pub fn test_mul_random_d() {
    let m1 = CpuTensor::new(
        vec![6, 2],
        vec![
            9864, 607, 1349, 7440, 593, 8303, 2657, 2637, 6391, 3961, 8372, 7166,
        ],
    )
    .unwrap();
    let m2 = CpuTensor::new(vec![2, 3], vec![8, 30, 31, 11, 22, 11]).unwrap();
    let result = CpuTensor::new(
        vec![6, 3],
        vec![
            85589, 309274, 312461, 92632, 204150, 123659, 96077, 200456, 109716, 50263, 137724,
            111374, 94699, 278872, 241692, 145802, 408812, 338358,
        ],
    )
    .unwrap();

    assert_eq!(result, (m1 * m2).unwrap());
}

#[test]
pub fn test_mul_float() {
    let m1 = CpuTensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m2 = CpuTensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let result = CpuTensor::new(vec![2, 2], vec![19.0, 22.0, 43.0, 50.0]).unwrap();

    assert_eq!(result, (m1 * m2).unwrap());
}

#[test]
fn test_matmul_identity() {
    let a = CpuTensor::<f32>::new(vec![2, 2], vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]).unwrap();
    let identity =
        CpuTensor::<f32>::new(vec![2, 2], vec![1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32]).unwrap();

    let result = a.matmul(&identity).unwrap();
    assert_eq!(result.get_data(), a.get_data());
}

#[test]
fn test_matmul_vector_dot_product() {
    // (1x3) * (3x1) = (1x1)
    let a = CpuTensor::new(vec![1, 3], vec![1.0_f32, 2.0_f32, 3.0_f32]).unwrap();
    let b = CpuTensor::<f32>::new(vec![3, 1], vec![4.0_f32, 5.0_f32, 6.0_f32]).unwrap();

    let result = a.matmul(&b).unwrap();
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(result.get_data(), vec![32.0_f32]);
    assert_eq!(result.get_shape(), &vec![1, 1]);
}

#[test]
fn test_complex_tensor_mul() {
    let a = Complex::new(1.0, 2.0);
    let b = Complex::new(3.0, 4.0);
    let c = Complex::new(5.0, 6.0);
    let d = Complex::new(7.0, 8.0);

    let m1 = CpuTensor::new(vec![2, 2], vec![a, b, c, d]).unwrap();
    let m2 = CpuTensor::new(vec![2, 2], vec![a, c, b, d]).unwrap();

    let result = (m1 * m2).unwrap();

    let r1 = Complex::new(-10.0, 28.0);
    let r2 = Complex::new(-18.0, 68.0);
    let r3 = Complex::new(-18.0, 68.0);
    let r4 = Complex::new(-26.0, 172.0);

    let expected = CpuTensor::new(vec![2, 2], vec![r1, r2, r3, r4]).unwrap();
    assert_eq!(expected, result);
}

#[test]
fn test_complex_tensor_add() {
    let a = crate::Complex::new(1.0, 2.0);
    let b = crate::Complex::new(3.0, 4.0);
    let c = crate::Complex::new(5.0, 6.0);
    let d = crate::Complex::new(7.0, 8.0);

    let m1 = CpuTensor::new(vec![2, 2], vec![a, b, c, d]).unwrap();
    let m2 = CpuTensor::new(vec![2, 2], vec![a, c, b, d]).unwrap();

    let result = (m1 + m2).unwrap();

    let r1 = crate::Complex::new(2.0, 4.0);
    let r2 = crate::Complex::new(8.0, 10.0);
    let r3 = crate::Complex::new(8.0, 10.0);
    let r4 = crate::Complex::new(14.0, 16.0);

    let expected = CpuTensor::new(vec![2, 2], vec![r1, r2, r3, r4]).unwrap();
    assert_eq!(expected, result);
}

#[test]
pub fn test_scale_float() {
    let m1 = CpuTensor::new(vec![2, 2], vec![1.0, 2.0, -3.0, 4.0]).unwrap();
    let m2 = m1.scale(2.0).unwrap();
    let result = CpuTensor::new(vec![2, 2], vec![2.0, 4.0, -6.0, 8.0]).unwrap();

    println!("Result");
    m2.print_matrix();
    result.print_matrix();

    assert_eq!(result, m2);

    let m1 = CpuTensor::new(vec![1, 4], vec![1.0, 2.0, -3.0, -4.0]).unwrap();
    let m2 = m1.scale(3.0).unwrap();
    let result = CpuTensor::new(vec![1, 4], vec![3.0, 6.0, -9.0, -12.0]).unwrap();

    m2.print_matrix();
    result.print_matrix();

    assert_eq!(result, m2);
}
