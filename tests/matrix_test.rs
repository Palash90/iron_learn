use iron_learn::{init_context, Matrix};

#[test]
#[should_panic(expected = "MatrixShapeError")]
fn test_new_panic_on_shape() {
    Matrix::new(vec![1, 2, 3], vec![1, 2, 3]).unwrap();
}

#[test]
pub fn add_i8() {
    let m1 = Matrix::<i8>::new(vec![1, 2], vec![1i8, 2i8]).unwrap();
    let m2 = Matrix::new(vec![1, 2], vec![3i8, 4i8]).unwrap();
    let result = Matrix::new(vec![1, 2], vec![4i8, 6i8]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
pub fn add_i16() {
    let m1 = Matrix::new(vec![1, 2], vec![1i16, 2i16]).unwrap();
    let m2 = Matrix::new(vec![1, 2], vec![3i16, 4i16]).unwrap();
    let result = Matrix::new(vec![1, 2], vec![4i16, 6i16]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
pub fn add_i32() {
    let m1 = Matrix::new(vec![1, 2], vec![1, 2]).unwrap();
    let m2 = Matrix::new(vec![1, 2], vec![3, 4]).unwrap();
    let result = Matrix::new(vec![1, 2], vec![4, 6]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
#[should_panic(expected = "ShapeMismatch")]
pub fn add_with_shape_mismatch() {
    let m1 = Matrix::new(vec![1, 3], vec![1, 2, 5]).unwrap();
    let m2 = Matrix::new(vec![1, 2], vec![3, 4]).unwrap();
    let result = Matrix::new(vec![1, 2], vec![4, 6]).unwrap();

    assert_eq!(result, (m1 + m2).unwrap());
}

#[test]
pub fn mul_i32() {
    let _context = cust::quick_init().unwrap();

    init_context("Test", 1, true, Some(_context));

    let m1 = Matrix::new(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    let m2 = Matrix::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    let result = Matrix::new(vec![2, 2], vec![19, 22, 43, 50]).unwrap();

    assert_eq!(result, (m1 * m2).unwrap());
}

#[test]
#[should_panic(expected = "ShapeMismatch")]
pub fn mul_i32_panics() {
    let m1 = Matrix::new(vec![2, 1], vec![1, 2]).unwrap();
    let m2 = Matrix::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    let result = Matrix::new(vec![2, 2], vec![19, 22, 43, 50]).unwrap();

    assert_eq!(result, (m1 * m2).unwrap());
}

#[test]
pub fn mul_i32_reverse() {
    let m1 = Matrix::new(vec![1, 2], vec![1, 2]).unwrap();
    let m2 = Matrix::new(vec![2, 2], vec![5, 6, 7, 8]).unwrap();
    let result = Matrix::new(vec![1, 2], vec![19, 22]).unwrap();

    assert_eq!(result, (m1 * m2).unwrap());
}

#[test]
pub fn mul_random() {
    let m1 = Matrix::new(
        vec![6, 6],
        vec![
            693, 246, 267, 431, 327, 507, 990, 244, 93, 470, 604, 597, 543, 201, 429, 453, 975,
            101, 993, 319, 440, 617, 172, 669, 224, 98, 622, 826, 752, 855, 946, 105, 613, 352,
            756, 221,
        ],
    )
    .unwrap();
    let m2 = Matrix::new(
        vec![6, 6],
        vec![
            738, 55, 22, 872, 611, 303, 365, 543, 283, 374, 951, 798, 150, 866, 398, 272, 583, 470,
            92, 342, 83, 307, 882, 731, 769, 944, 548, 567, 601, 258, 143, 572, 527, 58, 446, 734,
        ],
    )
    .unwrap();
    let result = Matrix::new(
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
pub fn mul_random_diff() {
    let m1 = Matrix::new(
        vec![6, 2],
        vec![
            9864, 607, 1349, 7440, 593, 8303, 2657, 2637, 6391, 3961, 8372, 7166,
        ],
    )
    .unwrap();
    let m2 = Matrix::new(
        vec![2, 6],
        vec![
            6058, 1323, 1540, 5300, 1780, 2944, 9042, 4634, 2527, 8117, 1412, 1906,
        ],
    )
    .unwrap();
    let result = Matrix::new(
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
pub fn mul_random_d() {
    let m1 = Matrix::new(
        vec![6, 2],
        vec![
            9864, 607, 1349, 7440, 593, 8303, 2657, 2637, 6391, 3961, 8372, 7166,
        ],
    )
    .unwrap();
    let m2 = Matrix::new(vec![2, 3], vec![8, 30, 31, 11, 22, 11]).unwrap();
    let result = Matrix::new(
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
pub fn mul_float() {
    let _context = cust::quick_init().unwrap();
    init_context("Test", 1, true, Some(_context));

    let m1 = Matrix::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let m2 = Matrix::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let result = Matrix::new(vec![2, 2], vec![19.0, 22.0, 43.0, 50.0]).unwrap();

    assert_eq!(result, (m1 * m2).unwrap());
}
