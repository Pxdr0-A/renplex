pub mod math;
pub mod dataset;
pub mod layer;
pub mod network;

#[cfg(test)]
mod matrix_tests {
  use super::*;

  #[test]
  fn matrix_ops() {
    use math::matrix::Matrix;

    let mut m1 = Matrix::new([3, 3]);
    m1.add_row(vec![1.0, 2.0, 3.0]).unwrap();
    m1.add_row(vec![4.0, 5.0, 6.0]).unwrap();
    m1.add_row(vec![4.0, 5.0, 6.0]).unwrap();

    let mut m2 = Matrix::new([3, 3]);
    m2.add_row(vec![1.0, 1.0, 1.0]).unwrap();
    m2.add_row(vec![1.0, 1.0, 1.0]).unwrap();
    m2.add_row(vec![1.0, 1.0, 1.0]).unwrap();

    println!("{}", m1);
    println!("{}", m2);

    let m = m1.mul(&m2).unwrap();
    m1.add_mut(m2).unwrap();

    println!("{}", m);
  }
}