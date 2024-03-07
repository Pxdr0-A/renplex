use crate::math::matrix::Matrix;

#[derive(Debug)]
pub enum IOType<T> {
  Vector(Vec<T>),
  Matrix(Matrix<T>)
}

#[derive(Debug)]
pub enum IOShape {
  Vector(usize),
  Matrix([usize; 2])
}
