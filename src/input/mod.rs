use crate::math::matrix::Matrix;

pub enum InputType<T> {
  Vector(Vec<T>),
  Matrix(Matrix<T>)
}

pub enum OutputType<T> {
  Vector(Vec<T>),
  Matrix(Matrix<T>)
}

impl<T: Copy> OutputType<T> {
  pub fn convert(&self) -> InputType<T> {
    match self {
      OutputType::Vector(v) => { InputType::Vector(v.clone()) },
      OutputType::Matrix(m) => { InputType::Matrix(m.clone()) }
    }
  }
}

pub enum InputShape {
  Vector(usize),
  Matrix([usize; 2])
}

pub enum OutputShape {
  Vector(usize),
  Matrix([usize; 2])
}
