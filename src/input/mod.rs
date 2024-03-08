use crate::math::matrix::Matrix;

#[derive(Debug)]
pub enum ReleaseError {
  InvalidType
}

#[derive(Debug)]
pub enum IOType<T> {
  Vector(Vec<T>),
  Matrix(Matrix<T>)
}

impl<T> IOType<T> {
  pub fn release_vec(self) -> Result<Vec<T>, ReleaseError> {
    match self {
      IOType::Vector(vec) => { Ok(vec) },
      _ => { Err(ReleaseError::InvalidType) }
    }
  }

  pub fn release_mat(self) -> Result<Matrix<T>, ReleaseError> {
    match self {
      IOType::Matrix(mat) => { Ok(mat) },
      _ => { Err(ReleaseError::InvalidType) }
    }
  }
}

#[derive(Debug)]
pub enum IOShape {
  Vector(usize),
  Matrix([usize; 2])
}
