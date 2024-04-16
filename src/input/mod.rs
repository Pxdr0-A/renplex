use crate::math::matrix::Matrix;

#[derive(Debug)]
pub enum ReleaseError {
  InvalidType
}

#[derive(Debug, Clone)]
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

  pub fn as_mut(&mut self) -> &mut [T] {
    match self {
      IOType::Vector(vec) => { &mut vec[..] },
      IOType::Matrix(mat) => { mat.get_body_as_mut() }
    }
  }
}

impl<T: Copy> IOType<T> {
  pub fn to_vec(&self) -> Vec<T> {
    match self {
      IOType::Vector(vec) => { vec.clone() },
      IOType::Matrix(mat) => { mat.get_body().to_vec() }
    }
  }
}


pub enum ConnectError {
  InvalidType,
  InvalidLen,
  InvalidSize
}

pub enum TranferError {
  IncompatibleType
}

#[derive(Debug, PartialEq)]
pub enum IOShape {
  Vector(usize),
  Matrix([usize; 2])
}
