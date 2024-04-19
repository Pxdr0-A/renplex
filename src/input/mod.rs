use crate::math::matrix::Matrix;

#[derive(Debug)]
pub enum ReleaseError {
  InvalidType
}

#[derive(Debug, Clone)]
pub enum IOType<T> {
  Vector(Vec<T>),
  FeatureMaps(Vec<Matrix<T>>)
}

impl<T> IOType<T> {
  pub fn release_vec(self) -> Result<Vec<T>, ReleaseError> {
    match self {
      IOType::Vector(vec) => { Ok(vec) },
      _ => { Err(ReleaseError::InvalidType) }
    }
  }

  pub fn release_maps(self) -> Result<Vec<Matrix<T>>, ReleaseError> {
    match self {
      IOType::FeatureMaps(mat) => { Ok(mat) },
      _ => { Err(ReleaseError::InvalidType) }
    }
  }

  pub fn as_mut(&mut self) -> &mut [T] {
    match self {
      IOType::Vector(vec) => { &mut vec[..] },
      IOType::FeatureMaps(_mat) => { panic!("Requesting feature maps as mutable slice.") }
    }
  }
}

impl<T: Copy> IOType<T> {
  pub fn to_vec(&self) -> Vec<T> {
    match self {
      IOType::Vector(vec) => { vec.clone() },
      IOType::FeatureMaps(_mat) => { panic!("Requesting feature maps as vec.") }
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
  /// All features maps should have the same dimensions!
  FeatureMaps(usize)
}
