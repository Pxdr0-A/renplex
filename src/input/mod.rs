use crate::math::matrix::Matrix;

#[derive(Debug)]
pub enum ReleaseError {
  InvalidType
}

/// Types of input features to wrap the data with for generalization.
#[derive(Debug, Clone)]
pub enum IOType<T> {
  Scalar(Vec<T>),
  Matrix(Vec<Matrix<T>>)
}

impl<T> IOType<T> {
  /// Return a vector for the scalar features. Gives error if it is any other type.
  pub fn release_vec(self) -> Result<Vec<T>, ReleaseError> {
    match self {
      IOType::Scalar(vec) => { Ok(vec) },
      _ => { Err(ReleaseError::InvalidType) }
    }
  }

  /// Returns features maps. Gives error if it is any other type.
  pub fn release_maps(self) -> Result<Vec<Matrix<T>>, ReleaseError> {
    match self {
      IOType::Matrix(mat) => { Ok(mat) },
      _ => { Err(ReleaseError::InvalidType) }
    }
  }

  /// Returns a slice of the scalar input features.
  pub fn as_slice(&self) -> &[T] {
    match self {
      IOType::Scalar(vec) => { &vec[..] },
      IOType::Matrix(_mat) => { panic!("Requesting feature maps as mutable slice.") }
    }
  }

  /// Returns a slice of the scalar input features.
  pub fn as_mut(&mut self) -> &mut [T] {
    match self {
      IOType::Scalar(vec) => { &mut vec[..] },
      IOType::Matrix(_mat) => { panic!("Requesting feature maps as mutable slice.") }
    }
  }
}

impl<T: Copy> IOType<T> {
  /// Copies the scalar input features into a vector and returns them.
  pub fn to_vec(&self) -> Vec<T> {
    match self {
      IOType::Scalar(vec) => { vec.clone() },
      IOType::Matrix(_mat) => { panic!("Requesting feature maps as vec.") }
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

/// Interface for representing input and output shapes with not data attached as opposed to [`IOType`].
#[derive(Debug, PartialEq)]
pub enum IOShape {
  Scalar(usize),
  /// All features maps should have the same dimensions!
  Matrix(usize)
}
