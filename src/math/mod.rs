use std::ops::{AddAssign, SubAssign, Add, Mul};
use std::default::Default;

use self::cfloat::{Cf32, Cf64};

pub mod matrix;
pub mod random;
pub mod cfloat;

pub trait BasicOperations<T>: AddAssign + SubAssign + Add<Output=T> + Mul<Output=T> + Default + Copy {}

impl<T, U> BasicOperations<T> for U where U: AddAssign + SubAssign + Add<Output=T> + Mul<Output=T> + Default + Copy {}

pub trait Real {
  fn sigmoid(self) -> Self;
}

impl Real for f32 {
  fn sigmoid(self) -> Self {
    self.exp() / (1.0 + self.exp())
  }
}
impl Real for f64 {
  fn sigmoid(self) -> Self {
    self.exp() / (1.0 + self.exp())
  }
}

pub trait Complex {
  fn rit_sigmoid(self) -> Self;
}

impl Complex for Cf32 {
  fn rit_sigmoid(self) -> Self {
    Cf32 {
      x: self.x.exp() / (1.0 + self.y.exp()),
      y: self.y.exp() / (1.0 + self.y.exp())
    }
  }
}
impl Complex for Cf64 {
  fn rit_sigmoid(self) -> Self {
    Cf64 {
      x: self.x.exp() / (1.0 + self.y.exp()),
      y: self.y.exp() / (1.0 + self.y.exp())
    }
  }
}