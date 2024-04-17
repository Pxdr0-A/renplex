use crate::{err::LayerForwardError, input::{IOShape, IOType}};

use super::CLayer;

#[derive(Debug)]
pub struct Flatten {
  input_size: [usize; 2]
}

impl Flatten {
  pub fn is_empty(&self) -> bool {
    false
  }

  pub fn is_trainable(&self) -> bool {
    false
  }

  pub fn get_input_shape(&self) -> IOShape {
    IOShape::Matrix(self.input_size)
  }

  pub fn get_output_shape(&self) -> IOShape {
    IOShape::Vector(self.input_size[0] * self.input_size[1])
  }

  pub fn init(input_size: [usize; 2]) -> Flatten {
    Flatten { input_size }
  }

  pub fn trigger<T>(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    match input_type {
      IOType::Matrix(mat) => { Ok(IOType::Vector(mat.export_body())) },
      _ => { Err(LayerForwardError::InvalidInput) }
    }
  }

  pub fn foward<T>(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    self.trigger(input_type)
  }

  pub fn wrap<T>(self) -> CLayer<T> {
    CLayer::Flatten(self)
  }
}