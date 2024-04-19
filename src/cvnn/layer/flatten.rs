use crate::{err::LayerForwardError, input::{IOShape, IOType}};

use super::CLayer;

/// Flatten layer cares about the size of the images that it receives.
/// It needs to transfer this information for subsquent layers.
#[derive(Debug)]
pub struct Flatten {
  input_size: Vec<[usize; 2]>
}

impl Flatten {
  pub fn is_empty(&self) -> bool {
    false
  }

  pub fn is_trainable(&self) -> bool {
    false
  }

  pub fn get_input_shape(&self) -> IOShape {
    IOShape::FeatureMaps(self.input_size.len())
  }

  pub fn get_output_shape(&self) -> IOShape {
    let flatten_len = self.input_size
      .iter()
      .map(|elm| { elm[0] * elm[1] })
      .reduce(|acc, elm| { acc + elm })
      .unwrap();
    
    IOShape::Vector(flatten_len)
  }

  pub fn init(input_size: Vec<[usize; 2]>) -> Flatten {
    Flatten { input_size }
  }

  pub fn trigger<T>(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    match input_type {
      IOType::FeatureMaps(features) => {
        let flatten_features = features
          .into_iter()
          .map(|feature| { feature.export_body() })
          .flatten()
          .collect::<Vec<T>>();

        Ok(IOType::Vector(flatten_features))
      },
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