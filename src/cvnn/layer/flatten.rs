use crate::{err::LayerForwardError, input::{IOShape, IOType}};

use super::CLayer;

/// Flatten layer cares about the size of the images that it receives.
/// It needs to transfer this information for subsquent layers.
#[derive(Debug)]
pub struct Flatten {
  input_size: ([usize; 2], usize)
}

impl Flatten {
  pub fn is_empty(&self) -> bool {
    false
  }

  pub fn is_trainable(&self) -> bool {
    false
  }

  pub fn get_input_shape(&self) -> IOShape {
    IOShape::Matrix(self.input_size.1)
  }

  pub fn get_output_shape(&self) -> IOShape {
    let matrix_shape = self.input_size.0;

    IOShape::Scalar(matrix_shape[0] * matrix_shape[1] * self.input_size.1)
  }

  pub fn init(input_shape: [usize; 2], ni: usize) -> Flatten {
    Flatten { input_size: (input_shape, ni) }
  }

  pub fn foward<T: Clone>(&self, input_type: &IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    match input_type {
      IOType::Matrix(features) => {
        let flatten_features = features
          .into_iter()
          .flat_map(|feature| { feature.get_body().to_vec() })
          .collect::<Vec<T>>();

        Ok(IOType::Scalar(flatten_features))
      },
      _ => { Err(LayerForwardError::InvalidInput) }
    }
  }

  pub fn wrap<T>(self) -> CLayer<T> {
    CLayer::Flatten(self)
  }
}