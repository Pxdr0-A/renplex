use crate::{err::LayerForwardError, input::{IOShape, IOType}};

use super::CLayer;

/// Flatten layer cares about the size of the images that it receives.
/// It needs to transfer this information for subsquent layers.
#[derive(Debug)]
pub struct Flatten {
  input_size: ([usize; 2], usize)
}

impl Flatten {
  /// Says if the layer was not initialize. 
  /// 
  /// # Notes
  /// 
  /// This function will soon be deleted.
  pub fn is_empty(&self) -> bool {
    false
  }

  /// Says if the layer propagates derivatives, returning a boolean.
  pub fn propagates(&self) -> bool {
    false
  }

  /// Gives the input shape of the layer
  pub fn get_input_shape(&self) -> IOShape {
    IOShape::Matrix(self.input_size.1)
  }

  /// Gives the output shape of the layer
  pub fn get_output_shape(&self) -> IOShape {
    let matrix_shape = self.input_size.0;

    IOShape::Scalar(matrix_shape[0] * matrix_shape[1] * self.input_size.1)
  }

  /// Creates a flatten layer and returns it.
  /// 
  /// # Arguments
  /// 
  /// * `input_shape` - array with two elements representing the input matrix size.
  /// * `ni` - number of input features.
  pub fn init(input_shape: [usize; 2], ni: usize) -> Flatten {
    Flatten { input_size: (input_shape, ni) }
  }

  /// Returns a [`Result`] for the [`IOType<T>`] related to the prediction of the layer, 
  /// which the flatten version of a matrix or a set of matrices (input features).
  /// Error handling is not yet finished.
  /// 
  /// # Arguments
  /// * `input_type` - a reference to a [`IOType<T>`] representing the input features of the layer.
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

  /// Wraps the flatten layer into the general [`CLayer`] interface.
  pub fn wrap<T>(self) -> CLayer<T> {
    CLayer::Flatten(self)
  }
}