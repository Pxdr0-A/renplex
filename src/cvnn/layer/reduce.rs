use crate::{err::LayerForwardError, input::{IOShape, IOType}, math::BasicOperations};

use super::CLayer;

pub struct Reduce<T> {
  input_features_len: usize,
  block_size: [usize; 2],
  block_func: Box<dyn Fn(&[T]) -> T>
}

impl<T> std::fmt::Debug for Reduce<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Reduce")
      .field("block_size", &self.block_size)
      .field("block_func", &"Fn(&[T]) -> T")
      .finish()
  }
}

impl<T: BasicOperations<T>> Reduce<T> {
  pub fn is_empty(&self) -> bool {
    false
  }

  pub fn is_trainable(&self) -> bool {
    false
  }

  pub fn get_input_shape(&self) -> IOShape {
    IOShape::FeatureMaps(self.input_features_len)
  }

  pub fn get_output_shape(&self) -> IOShape {
    IOShape::FeatureMaps(self.input_features_len)
  }

  pub fn init(input_features_len: usize, block_size: [usize; 2], block_func: Box<dyn Fn(&[T]) -> T>) -> Reduce<T> {
    Reduce { input_features_len, block_size, block_func }
  }

  pub fn trigger(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    match input_type {
      IOType::FeatureMaps(features) => {
        let mut new_features = Vec::with_capacity(features.len());
        for feature in features.into_iter() {
          let new_feature = feature.block_reduce(self.block_size.as_slice(), &self.block_func).unwrap();
          new_features.push(new_feature);
        }

        Ok(IOType::FeatureMaps(new_features))
      },
      _ => { Err(LayerForwardError::InvalidInput) }
    }
  }

  pub fn foward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    self.trigger(input_type)
  }

  pub fn wrap(self) -> CLayer<T> {
    CLayer::Reduce(self)
  }
}