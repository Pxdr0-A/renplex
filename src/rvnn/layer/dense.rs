use crate::act::ActFunc;
use crate::math::matrix::{Matrix, SliceOps};
use crate::math::{BasicOperations, Real};
use crate::input::{InputType, OutputType};

use super::{LayerError, LayerLike};

pub struct DenseLayer<T> {
  weights: Matrix<T>,
  biases: Vec<T>,
  func: ActFunc
}

/* the activation function is what makes this layer real or complex */
/* the implementations are almost the same tho */
impl<T: Real + BasicOperations<T>> LayerLike<T> for DenseLayer<T> {

  fn forward(&self, input_type: InputType<T>) -> Result<OutputType<T>, LayerError> {
    match input_type {
      /* dense layer should receive a vector */
      InputType::Vector(input) => {
        /* instantiate the result (it is going to be a column matrix) */
        let mut res = self.weights
          .mul_slice(&input[..])
          .unwrap();

        /* add the biases to the result */
        res[..]
          .add_slice(&self.biases[..])
          .unwrap();

        /* calculate the activations */
        self.func
          .act(&mut res[..])
          .unwrap();

        /* layer returns a vector */
        Ok(OutputType::Vector(res))
      },
      _ => { Err(LayerError::InvalidInput) }
    }
  }
}