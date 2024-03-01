use crate::input::{InputType, OutputType};
use crate::math::{BasicOperations, Real};
use crate::rvnn::layer::Layer;


#[derive(Debug)]
pub enum ForwardError {
  MissingLayers
}

pub struct Network<T> {
  layers: Vec<Layer<T>>,
}

impl<T: Real + BasicOperations<T>> Network<T> {
  pub fn forward(&self, input_type: InputType<T>) -> Result<OutputType<T>, ForwardError> {
    if self.layers.len() == 0 { return Err(ForwardError::MissingLayers) }

    let mut layers_iter = self.layers.iter();
    let input_layer = layers_iter.next().unwrap();
    /* feed input */
    let mut out = input_layer
      .foward(input_type)
      .unwrap();

    for layer_ref in layers_iter {
      /* propagate through hidden layers */
      out = layer_ref
        .foward(out.convert())
        .unwrap();
    }

    Ok(out)
  }
}