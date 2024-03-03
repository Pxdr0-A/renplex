use crate::input::{IOShape, IOType};
use crate::math::{BasicOperations, Complex};
use crate::cvnn::layer::CLayer;
use crate::init::InitMethod;

use super::layer::CLayerLike;


#[derive(Debug)]
pub enum ForwardError {
  MissingLayers
}

#[derive(Debug)]
pub enum LayerAdditionError {
  MissingInput,
  ExistentInput,
  EarlyInitialization,
  IncompatibleIO
}

#[derive(Debug)]
pub struct CNetwork<T> {
  layers: Vec<CLayer<T>>,
}

impl<T: Complex + BasicOperations<T>> CNetwork<T> {
  pub fn new() -> CNetwork<T> {
    CNetwork {
      layers: Vec::new()
    }
  }

  /// Adds an empty input [`Layer`] to the [`Network`] and initializes it.
  ///
  /// # Arguments
  /// 
  /// * `layer` - Layer to add.
  /// * `input_shape` - Type of input of the layer containing the size. 
  /// Represents the number of weights that each neuron is going to have.
  /// * `units` - Number of neurons, will translate into the output size of the layer.
  /// * `method` - Initialization method
  pub fn add_input(&mut self, 
    mut layer: CLayer<T>, 
    input_shape: IOShape, 
    units: usize, 
    method: InitMethod, 
    seed: &mut u128
  ) -> Result<(), LayerAdditionError> {
    
    if self.layers.len() > 0 { return Err(LayerAdditionError::ExistentInput) }

    if layer.is_empty() { 
      /* check what layer is and initialize it */
      match &mut layer {
        CLayer::Dense(l) => {
          l.init_mut(input_shape, units, method, seed).unwrap();
        }
      }

      /* add input layer */
      self.layers.push(layer);

      Ok(())
    } else {
      Err(LayerAdditionError::EarlyInitialization)
    }
  }

  /// Adds an empty [`Layer`] to the [`Network`] and initializes it based on previous layer. [`InputShape`] is inferred. 
  pub fn add(&mut self, 
    mut layer: CLayer<T>, 
    units: usize,
    method: InitMethod, 
    seed: &mut u128
  ) -> Result<(), LayerAdditionError> {
    
    if self.layers.len() == 0 { return Err(LayerAdditionError::MissingInput) }

    if layer.is_empty() {
      /* check previous number of output (may depend on layer type) */
      let next_input_shape = self.layers
        .last()
        .unwrap()
        .get_output_shape();

      /* check what layer is and initialize it */
      match &mut layer {
        CLayer::Dense(l) => {
          match next_input_shape {
            /* cast the output shape of the previous layer onto the input shape of the next layer */
            IOShape::Vector(size) => { l.init_mut(IOShape::Vector(size), units, method, seed).unwrap(); },
            _ => { return Err(LayerAdditionError::IncompatibleIO) }
          }
        }
      }

      /* add the initialized layer */
      self.layers.push(layer);

      Ok(())
    } else {
      /* throw error (layer must be empty) */
      Err(LayerAdditionError::EarlyInitialization)
    }
  }

  pub fn forward(&self, input_type: IOType<T>) -> Result<IOType<T>, ForwardError> {
    if self.layers.len() == 0 { return Err(ForwardError::MissingLayers) }

    let mut layers_iter = self.layers.iter();
    let input_layer = layers_iter.next().unwrap();
    /* feed input */
    let mut out = input_layer
      .trigger(input_type)
      .unwrap();

    for layer_ref in layers_iter {
      /* propagate through hidden layers */
      /* there might be a better solution without using convert() */
      out = layer_ref
        .foward(out)
        .unwrap();
    }

    Ok(out)
  }
}
