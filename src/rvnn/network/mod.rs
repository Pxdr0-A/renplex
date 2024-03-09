use std::fmt::Debug;

use crate::input::{IOShape, IOType};
use crate::math::{BasicOperations, Real};
use crate::dataset::Dataset;
use crate::rvnn::layer::Layer;
use crate::init::InitMethod;
use crate::err::{ForwardError, LayerAdditionError, LossCalcError};

use super::layer::LayerLike;
use crate::opt::LossFunc;


#[derive(Debug)]
pub struct Network<T> {
  layers: Vec<Layer<T>>,
}

impl<T: Real + BasicOperations<T>> Network<T> {
  pub fn new() -> Network<T> {
    Network {
      layers: Vec::new()
    }
  }

  pub fn get_input_shape(&self) -> Result<IOShape, LayerAdditionError> {
    if self.layers.len() == 0 { return Err(LayerAdditionError::MissingInput) }
    
    Ok(self.layers[0].get_input_shape())
  }

  pub fn get_output_shape(&self) -> Result<IOShape, LayerAdditionError> {
    if self.layers.len() == 0 { return Err(LayerAdditionError::MissingInput) }
    
    Ok(self.layers.last().unwrap().get_output_shape())
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
    mut layer: Layer<T>, 
    input_shape: IOShape, 
    units: usize, 
    method: InitMethod, 
    seed: &mut u128
  ) -> Result<(), LayerAdditionError> {
    
    if self.layers.len() > 0 { return Err(LayerAdditionError::ExistentInput) }

    if layer.is_empty() { 
      /* check what layer is and initialize it */
      match &mut layer {
        Layer::Dense(l) => {
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
    mut layer: Layer<T>, 
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
        Layer::Dense(l) => {
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
      out = layer_ref
        .foward(out)
        .unwrap();
    }

    Ok(out)
  }

  pub fn backward(&mut self, data: Dataset<T, T>, loss_func: LossFunc) -> Result<(), ForwardError> {
    if self.layers.len() <= 1 { return Err(ForwardError::MissingLayers) }

    let (inputs, targets) = data.points_into_iter();
    let mut previous_prediction: IOType<T>;
    let mut current_prediction;
    for (input, target) in inputs.zip(targets) {
      previous_prediction =  input.clone();
      current_prediction = input.clone();
      /* propagate the signal until the last two layers */
      /* decrease the number of layers to go through by one until you reach the input */
      for l in 0..self.layers.len() {
        let mut layers_iter = self.layers
          .iter()
          .rev()
          .skip(l)
          .rev();

        let input_layer = layers_iter.next().unwrap();
        current_prediction = input_layer.trigger(input.clone()).unwrap();
        for layer in layers_iter {
          previous_prediction = current_prediction;
          current_prediction = layer.foward(input.clone()).unwrap();
        }
        /* do the logic that analyzes the last two outputs before updating prediction */
        /* logic that analyzes the previous_prediction versus current_prediction */
        /* you might need information about previous and current layer IO */
        /* calculate general derivatives weights, bias and previous layer */
        /* update parameters? */
      }
    }
    unimplemented!()
  }

  pub fn loss(&self, 
    data: Dataset<T, T>,
    loss_func: LossFunc,
  ) -> Result<Vec<T>, LossCalcError> {

    let mut loss_vals = Vec::with_capacity(data.get_n_points());

    let (input_chunks, target_chunks) = data.points_into_iter();
    let mut prediction;
    for (input, target) in input_chunks.zip(target_chunks) {
      prediction = self
        .forward(input)
        .unwrap();

      loss_vals.push(T::loss(prediction, target, &loss_func).unwrap());
    }

    Ok(loss_vals)
  }
}
