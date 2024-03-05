use crate::dataset::Dataset;
use crate::input::{IOShape, IOType};
use crate::math::matrix::Matrix;
use crate::math::{BasicOperations, Complex};
use crate::cvnn::layer::CLayer;
use crate::init::InitMethod;
use crate::err::{CostError, ForwardError, LayerAdditionError};

use super::layer::CLayerLike;
use super::Criteria;


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

  pub fn cost(&self, data: Dataset<T, T::Precision>, criteria: Criteria) -> Result<Matrix<T::Precision>, CostError>
    where T::Precision: Copy {
    /* do shape error handling */
    match self.get_input_shape().unwrap() {
      IOShape::Vector(len) => {
        let body_shape = data.body.get_shape();
        let target_shape = data.target.get_shape();
        if len != body_shape[1] { return Err(CostError::IncompatibleDataset) }
        /* check output shape */
        match self.get_output_shape().unwrap() {
          IOShape::Vector(len) => {
            let data_chunks = data.body.get_body().chunks(body_shape[1]);
            let target_chunks = data.target.get_body().chunks(target_shape[1]);
            let mut cost_func = Matrix::with_capacity([data_chunks.len(), len]);
            for (body, target) in data_chunks.zip(target_chunks) {
              cost_func.add_row(
                /* add the output cost */
                match self.forward(IOType::Vector(body.to_vec())).unwrap() {
                  /* calculate cost */
                  IOType::Vector(pred) => { T::cost(&pred[..], target, &criteria) },
                  _ => { return Err(CostError::InconsistentIO) }
                }
              ).unwrap();
            }
            Ok(cost_func)
          },
          IOShape::Matrix(_shape) => { unimplemented!() }
        }
      },
      IOShape::Matrix(_shape) => { unimplemented!() }
    }
  }
}
