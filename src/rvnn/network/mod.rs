use std::fmt::Debug;

use crate::input::{IOShape, IOType};
use crate::math::matrix::Matrix;
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

  /// Forwards a signal through the [`Network`].
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

  pub fn loss(&self, 
    data: Dataset<T, T>,
    loss_func: &LossFunc,
  ) -> Result<(T, Vec<T>), LossCalcError> {

    let mut loss_vals = Vec::with_capacity(data.get_n_points());

    let (input_chunks, target_chunks) = data.points_into_iter();
    let mut prediction;
    for (input, target) in input_chunks.zip(target_chunks) {
      prediction = self
        .forward(input)
        .unwrap();

      loss_vals.push(T::loss(prediction, target, &loss_func).unwrap());
    }

    let loss_len = loss_vals.len();
    let mean = loss_vals
      .iter()
      .fold(T::default(), |acc, elm| { acc + *elm }) / T::usize_to_real(loss_len);
    
    Ok((mean, loss_vals))
  }

  pub fn gradient_opt(&mut self, data: Dataset<T, T>, loss_func: LossFunc, lr: T) -> Result<(), ForwardError> {
    /* check the algo works for one layer */
    if self.layers.len() <= 1 { return Err(ForwardError::MissingLayers) }

    let (inputs, targets) = data.points_into_iter();
    let n_layers = self.layers.len();
    let mut previous_act;
    let mut current_act;
    let mut current_layer;
    let mut dldw;
    let mut dldb;
    /* derivatives to accumulate */
    let mut dldw_per_layer = vec![Matrix::new(); n_layers];
    let mut dldb_per_layer = vec![Matrix::new(); n_layers];
    /* generic counter */
    let mut count = T::default();

    for (input, target) in inputs.zip(targets) {
      previous_act =  input.clone();
      /* initial value of loss derivative */
      let initial_pred = self.forward(input.clone()).unwrap();
      let mut dlda = T::d_loss(
        initial_pred,
        target.clone(), 
        &loss_func
      ).unwrap().to_vec();
      /* decrease the number of layers to go through by one until you reach the input */
      for l in 0..self.layers.len() {
        /* process for getting to adjacent layer signals back to input */
        let mut layers_iter = self.layers
          .iter()
          .rev()
          .skip(l)
          .rev();

        let input_layer = layers_iter.next().unwrap();
        current_layer = Some(input_layer);

        current_act = input_layer.trigger(input.clone()).unwrap();
        for layer in layers_iter {
          previous_act = current_act.clone();

          current_layer = Some(layer);
          current_act = layer.foward(current_act).unwrap();
        }


        /* do the logic that analyzes the last two outputs */
        /* dadq for all of the layer's neurons */
        let layer1 = current_layer.unwrap();
        if !layer1.is_trainable() {
          /* layer is not trainable, do not waste time */
          continue;
        }

        (dldw, dldb, dlda) = layer1.compute_derivatives(&previous_act, dlda).unwrap();

        dldw_per_layer[n_layers-l-1].add_mut(&dldw).unwrap();
        dldb_per_layer[n_layers-l-1].add_mut(&dldb).unwrap();
      }

      count += T::unit();
    }

    /* divide the gradient by the count of data samples */
    let scale_param = lr / count;
    for ((mut dldw_l, mut dldb_l), layer) in dldw_per_layer.into_iter().zip(dldb_per_layer).zip(self.layers.iter_mut()) {
      if !layer.is_trainable() {
        continue;
      }
      
      dldw_l.mul_mut_scalar(scale_param).unwrap();
      dldb_l.mul_mut_scalar(scale_param).unwrap();

      /* update the weights of layer l */
      layer.gradient_adjustment(dldw_l, dldb_l).unwrap();
    }

    Ok(())
  }
}
