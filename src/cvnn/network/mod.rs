use crate::dataset::Dataset;
use crate::input::{IOShape, IOType};
use crate::math::matrix::Matrix;
use crate::math::{BasicOperations, Real, Complex};
use crate::cvnn::layer::CLayer;
use crate::init::InitMethod;
use crate::err::{LossCalcError, ForwardError, LayerAdditionError};
use crate::opt::ComplexLossFunc;

use super::layer::CLayerLike;

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

  pub fn intercept(&self, input_type: IOType<T>, index: usize) -> Result<(IOType<T>, &CLayer<T>), ForwardError> {
    if index > self.layers.len() - 1 { return Err(ForwardError::InvalidLayerIndex) }

    let mut layers_iter = self.layers.iter();
    let mut previous_act = input_type.clone();

    let input_layer = layers_iter.next().unwrap();

    if index == 0 {
      return Ok((previous_act, input_layer))
    } else {
      previous_act = input_layer.trigger(input_type.clone()).unwrap();
      let mut current_act;

      /* go through hidden layers */
      for (current_index, layer_ref) in layers_iter.enumerate() {
        current_act = layer_ref
          .foward(previous_act.clone())
          .unwrap();

        /* -1 because the input layer is already gone */
        if current_index == index-1 {
          return Ok((previous_act, layer_ref))
        }

        previous_act = current_act;
      }
    }

    panic!("Something went terribily wrong!");
  }

  pub fn loss(&self, 
    data: Dataset<T, T>,
    loss_func: &ComplexLossFunc,
  ) -> Result<(T::Precision, Vec<T::Precision>), LossCalcError> {

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
      .fold(T::Precision::default(), |acc, elm| { acc + *elm }) / T::Precision::usize_to_real(loss_len);
    
    Ok((mean, loss_vals))
  }

  pub fn gradient_opt(&mut self, data: Dataset<T, T>, loss_func: ComplexLossFunc, lr: T) -> Result<(), ForwardError> {
    let n_layers = self.layers.len();
    if n_layers <= 1 { return Err(ForwardError::MissingLayers) }

    /* main derivatives */
    let mut dldw;
    let mut dldb;

    /* derivatives to accumulate */
    let mut dldw_per_layer = vec![Matrix::new(); n_layers];
    let mut dldb_per_layer = vec![Matrix::new(); n_layers];

    let (inputs, targets) = data.points_into_iter();

    let batch_size = inputs.len();
    let mut is_input: bool;

    /* accumulate weight and bias derivative */
    for (input, target) in inputs.zip(targets) {
      /* initial prediction */
      let initial_pred = self.forward(input.clone()).unwrap();
      /* initial value of loss derivatives */
      let mut dlda = T::d_loss(
        initial_pred.clone(),
        target.clone(), 
        &loss_func
      ).unwrap().to_vec();
      /* initial conjugate derivative of loss */
      let mut dlda_conj: Vec<T> = dlda
        .iter()
        .map(|elm| { elm.conj() })
        .collect();

      /* decrease the number of layers to go through by one until you reach the input */
      /* propagate the derivatives backwards */
      for l in 0..self.layers.len() {
        /* process for getting previous signal of a layer */
        let (previous_act, last_layer) = self
          .intercept(input.clone(), n_layers-l-1)
          .unwrap();

        if !last_layer.is_trainable() {
          /* layer is not trainable, do not waste time */
          continue;
        }

        is_input = if n_layers-l-1 == 0 { true } else { false };
        (dldw, dldb, dlda, dlda_conj) = last_layer.compute_derivatives(is_input, &previous_act, dlda, dlda_conj).unwrap();

        dldw_per_layer[n_layers-l-1].add_mut(&dldw).unwrap();
        dldb_per_layer[n_layers-l-1].add_mut(&dldb).unwrap();
      }
    }

    /* divide the gradient by the count of data samples */
    let scale_param = lr / T::usize_to_complex(batch_size);
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

  pub fn split_gradient_opt(&mut self) {
    unimplemented!()
  }

  pub fn mvn_opt(&mut self) {
    unimplemented!()
  }
}
