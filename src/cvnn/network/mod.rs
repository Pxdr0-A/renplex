use crate::dataset::Dataset;
use crate::input::{IOShape, IOType};
use crate::math::{BasicOperations, Real, Complex};
use crate::math::matrix::SliceOps;
use crate::cvnn::layer::CLayer;
use crate::err::{LossCalcError, ForwardError, LayerAdditionError};
use crate::opt::ComplexLossFunc;


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
  pub fn add_input(&mut self, layer: CLayer<T>) -> Result<(), LayerAdditionError> {
    
    if self.layers.len() > 0 { return Err(LayerAdditionError::ExistentInput) }

    if !layer.is_empty() { 
      /* add input layer */
      self.layers.push(layer);

      Ok(())
    } else {
      Err(LayerAdditionError::MissingInitialization)
    }
  }

  /// Adds an already initialized layer.
  /// Layers need to be always initialized individually since that always have different specifications.
  pub fn add(&mut self, layer: CLayer<T>) {
    /* check if is is empty */
    /* if it is, return error */
    /* also return error if shapes do not match */
    unimplemented!()
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

  pub fn max_pred_test(&self, data: Dataset<T, T>) -> T::Precision {
    let (input_chunks, target_chunks) = data.points_into_iter();
    let mut prediction;
    let mut pred;
    let mut targ;

    let batch_len = target_chunks.len();
    let mut results = Vec::with_capacity(batch_len);
    for (input, target) in input_chunks.zip(target_chunks) {
      prediction = self
        .forward(input)
        .unwrap();

      pred = prediction.release_vec().unwrap();
      targ = target.release_vec().unwrap();
      let (pred_index, _) = pred
        .into_iter()
        .enumerate()
        .fold((usize::default(), T::default()), |acc, elm| { 
          if elm.1 > acc.1 { elm } else { acc } 
        });

      let (targ_index, _) = targ
        .into_iter()
        .enumerate()
        .fold((usize::default(), T::default()), |acc, elm| { 
          if elm.1 > acc.1 { elm } else { acc } 
        });

      results.push(if targ_index == pred_index {1_usize} else {0_usize});
    }

    let acc: usize = results.into_iter().sum();

    T::Precision::usize_to_real(acc) / T::Precision::usize_to_real(batch_len)
  }

  pub fn gradient_opt(&mut self, data: Dataset<T, T>, loss_func: ComplexLossFunc, lr: T) -> Result<(), ForwardError> {
    let n_layers = self.layers.len();
    if n_layers <= 1 { return Err(ForwardError::MissingLayers) }

    /* main derivatives */
    let mut dldw;
    let mut dldb;

    /* derivatives to accumulate */
    /* maybe they need to be allocated first! */
    /* maybe a get_number_of_params method to the layers */
    let mut dldw_per_layer = Vec::with_capacity(n_layers);
    let mut dldb_per_layer = Vec::with_capacity(n_layers);
    let mut total_params: usize = 0;
    for layer in self.layers.iter() {
      if layer.is_trainable() {
        let (weights_len, bias_len) = layer.params_len();
      
        dldw_per_layer.push(vec![T::default(); weights_len]);
        dldb_per_layer.push(vec![T::default(); bias_len]);

        total_params += weights_len + bias_len;
      } else {
        dldw_per_layer.push(Vec::new());
        dldb_per_layer.push(Vec::new());
      }
    }

    println!("Total Number of Parameters: {}", total_params);

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

        /* solve the empty vec problem */
        dldw_per_layer[n_layers-l-1].add_slice(&dldw).unwrap();
        dldb_per_layer[n_layers-l-1].add_slice(&dldb).unwrap();
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
      layer.neg_conj_adjustment(dldw_l, dldb_l).unwrap();
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
