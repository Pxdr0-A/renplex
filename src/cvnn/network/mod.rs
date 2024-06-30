//! Module containing operations within CVNN.

use crate::dataset::Dataset;
use crate::input::{IOShape, IOType};
use crate::math::{BasicOperations, Real, Complex};
use crate::math::matrix::SliceOps;
use crate::cvnn::layer::CLayer;
use crate::err::{LossCalcError, ForwardError, LayerAdditionError};
use crate::opt::ComplexLossFunc;


/// Struct that represents a Complex-Valued Neural Network with definable percision.
#[derive(Debug)]
pub struct CNetwork<T> {
  layers: Vec<CLayer<T>>,
}

impl<T: Complex + BasicOperations<T>> CNetwork<T> {
  /// Creates a new empty network.
  pub fn new() -> CNetwork<T> {
    CNetwork {
      layers: Vec::new()
    }
  }

  /// Returns the total number of parameters of the network.
  pub fn params_len(&self) -> usize {
    let mut len = 0;
    for l in self.layers.iter() {
      let layer_len = l.params_len();
      len += layer_len.0 + layer_len.1;
    }
    
    len
  }

  /// Returns the input shape of the network, 
  /// which is the input shape of the layer added with the `add_input()` method.
  pub fn get_input_shape(&self) -> Result<IOShape, LayerAdditionError> {
    if self.layers.len() == 0 { return Err(LayerAdditionError::MissingInput) }
    
    Ok(self.layers[0].get_input_shape())
  }

  /// Returns the output shape of the network, 
  /// which is the output shape of the last layer added with the `add()` method.
  pub fn get_output_shape(&self) -> Result<IOShape, LayerAdditionError> {
    if self.layers.len() == 0 { return Err(LayerAdditionError::MissingInput) }
    
    Ok(self.layers.last().unwrap().get_output_shape())
  }

  /// Adds an empty input [`CLayer`] to the [`CNetwork`] and initializes it.
  ///
  /// # Arguments
  /// 
  /// * `layer` - layer to add already initialized.
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
  pub fn add(&mut self, layer: CLayer<T>) -> Result<(), LayerAdditionError> {
    if layer.is_empty() { return Err(LayerAdditionError::MissingInitialization) }

    /* this already takes care if it is a matrix or a vector */
    /* also takes care of the shape and size respectively */
    /* still the user has to request a coherent amount of units (input, ouput shape) */
    /* the network will not initialize the layer based previous output */
    let last_layer_out = self.layers.last().unwrap().get_output_shape();
    if last_layer_out != layer.get_input_shape() { return Err(LayerAdditionError::IncompatibleIO) }

    self.layers.push(layer);

    Ok(())
  }

  /// Forwards and input throught the network to perform a prediction returning a [`Result`] 
  /// for the respective [`IOType<T>`].
  /// 
  /// # Arguments
  /// * `input_type` - a reference to a [`IOType<T>`] representing the input features of the
  /// input layer.
  pub fn forward(&self, input_type: &IOType<T>) -> Result<IOType<T>, ForwardError> {
    let layers_len = self.layers.len();
    if layers_len == 0 { return Err(ForwardError::MissingLayers) }
    
    /* feed input */    
    let mut layers_iter = self.layers.iter();
    let input_layer = layers_iter.next().unwrap();

    let mut output = input_layer.foward(input_type).unwrap();
    if layers_len > 1 {
      output = layers_iter.fold(output, |acc, layer| {
        layer.foward(&acc).unwrap()
      });
    }

    Ok(output)
  }

  /// Returns the output features of the index-th layer and a reference to that layer.
  /// Error handling is not yet properly implemented.
  /// 
  /// # Arguments
  /// * `input_type` - a reference to a [`IOType<T>`] representing the input features of the input layer.
  /// 
  /// #  Notes
  /// 
  /// Not an heavily used function but might be for obtaining the activations in the
  /// back-propagation algorithm.
  pub fn intercept(&self, input_type: IOType<T>, index: usize) -> Result<(IOType<T>, &CLayer<T>), ForwardError> {
    if index > self.layers.len() - 1 { return Err(ForwardError::InvalidLayerIndex) }
    
    /* go through hidden layers */
    let layers_iter = self.layers.iter();
    let mut previous_act = input_type;
    for (current_index, layer_ref) in layers_iter.enumerate() {
      if current_index == index { return Ok((previous_act, layer_ref)) }

      previous_act = layer_ref
        .foward(&previous_act)
        .unwrap();
    }

    panic!("Something went terribily wrong.");
  }

  /// Returns a vector with all activations of the network from input to output.
  /// Error handling is not yet properly managed.
  /// 
  /// # Arguments
  /// * `input_type` - a reference to a [`IOType<T>`] representing the input features of the input layer.
  pub fn collect_acts(&self, input_type: IOType<T>) -> Result<Vec<IOType<T>>, ForwardError> {
    if self.layers.len() == 0 { return Err(ForwardError::MissingLayers) }
    
    /* feed input */
    let mut outs = vec![input_type];
    let layers_iter = self.layers.iter();
    for layer_ref in layers_iter {
      /* propagate through hidden layers */
      outs.push(
        layer_ref
          .foward(outs.last().unwrap())
          .unwrap()
      );
    }

    Ok(outs)
  }

  /// Returns the loss of a network with respect to a data batch.
  /// 
  /// # Arguments
  /// 
  /// * `data` - batch of data to calculate the loss with.
  /// * `loss_func` - type of loss function to use in the calculation.
  pub fn loss(&self, 
    data: &Dataset<T, T>,
    loss_func: &ComplexLossFunc,
  ) -> Result<T::Precision, LossCalcError> {

    let mut loss_vals = Vec::new();

    let (input_chunks, target_chunks) = data.points_as_iter();
    for (input, target) in input_chunks.zip(target_chunks) {
      let prediction = self
        .forward(input)
        .unwrap();
      
      loss_vals.push(T::loss(target, &prediction, &loss_func).unwrap());
    }

    let loss_len = loss_vals.len();
    let total = loss_vals
      .into_iter()
      .reduce(|acc, elm| { acc + elm })
      .unwrap();
    let mean = total / T::Precision::usize_to_real(loss_len);
    
    Ok(mean)
  }

  /// Returns the accuracy of a network when faced with a batch of data based on 
  /// maximum absolute value.
  /// 
  /// # Arguments
  /// 
  /// * `data` - batch of data to calculate the accuracy with.
  pub fn max_pred_test(&self, data: &Dataset<T, T>) -> T::Precision {
    let (input_chunks, target_chunks) = data.points_as_iter();
    let batch_len = target_chunks.len();

    let mut results = Vec::new();
    for (input, target) in input_chunks.zip(target_chunks) {
      let prediction = self
        .forward(input)
        .unwrap();

      let pred = prediction.as_slice();
      let targ = target.as_slice();

      // finding max index
      let (pred_index, _) = pred
        .into_iter()
        .enumerate()
        .fold((usize::default(), T::default()), |acc, elm| { 
          if *elm.1 > acc.1 { (elm.0, *elm.1) } else { acc } 
        });
      let (targ_index, _) = targ
        .into_iter()
        .enumerate()
        .fold((usize::default(), T::default()), |acc, elm| { 
          if *elm.1 > acc.1 { (elm.0, *elm.1) } else { acc } 
        });

      results.push(if targ_index == pred_index {1_usize} else {0_usize});
    }

    let acc: usize = results.into_iter().sum();

    T::Precision::usize_to_real(acc) / T::Precision::usize_to_real(batch_len)
  }

  /// Optimizes a CVNN with the fully complex back-propagation algorithm for 
  /// gradient-based otpimization.
  /// 
  /// # Arguments
  /// 
  /// * `data` - batch of data to calculate the gradients with.
  /// * `loss_func` - type of loss function to optimize.
  pub fn gradient_opt(&mut self, data: Dataset<T, T>, loss_func: &ComplexLossFunc, lr: T) -> Result<(), ForwardError> {
    let n_layers = self.layers.len();
    if n_layers <= 1 { return Err(ForwardError::MissingLayers) }

    // derivatives to accumulate
    let mut dldw_per_layer = Vec::new();
    let mut dldb_per_layer = Vec::new();
    let mut _total_params: usize = 0;
    // allocate the memory for the gradients
    for layer in self.layers.iter() {
      if layer.propagates() {
        let (weights_len, bias_len) = layer.params_len();
      
        dldw_per_layer.push(vec![T::default(); weights_len]);
        dldb_per_layer.push(vec![T::default(); bias_len]);

        _total_params += weights_len + bias_len;
      } else {
        dldw_per_layer.push(Vec::new());
        dldb_per_layer.push(Vec::new());
      }
    }

    // make an iterator of the data points 
    // (to be used for training)
    let (inputs, targets) = data.points_into_iter();

    let batch_size = inputs.len();
    for (input, target) in inputs.zip(targets) {
      // collect all activations of the network
      // then reverse them to back-propagate values
      let mut activations = self
        .collect_acts(input)
        .unwrap()
        .into_iter()
        .rev();
      
      /* initial prediction */
      let prediction = activations.next().unwrap();
      /* initial value of loss derivatives */
      let mut dlda = T::d_loss(
        &target,
        &prediction, 
        &loss_func
      ).unwrap();
      drop(prediction); drop(target);

      /* initial conjugate derivative of loss */
      let mut dlda_conj: Vec<T> = dlda
        .iter()
        .map(|elm| { elm.conj() })
        .collect();

      // if the layer is trainable or propagates derivatives
      // extract derivatives
      // activation is consumed here iteratively
      // memory starts flushing out
      for (l, (prev_act, layer)) in activations.zip(self.layers.iter().rev()).enumerate() {
        if layer.propagates() {
          let dldw; let dldb;
          // loss and conj loss derivative are being updated
          // derivative computation only needs previous activation
          (dldw, dldb, dlda, dlda_conj) = layer.compute_derivatives(&prev_act, dlda, dlda_conj).unwrap();
          dldw_per_layer[n_layers-l-1].add_slice_mut(&dldw).unwrap();
          dldb_per_layer[n_layers-l-1].add_slice_mut(&dldb).unwrap(); 
        }
      }
      // drop unecessary memory usage
      drop(dlda); drop(dlda_conj);
    }

    // divide the gradient by the count of data samples
    let scale_param = lr / T::usize_to_complex(batch_size);

    // iterator containing the gradients per layer
    // to update layers weights with the gradients
    let update_iter = dldw_per_layer
      .into_iter()
      .zip(dldb_per_layer.into_iter())
      .zip(self.layers.iter_mut());

    update_iter.for_each(|((mut dldw, mut dldb), layer)| {
      if layer.propagates() {
        // scale the gradients with:
        // learning rate 
        // and number of data points in the batch
        dldw.mul_mut_scalar(scale_param).unwrap();
        dldb.mul_mut_scalar(scale_param).unwrap();

        // adjust the weights and biases of that layer
        layer.neg_conj_adjustment(dldw, dldb).unwrap();
      }
    });

    Ok(())
  }

  /// Multiple valued neurons method of optimization. 
  /// Non-gradient based approach that is not yet fully implemented.
  pub fn mvn_opt(&mut self) {
    unimplemented!()
  }
}
