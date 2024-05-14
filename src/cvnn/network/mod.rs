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

  pub fn forward(&self, input_type: IOType<T>) -> Result<IOType<T>, ForwardError> {
    if self.layers.len() == 0 { return Err(ForwardError::MissingLayers) }
    
    /* feed input */
    let mut out = input_type;
    let layers_iter = self.layers.iter();
    for layer_ref in layers_iter {
      /* propagate through hidden layers */
      /* there might be a better solution without using convert() */
      out = layer_ref
        .foward(&out)
        .unwrap();
    }

    Ok(out)
  }

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

  pub fn collect_acts(&self, input_type: IOType<T>) -> Result<Vec<IOType<T>>, ForwardError> {
    if self.layers.len() == 0 { return Err(ForwardError::MissingLayers) }
    
    /* feed input */
    let mut outs = vec![input_type];
    let layers_iter = self.layers.iter();
    for layer_ref in layers_iter {
      /* propagate through hidden layers */
      /* there might be a better solution without using convert() */
      outs.push(
        layer_ref
          .foward(outs.last().unwrap())
          .unwrap()
      );
    }

    Ok(outs)
  }

  pub fn loss(&self, 
    data: Dataset<T, T>,
    loss_func: &ComplexLossFunc,
  ) -> Result<T::Precision, LossCalcError> {

    let mut loss_vals = Vec::new();

    let (input_chunks, target_chunks) = data.points_into_iter();
    let mut prediction;
    for (input, target) in input_chunks.zip(target_chunks) {
      prediction = self
        .forward(input)
        .unwrap();
      
      loss_vals.push(T::loss(prediction, target, &loss_func).unwrap());
    }

    let loss_len = loss_vals.len();
    let total = loss_vals
      .into_iter()
      .reduce(|acc, elm| { acc + elm })
      .unwrap();
    let mean = total / T::Precision::usize_to_real(loss_len);
    
    Ok(mean)
  }

  pub fn max_pred_test(&self, data: Dataset<T, T>) -> T::Precision {
    let (input_chunks, target_chunks) = data.points_into_iter();
    let mut prediction;
    let mut pred;
    let mut targ;

    let batch_len = target_chunks.len();
    let mut results = Vec::new();
    for (input, target) in input_chunks.zip(target_chunks) {
      prediction = self
        .forward(input)
        .unwrap();

      /* Potential spot for future optimization. */
      /* But maybe not... This arrays are usually small. */
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

  pub fn gradient_opt(&mut self, data: Dataset<T, T>, loss_func: &ComplexLossFunc, lr: T) -> Result<(), ForwardError> {
    let n_layers = self.layers.len();
    if n_layers <= 1 { return Err(ForwardError::MissingLayers) }

    /* derivatives to accumulate */
    /* maybe they need to be allocated first! */
    /* maybe a get_number_of_params method to the layers */
    let mut dldw_per_layer = Vec::new();
    let mut dldb_per_layer = Vec::new();
    let mut _total_params: usize = 0;
    for layer in self.layers.iter() {
      if layer.is_trainable() {
        let (weights_len, bias_len) = layer.params_len();
      
        dldw_per_layer.push(vec![T::default(); weights_len]);
        dldb_per_layer.push(vec![T::default(); bias_len]);

        _total_params += weights_len + bias_len;
      } else {
        dldw_per_layer.push(Vec::new());
        dldb_per_layer.push(Vec::new());
      }
    }

    let (inputs, targets) = data.points_into_iter();

    let batch_size = inputs.len();
    /* accumulate weight and bias derivative */
    for (input, target) in inputs.zip(targets) {
      let mut activations = self.collect_acts(input).unwrap().into_iter().rev();
      /* initial prediction */
      let initial_pred = activations.next().unwrap();
      /* initial value of loss derivatives */
      let mut dlda = T::d_loss(
        initial_pred,
        target,
        &loss_func
      ).unwrap().to_vec();
      /* initial conjugate derivative of loss */
      let mut dlda_conj: Vec<T> = dlda
        .iter()
        .map(|elm| { elm.conj() })
        .collect();

      for (l, (prev_act, layer)) in activations.zip(self.layers.iter().rev()).enumerate() {
        if layer.is_trainable() {
          let dldw; let dldb;
          (dldw, dldb, dlda, dlda_conj) = layer.compute_derivatives(&prev_act, dlda, dlda_conj).unwrap();

          dldw_per_layer[n_layers-l-1].add_slice_mut(&dldw).unwrap();
          dldb_per_layer[n_layers-l-1].add_slice_mut(&dldb).unwrap(); 
        }
      }

      /* decrease the number of layers to go through by one until you reach the input */
      /* propagate the derivatives backwards */
      /* Maybe it is not worth it to do the intercept everytime!!! */
      /* CORRECT THIS! */
      /*
      for l in 0..n_layers {
        /* process for getting previous signal of a layer */
        /* layer is trainable if it obeys this condition */
        let (previous_act, last_layer) = self
          .intercept(input.clone(), n_layers-l-1)
          .unwrap();

        if last_layer.is_trainable() {
          (dldw, dldb, dlda, dlda_conj) = last_layer.compute_derivatives(&previous_act, dlda, dlda_conj).unwrap();

          dldw_per_layer[n_layers-l-1].add_slice_mut(&dldw).unwrap();
          dldb_per_layer[n_layers-l-1].add_slice_mut(&dldb).unwrap(); 
        }
      }
      */
    }

    /* divide the gradient by the count of data samples */
    let scale_param = lr / T::usize_to_complex(batch_size);
    let update_iter = dldw_per_layer
      .into_iter()
      .zip(dldb_per_layer.into_iter())
      .zip(self.layers.iter_mut());

    update_iter.for_each(|((mut dldw, mut dldb), layer)| {
      if layer.is_trainable() {
        dldw.mul_mut_scalar(scale_param).unwrap();
        dldb.mul_mut_scalar(scale_param).unwrap();

        layer.neg_conj_adjustment(dldw, dldb).unwrap();
      }
    });

    Ok(())
  }

  pub fn split_gradient_opt(&mut self) {
    unimplemented!()
  }

  pub fn mvn_opt(&mut self) {
    unimplemented!()
  }
}
