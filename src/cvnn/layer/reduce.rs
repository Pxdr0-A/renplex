use crate::{err::{GradientError, LayerForwardError}, input::{IOShape, IOType}, math::{matrix::{Matrix, SliceToMatrix}, BasicOperations}};

use super::{CLayer, ComplexDerivatives};

pub struct Reduce<T> {
  input_features_len: usize,
  block_size: [usize; 2],
  block_func: Box<dyn Fn(&[T]) -> T>,
  interp_kernel: Matrix<T>
}

impl<T> std::fmt::Debug for Reduce<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Reduce")
      .field("block_size", &self.block_size)
      .field("block_func", &"Fn(&[T]) -> T")
      .finish()
  }
}

impl<T: BasicOperations<T>> Reduce<T> {
  /// Says if the layer was not initialize. 
  /// 
  /// # Notes
  /// 
  /// This function will soon be deleted.
  pub fn is_empty(&self) -> bool {
    false
  }

  /// Says if the layer propagates derivatives, returning a boolean.
  pub fn propagates(&self) -> bool {
    true
  }

  /// Gives the number of parameters involved in the Layer. Which in this case is 0.
  pub fn params_len(&self) -> (usize, usize) {
    (0, 0)
  }

  /// Gives the input shape of the layer
  pub fn get_input_shape(&self) -> IOShape {
    IOShape::Matrix(self.input_features_len)
  }

  /// Gives the output shape of the layer
  pub fn get_output_shape(&self) -> IOShape {
    IOShape::Matrix(self.input_features_len)
  }

  /// Creates a reduce layer and returns it initialized.
  /// 
  /// # Arguments
  /// 
  /// * `input_features_len` - number of input features.
  /// * `block_size` - array of two element indicating the size of the block to reduce the matrix with.
  /// * `block_func` - function that takes a slice of complex numbers representing a flatten block of the
  /// matrix and reduces it to one complex value for image down-sampling.
  /// * `interp_kernel` - kernel that will be involved in the up-sampling of the gradients and conjugate
  /// gradients when back-propagating the derivatives. Usually a mean or a Gaussian kernel, work very well :).
  pub fn init(
    input_features_len: usize, 
    block_size: [usize; 2], 
    block_func: Box<dyn Fn(&[T]) -> T>, 
    interp_kernel: Matrix<T>
  ) -> Reduce<T> {
    Reduce { input_features_len, block_size, block_func, interp_kernel }
  }

  /// Returns a [`Result`] for the [`IOType<T>`] related to the prediction of the layer.
  /// Error handling is not yet finished.
  /// 
  /// # Arguments
  /// * `input_type` - a reference to a [`IOType<T>`] representing the input features of the layer.
  pub fn foward(&self, input_type: &IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    match input_type {
      IOType::Matrix(features) => {
        let mut new_features = Vec::with_capacity(features.len());
        for feature in features.into_iter() {
          let new_feature = feature.block_reduce(
            self.block_size.as_slice(), 
            &self.block_func
          ).unwrap();

          new_features.push(new_feature);
        }

        Ok(IOType::Matrix(new_features))
      },
      _ => { Err(LayerForwardError::InvalidInput) }
    }  
  }

  /// Return a [`Result`] for the derivatives and conjugate derivatives of the layer.
  /// 
  /// # Arguments
  /// * `previous_act` - a reference to a [`IOType<T>`] representing the input features of the layer.
  /// * `dlda` - gradients from an upper layer.
  /// * `dlda_conj` - conjugate gradients from an upper layer.
  pub fn compute_derivatives(&self, previous_act: &IOType<T>, dlda: Vec<T>, dlda_conj: Vec<T>) -> Result<ComplexDerivatives<T>, GradientError> {
    /* perform unpooling or upsampling */
    match previous_act {
      IOType::Matrix(features) => {
        let features_len = features.len();
        let features_chunks = dlda.len() / features_len;

        let mut new_dlda = Vec::new();
        let mut new_dlda_conj = Vec::new();
        /* restore derivatives shape */
        let mut dlda_feats = dlda.chunks(features_chunks);
        let mut dlda_conj_feats = dlda_conj.chunks(features_chunks);
        for feature in features.iter() {
          let original_shape = feature.get_shape();
          let final_shape = original_shape
            .iter()
            .zip(self.block_size.iter())
            .map(|(elm, block_dim)| { *elm / *block_dim })
            .collect::<Vec<usize>>();
          
          let dlda_upsampled = dlda_feats.next().unwrap()
            .to_matrix([final_shape[0], final_shape[1]])
            .unwrap()
            .fractional_upsampling(&self.block_size, &self.interp_kernel)
            .unwrap();

          let dlda_conj_upsampled = dlda_conj_feats.next().unwrap()
            .to_matrix([final_shape[0], final_shape[1]])
            .unwrap()
            .fractional_upsampling(&self.block_size, &self.interp_kernel)
            .unwrap();
          
          new_dlda.extend(dlda_upsampled.export_body());
          new_dlda_conj.extend(dlda_conj_upsampled.export_body());
        }

        Ok((Vec::new(), Vec::new(), new_dlda, new_dlda_conj))
      },
      _ => { return Err(GradientError::InconsistentShape) }
    }
  }

  /// Adjusts the parameters of the layer with negative conjugate.
  /// 
  /// # Arguments
  /// 
  /// * `dldw` - adjustments on the weights.
  /// * `dldb` - adjustments on the biases.
  pub fn neg_conj_adjustment(&mut self, _dldw: Vec<T>, _dldb: Vec<T>) -> Result<(), GradientError> {
    /* Skip! Nothing to update. */
    
    Ok(())
  }

  /// Wraps the reduce layer into the general [`CLayer`] interface.
  pub fn wrap(self) -> CLayer<T> {
    CLayer::Reduce(self)
  }
}