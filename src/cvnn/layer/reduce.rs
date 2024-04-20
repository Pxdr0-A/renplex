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
  pub fn is_empty(&self) -> bool {
    false
  }

  pub fn is_trainable(&self) -> bool {
    true
  }

  pub fn params_len(&self) -> (usize, usize) {
    (0, 0)
  }

  pub fn get_input_shape(&self) -> IOShape {
    IOShape::FeatureMaps(self.input_features_len)
  }

  pub fn get_output_shape(&self) -> IOShape {
    IOShape::FeatureMaps(self.input_features_len)
  }

  pub fn init(input_features_len: usize, block_size: [usize; 2], block_func: Box<dyn Fn(&[T]) -> T>, interp_kernel: Matrix<T>) -> Reduce<T> {
    Reduce { input_features_len, block_size, block_func, interp_kernel }
  }

  pub fn trigger(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    match input_type {
      IOType::FeatureMaps(features) => {
        let mut new_features = Vec::with_capacity(features.len());
        for feature in features.into_iter() {
          let new_feature = feature.block_reduce(self.block_size.as_slice(), &self.block_func).unwrap();
          new_features.push(new_feature);
        }

        Ok(IOType::FeatureMaps(new_features))
      },
      _ => { Err(LayerForwardError::InvalidInput) }
    }
  }

  pub fn foward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    self.trigger(input_type)
  }

  pub fn compute_derivatives(&self, _is_input: bool, previous_act: &IOType<T>, dlda: Vec<T>, dlda_conj: Vec<T>) -> Result<ComplexDerivatives<T>, GradientError> {
    /* perform unpooling or upsampling */
    match previous_act {
      IOType::FeatureMaps(features) => {
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
            .zip(self.block_size)
            .map(|(elm, block_dim)| { *elm / block_dim })
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
          
          new_dlda.append(&mut dlda_upsampled.export_body());
          new_dlda_conj.append(&mut dlda_conj_upsampled.export_body());
        }
        drop(dlda_feats); drop(dlda_conj_feats);

        Ok((Vec::new(), Vec::new(), new_dlda, new_dlda_conj))
      },
      _ => { return Err(GradientError::InconsistentShape) }
    }
  }

  pub fn neg_conj_adjustment(&mut self, _dldw: Vec<T>, _dldb: Vec<T>) -> Result<(), GradientError> {
    /* Skip! Nothing to update. */
    
    Ok(())
  }

  pub fn wrap(self) -> CLayer<T> {
    CLayer::Reduce(self)
  }
}