use crate::{act::ComplexActFunc, err::{GradientError, LayerForwardError, LayerInitError}, init::InitMethod, input::{IOShape, IOType}, math::{matrix::Matrix, BasicOperations, Complex}};
use crate::math::matrix::SliceOps;
use super::{CLayer, ComplexDerivatives};

/// Layer that computes only padded convolution.
#[derive(Debug)]
pub struct ConvCLayer<T> {
  input_features_len: usize,
  /// Matrix with shape [number of filters, number of channels]
  kernels: Matrix<Matrix<T>>,
  /// Each output feature map gets its own bias.
  biases: Vec<T>,
  func: ComplexActFunc
}

impl<T: Complex + BasicOperations<T>> ConvCLayer<T> {
  pub fn is_empty(&self) -> bool {
    if self.kernels.get_shape() == &[0, 0] { true }
    else { false }
  }

  pub fn is_trainable(&self) -> bool {
    true
  }

  pub fn params_len(&self) -> (usize, usize) {
    let mut kernel_params: usize = 0;
    for kernel in self.kernels.get_body().iter() {
      let kernel_shape = kernel.get_shape();
      kernel_params += kernel_shape[0] * kernel_shape[1];
    }

    let bias_params = self.biases.len();

    (kernel_params, bias_params)
  }

  /// Technically just gives the shape that the input features should have.
  /// It does not matter how many features you give.
  pub fn get_input_shape(&self) -> IOShape {
    IOShape::FeatureMaps(self.input_features_len)
  }

  pub fn get_output_shape(&self) -> IOShape {
    IOShape::FeatureMaps(self.kernels.get_shape()[0])
  }

  pub fn init(
    input_shape: IOShape,
    filters_len: usize,
    kernel_size: [usize; 2],
    func: ComplexActFunc,
    kernel_method: InitMethod,
    method: InitMethod,
    seed: &mut u128
  ) -> Result<Self, LayerInitError> {

    match input_shape {
      IOShape::FeatureMaps(input_features_len) => {
        let mut kernels = Vec::new();
        let mut biases = Vec::new();

        match kernel_method {
          /* method for the kernels */
          InitMethod::Random(scale) => {
            /* going through the number of pixels */
            for _filter in 0..filters_len {
              for _channel in 0..input_features_len {
                let mut kernel = Vec::new();
                for _ in 0..kernel_size[0] {
                  for _ in 0..kernel_size[1] {
                    kernel.push(T::gen(seed, scale));
                  }
                }
                /* add a channel */
                kernels.push(Matrix::from_body(kernel, kernel_size));
              }
            }
          }
        }

        match method {
          /* method for the biases */
          InitMethod::Random(scale) => {
            for _ in 0..filters_len {
              biases.push(T::gen(seed, scale));
            }
          }
        }

        let kernels = Matrix::from_body(
          kernels, 
          [filters_len, input_features_len]
        );

        Ok(Self { input_features_len, kernels, biases, func })
      },
      _ => { Err(LayerInitError::InvalidInputShape) }
    }
  }

  pub fn forward(&self, input_type: &IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    match input_type {
      IOType::FeatureMaps(input) => {
        let kernels_shape = self.kernels.get_shape();
        let channels = kernels_shape[1];
        let func = &self.func;
        /* it has to be the same for all */
        let filters_shape = self.kernels.elm(0, 0).unwrap().get_shape();
        let input_shape = input[0].get_shape();
        /* output of the convolutions */
        let output_shape = [input_shape[0]-(filters_shape[0]-1), input_shape[1]-(filters_shape[1]-1)];

        let input_features_len = input.len();
        if channels != input_features_len { return Err(LayerForwardError::InvalidInput) }

        let output_features = self.kernels
          .rows_as_iter()
          .zip(self.biases.iter())
          .map(|(filter, bias)| {
            let init_matrix = Matrix::from_body(
              vec![T::default(); output_shape[0]*output_shape[1]], 
              output_shape
            );

            let mut output_feature = input
              .iter()
              .zip(filter.iter())
              .fold(init_matrix, |mut acc, (feature, kernel)| {
                /* going through channels */
                /* CHANGED HERE FOR COMPLEX CONVOLUTION OR NOT */
                let convolved_feature = feature.convolution(kernel).unwrap();
                acc.add_mut(&convolved_feature).unwrap();

                acc
              });

            output_feature.add_mut_scalar(*bias).unwrap();
            let output_body = output_feature.get_body_as_mut();
            T::activate_mut(output_body, func);
            
            output_feature
          }).collect::<Vec<_>>();

        Ok(IOType::FeatureMaps(output_features))
      },
      _ => { Err(LayerForwardError::InvalidInput) }
    }
  }

  pub fn compute_derivatives(&self, previous_act: &IOType<T>, dlda: Vec<T>, dlda_conj: Vec<T>) -> Result<ComplexDerivatives<T>, GradientError> {
    match previous_act {
      IOType::FeatureMaps(input) => {
        let n_input_features = input.len();
        /* all of the shapes of the inputs should be the same */
        /* because previous layers always decrease the size equally throughout features */
        let input_shape = input[0].get_shape();
        let kernels_shape = self.kernels.get_shape();
        let kernel_shape = self.kernels.elm(0, 0).unwrap().get_shape();
        let padx = kernel_shape[0] - 1;
        let pady = kernel_shape[1] - 1;
        let output_shape = [input_shape[0] - padx, input_shape[1] - pady];
        
        let act_func = &self.func;

        let q = self.compute_q(input);

        /*  CHECK IF CALCULATING THE DERIVATIVE OF THE ACTIVATION HERE IS CORRECT */
        /* yap it seems to be correct */
        let mut dlda_dadq = q
          .iter()
          .zip(dlda.into_iter())
          .map(|(elm, dlda_val)| {
            /* dadq * dlda values */
            elm.d_activate(act_func) * dlda_val
          }).collect::<Vec<_>>();

        let mut dlda_conj_da_conj_dq = q
          .iter()
          .zip(dlda_conj.into_iter())
          .map(|(elm, dlda_conj_val)| {
            /* da_conj_dq * dlda_conj values */
            elm.d_conj_activate(act_func).conj() * dlda_conj_val
          }).collect::<Vec<_>>();

        drop(q);

        /* all output features have the same size since previous filters have all the same size */
        let out_feat_size = dlda_dadq.len() / kernels_shape[0];

        /* go through filters to update their derivatives */
        /* each loss gradient chunk is related to a single filter */

        /* can be optimized! */
        let mut dldk = Vec::new();
        let mut dldb = Vec::new();
        let mut new_dlda = vec![T::default(); input_shape[0] * input_shape[1] * n_input_features];
        let mut new_dlda_conj = vec![T::default(); input_shape[0] * input_shape[1] * n_input_features];
        let filters = self.kernels.rows_as_iter();
        filters.for_each(|filter| {
          /* collecting loss derivatives to matrices */
          /* maybe there is a better way */
          let dlda_dadq_feat = Matrix::from_body(
            dlda_dadq.drain(0..out_feat_size).collect::<Vec<_>>(), 
            output_shape
          );
          let dlda_conj_da_conj_dq_feat = Matrix::from_body(
            dlda_conj_da_conj_dq.drain(0..out_feat_size).collect::<Vec<_>>(), 
            output_shape
          );

          /* convolve dlda_feat with all input channels to get kernel derivatives for each channel */
          /* gives a vec that goes through all channels of a single filter */
          /* VERIFY IF THIS IS CORRECT */
          let dldk_per_filter = input
            .into_iter()
            .flat_map(|feature| {
              /* CHANGED HERE FOR COMPLEX CONVOLUTION OR NOT */
              let mut dldk_term1 = feature.convolution(&dlda_dadq_feat).unwrap();
              let dldk_term2 = feature.convolution(&dlda_conj_da_conj_dq_feat).unwrap();
              
              dldk_term1.add_mut(&dldk_term2).unwrap();
              dldk_term1.export_body()
            }).collect::<Vec<_>>();
          
          dldk.extend(dldk_per_filter);

          /* calculate bias derivative */
          let dldb_per_filter = dlda_dadq_feat
            .get_body()
            .add_slice(dlda_conj_da_conj_dq_feat.get_body())
            .unwrap();
          dldb.push(dldb_per_filter.into_iter().reduce(|acc, elm| { acc + elm }).unwrap());

          let fliped_filter = filter
            .iter()
            .map(|kernel| {
              kernel.flip().unwrap()
            })
            .collect::<Vec<_>>();

          /* calculate loss derivative */
          let dlda_padded = dlda_dadq_feat.pad((padx, pady));
          let dlda_conj_padded = dlda_conj_da_conj_dq_feat.pad((padx, pady));
          /* the next convolutions are technically full convolutions */
          /* padded + flipped kernel */
          let new_dlda_acc = fliped_filter
            .iter()
            .flat_map(|flip_kernel| {
              /* CHANGED HERE FOR COMPLEX CONVOLUTION OR NOT */
              dlda_padded
                .convolution(flip_kernel)
                .unwrap()
                .export_body()
            }).collect::<Vec<_>>();
          
          let new_dlda_conj_acc = fliped_filter
            .iter()
            .flat_map(|flip_kernel| {
              /* CHANGED HERE FOR COMPLEX CONVOLUTION OR NOT */
              dlda_conj_padded
                .convolution(flip_kernel)
                .unwrap()
                .export_body()
            }).collect::<Vec<_>>();
          
          new_dlda.add_slice_mut(&new_dlda_acc).unwrap();
          new_dlda_conj.add_slice_mut(&new_dlda_conj_acc).unwrap();
        });

        Ok((dldk, dldb, new_dlda, new_dlda_conj))
      },
      _ => { panic!("Something went terribily wrong.") }
    }
  }

  pub fn neg_conj_adjustment(&mut self, dldw: Vec<T>, dldb: Vec<T>) -> Result<(), GradientError> {
    let dldw_size = dldw.len();
    let dldb_size = dldb.len();
    
    /* if there is an error it can be here */
    let (weights, biases) = self.params_len();

    if dldb_size != biases {
      return Err(GradientError::InconsistentShape)
    } 
    if dldw_size != weights {
      return Err(GradientError::InconsistentShape)
    }

    self.kernels
      .get_body_as_mut()
      .iter_mut()
      .flat_map(|elm| elm.get_body_as_mut())
      .zip(dldw.into_iter())
      .for_each(|(elm, dk)| { *elm -= dk.conj(); });

    self.biases.iter_mut().zip(dldb).for_each(|(bias, db)| {
      *bias -= db.conj();
    });

    Ok(())
  }

  pub fn wrap(self) -> CLayer<T> {
    CLayer::Convolutional(self)
  }

  fn compute_q(&self, input: &Vec<Matrix<T>>) -> Vec<T> {
    /* it has to be the same for all */
    let filters_shape = self.kernels.elm(0, 0).unwrap().get_shape();
    let input_shape = input[0].get_shape();
    /* output of the convolutions */
    let output_shape = [input_shape[0]-(filters_shape[0]-1), input_shape[1]-(filters_shape[1]-1)];

    let output_features_flat = self.kernels
      .rows_as_iter()
      .zip(self.biases.iter())
      .flat_map(|(filter, bias)| {
        let init_matrix = Matrix::from_body(
          vec![T::default(); output_shape[0]*output_shape[1]], 
          output_shape
        );

        let mut output_feature = input
          .iter()
          .zip(filter.iter())
          .fold(init_matrix, |mut acc, (feature, kernel)| {
            /* going through channels */
            /* CHANGED HERE FOR COMPLEX CONVOLUTION OR NOT */
            let convolved_feature = feature.convolution(kernel).unwrap();
            acc.add_mut(&convolved_feature).unwrap();

            acc
          });

        output_feature.add_mut_scalar(*bias).unwrap();
        
        output_feature.export_body()
      }).collect::<Vec<_>>();
    
    output_features_flat
  }
}