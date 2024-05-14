use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;

use crate::{act::ComplexActFunc, err::{GradientError, LayerForwardError, LayerInitError}, init::InitMethod, input::{IOShape, IOType}, math::{matrix::{Matrix, SliceOps}, BasicOperations, Complex}};
use super::{CLayer, ComplexDerivatives};

/// Layer that computes only padded convolution.
#[derive(Debug)]
pub struct ConvCLayer<T> {
  input_features_len: usize,
  kernels: Vec<Matrix<T>>,
  /// Each output feature map gets its own bias.
  biases: Vec<T>,
  func: ComplexActFunc
}

impl<T: Complex + BasicOperations<T>> ConvCLayer<T> {
  pub fn is_empty(&self) -> bool {
    if self.kernels.len() == 0 { true }
    else { false }
  }

  pub fn is_trainable(&self) -> bool {
    true
  }

  pub fn params_len(&self) -> (usize, usize) {
    let mut kernel_params: usize = 0;
    for kernel in self.kernels.iter() {
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
    IOShape::FeatureMaps(self.input_features_len * self.kernels.len())
  }

  pub fn init(
    input_shape: IOShape,
    kernel_sizes: Vec<[usize; 2]>,
    func: ComplexActFunc,
    kernel_method: InitMethod,
    method: InitMethod,
    seed: &mut u128
  ) -> Result<Self, LayerInitError> {

    match input_shape {
      IOShape::FeatureMaps(input_features_len) => {
        let depth = kernel_sizes.len();
        let output_features_len = input_features_len * depth;

        let mut kernels = Vec::with_capacity(depth);
        let mut biases = Vec::new();

        match kernel_method {
          /* method for the kernels */
          InitMethod::Random(scale) => {
            let mut kernel = Vec::new();
            for size in kernel_sizes.into_iter() {
              for _ in 0..size[0] {
                for _ in 0..size[1] {
                  kernel.push(T::gen(seed, scale));
                }
              }

              kernels.push(Matrix::from_body(kernel.clone(), size));
              kernel.drain(..);
            }
          }
        }

        match method {
          /* method for the biases */
          InitMethod::Random(scale) => {
            for _ in 0..output_features_len {
              biases.push(T::gen(seed, scale));
            }
          }
        }

        Ok(Self { input_features_len, kernels, biases, func })
      },
      _ => { Err(LayerInitError::InvalidInputShape) }
    }
  }

  pub fn forward(&self, input_type: &IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    match input_type {
      IOType::FeatureMaps(input) => {
        let n_feature_maps = input.len();
        let depth = self.kernels.len();
        let n_output_features = n_feature_maps * depth;

        if n_output_features != self.biases.len() {
          return Err(LayerForwardError::InvalidInput)
        }

        /* you can make chunks out of this to zip */
        //let bias_iter = self.biases.chunks(depth);
        let bias_iter = self.biases.par_chunks(depth);

        let out = input
          //.into_iter()
          .into_par_iter()
          .zip(bias_iter)
          .flat_map(|(input_feature_map, biases)| {
            let biases_iter = biases
              //.into_iter();
              .into_par_iter();
            self.kernels
              //.iter()
              .par_iter()
              .zip(biases_iter)
              .map(|(kernel, bias)| {
                let mut output_map = input_feature_map.conv(&kernel).unwrap();

                /* make the chunks */
                output_map.add_mut_scalar(*bias).unwrap();
                T::activate_mut(output_map.get_body_as_mut(), &self.func);
    
                output_map
              }).collect::<Vec<_>>()
          })
          .collect::<Vec<_>>();

        Ok(IOType::FeatureMaps(out))
      },
      _ => { Err(LayerForwardError::InvalidInput) }
    }
  }

  pub fn compute_derivatives(&self, previous_act: &IOType<T>, dlda: Vec<T>, dlda_conj: Vec<T>) -> Result<ComplexDerivatives<T>, GradientError> {
    match previous_act {
      IOType::FeatureMaps(input) => {
        let n_input_features = input.len();
        let depth = self.kernels.len();
        let act_func = &self.func;

        let q = self.compute_q(input);

        let dlda_dadq = q
          .iter()
          .zip(dlda.into_iter())
          .map(|(elm, dlda_val)| {
            /* dadq * dlda values */
            elm.d_activate(act_func) * dlda_val
          }).collect::<Vec<_>>();

        let dlda_conj_da_conj_dq = q
          .iter()
          .zip(dlda_conj.into_iter())
          .map(|(elm, dlda_conj_val)| {
            /* da_conj_dq * dlda_conj values */
            elm.d_conj_activate(act_func).conj() * dlda_conj_val
          }).collect::<Vec<_>>();

        drop(q);

        let chunks = dlda_dadq.len() / n_input_features;
        /* first divide the iterator in ((output_features of input1), (output_features of input2), ...) */
        //let loss_derivatives = dlda_dadq.chunks(chunks).zip(dlda_conj_da_conj_dq.chunks(chunks));
        let loss_derivatives = dlda_dadq.par_chunks(chunks).zip(dlda_conj_da_conj_dq.par_chunks(chunks));
        let data = input
          //.into_iter()
          .into_par_iter()
          .zip(loss_derivatives)
          /* transfer this for_each also to map */
          .map(|(input_feature, (out_feats_dldq, out_feats_dldq_conj))| {
            /* go through the depth of the layer */
            /* divide out_feats_dldq and out_feats_dldq_conj in <depth> chunks and zip it! */
            /* each feature has <depth> output features */
            let size = out_feats_dldq.len() / depth;
            //let out_feats_derivative = out_feats_dldq.chunks(size).zip(out_feats_dldq_conj.chunks(size));
            let out_feats_derivative = out_feats_dldq.par_chunks(size).zip(out_feats_dldq_conj.par_chunks(size));
            let data = self.kernels
              //.iter()
              .par_iter()
              .zip(out_feats_derivative)
              .map(|(kernel, (dlda_dadq, dlda_conj_da_conj_dq))| {
                let kernel_shape = kernel.get_shape();

                let mut dldk_body = Vec::new();
                for i in 0..kernel_shape[0] {
                  for j in 0..kernel_shape[1] {
                    let dqdk_nm = input_feature
                      .dconv((i, j), kernel_shape)
                      .unwrap();

                    /* MIGHT BE SCALAR PRODUCT ONLY WITH DLDA */
                    let elm = dlda_dadq.scalar_prod(dqdk_nm.get_body()).unwrap();
                    let elm_conj = dlda_conj_da_conj_dq.scalar_prod(dqdk_nm.get_body()).unwrap();

                    dldk_body.push(elm + elm_conj);
                  }
                }

                /* it is implicitly multiplied by 1 */
                let dldb_feat = dlda_dadq
                  .iter()
                  .zip(dlda_conj_da_conj_dq.iter())
                  .fold(T::default(), |acc, (lhs, rhs)| { acc + (*lhs + *rhs) });
                let dldb_feat = vec![dldb_feat];

                /* loss derivatives wtr previous activation */
                /* perform backward convolution (flip kernel or reverse order) */
                /* derivative of the current input feature with respect to the respective output feature */
                /* input feature whose output feature comes from the current kernel */
                /* THIS DERIVATIVE MIGHT BE WRONG! Verify! */
                let new_dqda = input_feature.deconv(&kernel).unwrap();
                let new_dqda_body = new_dqda.get_body();
                let lhs = dlda_dadq.mul_slice(new_dqda_body).unwrap();
                let rhs = dlda_conj_da_conj_dq.mul_slice(new_dqda_body).unwrap();

                let new_dlda_feat_update = lhs.add_slice(&rhs).unwrap();
                /* maybe this is an unecessary transition */
                //lhs.iter_mut().for_each(|elm| { *elm = elm.conj(); });
                //rhs.iter_mut().for_each(|elm| { *elm = elm.conj(); });
                let new_dlda_conj_feat_update = new_dlda_feat_update
                  .iter()
                  .map(|elm| { elm.conj() })
                  .collect::<Vec<_>>();

                /* should return the kernels derivative and the respective feature update derivatives */
                (dldk_body, dldb_feat, new_dlda_feat_update, new_dlda_conj_feat_update)
              }).collect::<Vec<_>>();

            /* par reduce will desync the vectors */
            let (dldk, dldb, new_dlda_feat, new_dlda_conj_feat) = data
              .into_iter()
              .reduce(|mut acc, elm| {
                acc.0.extend(elm.0);
                acc.1.extend(elm.1);
                (acc.0,  acc.1, acc.2.add_slice(&elm.2).unwrap(), acc.3.add_slice(&elm.3).unwrap())
            }).unwrap();

            /* needs to return new_dlda_feat (respective to each input feature) */
            /* every dldk_d respective to each input feature (they need to be reduced along the input feature "axis") */
            (dldk, dldb, new_dlda_feat, new_dlda_conj_feat)
        }).collect::<Vec<_>>();

        let res = data
          .into_iter()
          .reduce(|mut acc, elm| {
            // bias derivative
            acc.1.extend(elm.1);
            // dlda derivative
            acc.2.extend(elm.2);
            // dlda_conj derivative
            acc.3.extend(elm.3);

            (acc.0.add_slice(&elm.0).unwrap(), acc.1, acc.2, acc.3)
        }).unwrap();

        Ok(res)
      },
      _ => { panic!("Something went terribily wrong.") }
    }
  }

  pub fn neg_conj_adjustment(&mut self, dldw: Vec<T>, dldb: Vec<T>) -> Result<(), GradientError> {
    let dldw_size = dldw.len();
    let dldb_size = dldb.len();
    
    /* if there is an error it can be here */
    let (weights, biases) = self.params_len();

    if dldb_size != biases{
      return Err(GradientError::InconsistentShape)
    } 
    if dldw_size != weights {
      return Err(GradientError::InconsistentShape)
    }

    self.kernels
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
    let depth = self.kernels.len();

    /* you can make chunks out of this to zip */
    //let bias_iter = self.biases.chunks(depth);
    let bias_iter = self.biases.par_chunks(depth);
    
    input
      //.into_iter()
      .into_par_iter()
      .zip(bias_iter)
      .flat_map(|(input_feature_map, biases)| {
        self.kernels
          //.iter()
          .par_iter()
          .zip(biases)
          .flat_map(|(kernel, bias)| {
            let mut output_map = input_feature_map.conv(&kernel).unwrap();

            /* make the chunks */
            output_map.add_mut_scalar(*bias).unwrap();

            output_map.export_body()
          })
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>()
  }
}