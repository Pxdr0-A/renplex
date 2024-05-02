use std::sync::Mutex;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};

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

  pub fn forward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    match input_type {
      IOType::FeatureMaps(input) => {
        let n_feature_maps = input.len();
        let depth = self.kernels.len();
        let n_output_features = n_feature_maps * depth;

        if n_output_features != self.biases.len() {
          return Err(LayerForwardError::InvalidInput)
        }

        let bias_iter = Mutex::new(self.biases.iter());
        let output_feature_maps = Mutex::new(Vec::new());
        input
          .into_par_iter()
          .for_each(|input_feature_map| {
            self.kernels.par_iter().for_each(
              |kernel| {
                let mut output_map = input_feature_map.conv(&kernel).unwrap();
    
                let bias = bias_iter.lock().unwrap().next().unwrap();
                output_map.add_mut_scalar(*bias).unwrap();
                
                T::activate_mut(output_map.get_body_as_mut(), &self.func);
    
                output_feature_maps.lock().unwrap().push(output_map);
              }
            );
          });

        Ok(IOType::FeatureMaps(Mutex::into_inner(output_feature_maps).unwrap()))
      },
      _ => { Err(LayerForwardError::InvalidInput) }
    }
  }

  pub fn compute_derivatives(&self, previous_act: &IOType<T>, dlda: Vec<T>, dlda_conj: Vec<T>) -> Result<ComplexDerivatives<T>, GradientError> {
    match previous_act {
      IOType::FeatureMaps(input) => {
        let q = self.compute_q(input);

        let n_output_features = q.len();

        let mut dadq = q.clone();
        for feature in dadq.iter_mut() {
          let body = feature.get_body_as_mut();
          T::d_activate_mut(body, &self.func);
        }
        let mut dadq_conj = q;
        for feature in dadq_conj.iter_mut() {
          let body = feature.get_body_as_mut();
          T::d_conj_activate_mut(body, &self.func);
        }
        let mut da_conj_dq = dadq_conj.clone();
        for feature in da_conj_dq.iter_mut() {
          feature
            .get_body_as_mut()
            .iter_mut()
            .for_each(|elm| { *elm = elm.conj() });
        }

        let dadq = dadq.into_iter().map(|feature| { feature.export_body() }).flatten().collect::<Vec<T>>();
        let da_conj_dq = da_conj_dq.into_iter().map(|feature| { feature.export_body() }).flatten().collect::<Vec<T>>();

        let dlda_dadq = dlda.mul_slice(&dadq).unwrap();
        drop(dlda); drop(dadq);
        let dlda_conj_da_conj_dq = dlda_conj.mul_slice(&da_conj_dq).unwrap();
        drop(dlda_conj); drop(da_conj_dq);

        /* initializing dldw */
        let mut dldk_d = Vec::new();
        for kernel in self.kernels.iter() {
          let kernel_shape = kernel.get_shape();
          let kernel_len = kernel_shape[0] * kernel_shape[1];
          dldk_d.push(Matrix::from_body(vec![T::default(); kernel_len], [kernel_shape[0], kernel_shape[1]]));
        }

        /* kernels derivatives (through the depth) */
        let dldk_d = Mutex::new(dldk_d);
        let dldb = Mutex::new(Vec::new());
        /* initializing backprogation derivatives */
        let mut new_dlda = Vec::new();
        let mut new_dlda_conj = Vec::new();

        let chunks = dlda_dadq.len() / n_output_features;
        let loss_derivatives = Mutex::new(dlda_dadq.chunks(chunks).zip(dlda_conj_da_conj_dq.chunks(chunks)));

        let input_iter = input.into_iter();
        input_iter.for_each(|input_feature| {
          let input_feature_shape = input_feature.get_shape();
          let input_feature_len = input_feature_shape[0] * input_feature_shape[1];
          
          let new_dlda_feat_update = Mutex::new(vec![T::unit(); input_feature_len]);
          let new_dlda_conj_feat_update = Mutex::new(vec![T::unit(); input_feature_len]);

          /* go through the depth of the layer */
          self.kernels.par_iter().zip(dldk_d.lock().unwrap().par_iter_mut())
            .for_each(|(kernel, dldk)| {
              /* output feature respective to the kernel and current input feature */
              let (dlda_dadq, dlda_conj_da_conj_dq) = loss_derivatives.lock().unwrap().next().unwrap();

              let kernel_shape = kernel.get_shape();
              for (index, dldk_update) in dldk.get_body_as_mut().iter_mut().enumerate() {
                /* go through all kernel points */
                let pos = (index / kernel_shape[0], index % kernel_shape[1]);
                let dqdk_nm = input_feature
                  .dconv(pos, kernel_shape)
                  .unwrap();

                /* MIGHT BE SCALAR PRODUCT ONLY WITH DLDA */
                let elm = dlda_dadq.scalar_prod(dqdk_nm.get_body()).unwrap();
                let elm_conj = dlda_conj_da_conj_dq.scalar_prod(dqdk_nm.get_body()).unwrap();

                *dldk_update += elm + elm_conj;
              }

              /* update dlda and dlda_conj */
              /* perform backward convolution (flip kernel or reverse order) */
              /* derivative of the current input feature with respect to the respective output feature */
              /* input feature whose output feature comes from the current kernel */
              let dldb_per_feature = dlda_dadq
                .add_slice(dlda_conj_da_conj_dq)
                .unwrap()
                .into_iter()
                .reduce(|acc, elm| { acc + elm })
                .unwrap();
              dldb.lock().unwrap().push(dldb_per_feature);

              /* previous activation */
              /* THIS DERIVATIVE MIGHT BE WRONG! Verify! */
              let new_dqda = input_feature.deconv(&kernel).unwrap();
              let new_dqda_body = new_dqda.get_body();
              let mut lhs = dlda_dadq.mul_slice(new_dqda_body).unwrap();
              let mut rhs = dlda_conj_da_conj_dq.mul_slice(new_dqda_body).unwrap();
              
              new_dlda_feat_update.lock().unwrap().add_slice_mut(&lhs.add_slice(&rhs).unwrap()).unwrap();
              lhs.iter_mut().for_each(|elm| { *elm = elm.conj() });
              rhs.iter_mut().for_each(|elm| { *elm = elm.conj() });
              new_dlda_conj_feat_update.lock().unwrap().add_slice_mut(&lhs.add_slice(&rhs).unwrap()).unwrap();
            });

          /* update derivatives respective to the input feature */
          new_dlda.extend(Mutex::into_inner(new_dlda_feat_update).unwrap());
          new_dlda_conj.extend(Mutex::into_inner(new_dlda_conj_feat_update).unwrap());
        });

        /* consume mutexes */
        let dldk_d = Mutex::into_inner(dldk_d)
          .unwrap()
          .into_iter()
          .map(|feature| { feature.export_body() })
          .flatten()
          .collect::<Vec<T>>();
        let dldb = Mutex::into_inner(dldb).unwrap();

        Ok((dldk_d, dldb, new_dlda, new_dlda_conj))
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
    
    /* part that could use optimization */
    /* verify if the order is correct */
    let mut dldw_iter = dldw.into_iter();
    for kernel in self.kernels.iter_mut() {
      for elm in kernel.get_body_as_mut().iter_mut() {
        *elm -= dldw_iter.next().unwrap().conj();
      }
    }

    for (bias, db) in self.biases.iter_mut().zip(dldb) {
      *bias -= db.conj();
    }

    Ok(())
  }

  pub fn wrap(self) -> CLayer<T> {
    CLayer::Convolutional(self)
  }

  fn compute_q(&self, input: &Vec<Matrix<T>>) -> Vec<Matrix<T>> {
    let bias_iter = Mutex::new(self.biases.iter());
    let output_feature_maps = Mutex::new(Vec::new());
    input
      .into_par_iter()
      .for_each(|input_feature_map| {
        self.kernels.par_iter().for_each(
          |kernel| {
            let mut output_map = input_feature_map.conv(&kernel).unwrap();

            let bias = bias_iter.lock().unwrap().next().unwrap();
            output_map.add_mut_scalar(*bias).unwrap();
            
            output_feature_maps.lock().unwrap().push(output_map);
          }
        );
      });
    
    Mutex::into_inner(output_feature_maps).unwrap()
  }
}