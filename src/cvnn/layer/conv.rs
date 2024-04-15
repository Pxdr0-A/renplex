use crate::{act::ComplexActFunc, err::{GradientError, LayerForwardError, LayerInitError}, init::{InitMethod, PredictModel}, input::{IOShape, IOType}, math::{matrix::{Matrix, SliceOps}, BasicOperations, Complex}};

use super::ComplexDerivatives;

#[derive(Debug)]
pub struct ConvCLayer<T> {
  kernels: Vec<Matrix<T>>,
  biases: Matrix<T>,
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
    let mut params: usize = 0;
    for kernel in self.kernels.iter() {
      let kernel_shape = kernel.get_shape();
      params += kernel_shape[0] * kernel_shape[1];
    }

    let bias_shape = self.biases.get_shape();

    (params, bias_shape[0]*bias_shape[1])
  }

  pub fn get_input_shape(&self) -> IOShape {
    let bias_shape = self.biases.get_shape();
    IOShape::Matrix([bias_shape[0], bias_shape[1]])
  }

  pub fn get_output_shape(&self) -> IOShape {
    let bias_shape = self.biases.get_shape();
    IOShape::Matrix([bias_shape[0], bias_shape[1]])
  }

  pub fn init(
    input_shape: IOShape,
    kernel_sizes: Vec<[usize; 2]>,
    func: ComplexActFunc,
    method: InitMethod,
    seed: &mut u128
  ) -> Result<Self, LayerInitError> {

    match input_shape {
      IOShape::Matrix(dim) => {
        let depth = kernel_sizes.len();
        let mut kernels = Vec::with_capacity(depth);
        let mut kernel = Vec::new();
        let mut biases_body = Vec::with_capacity(dim[0] * dim[1]);

        match method {
          InitMethod::Random(scale) => {
            for size in kernel_sizes.into_iter() {
              if (size[0] - 1) / 2 >= dim[0] || (size[1] - 1) / 2 >= dim[1] {
                return Err(LayerInitError::InvalidInputShape)
              }
    
              for _ in 0..size[0] {
                for _ in 0..size[1] {
                  kernel.push(T::gen(seed, scale));
                }
              }

              kernels.push(Matrix::from_body(kernel.clone(), size));
              kernel.drain(..);
            }

            for _ in 0..dim[0] {
              for _ in 0..dim[1] {
                biases_body.push(T::gen(seed, scale));
              }
            }
          }
        }

        Ok(
          Self {
            kernels,
            biases: Matrix::from_body(biases_body, dim),
            func
          }
        )
      },
      _ => { Err(LayerInitError::InvalidInputShape) }
    }
  }

  pub fn trigger(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    match input_type {
      IOType::Matrix(input) => {
        let mut feature_map = input;
        for kernel in self.kernels.iter() {
          feature_map = feature_map.conv(kernel).unwrap();
        }

        feature_map.add_mut(&self.biases).unwrap();

        T::activate_mut(feature_map.get_body_as_mut(), &self.func);

        Ok(IOType::Matrix(feature_map))
      },
      _ => { Err(LayerForwardError::InvalidInput) }
    }
  }

  pub fn forward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    self.trigger(input_type)
  }

  fn trigger_q(&self, input: &Matrix<T>) -> Matrix<T> {
    let mut kernels_iter = self.kernels.iter();
    let mut feature_map = input.conv(kernels_iter.next().unwrap()).unwrap();
    for kernel in kernels_iter {
      feature_map = feature_map.conv(kernel).unwrap();
    }

    feature_map.add_mut(&self.biases).unwrap();

    feature_map
  }

  fn foward_q(&self, input: &Matrix<T>) -> Matrix<T> {
    self.trigger_q(input)
  }

  pub fn compute_derivatives(&self, is_input: bool, previous_act: &IOType<T>, dlda: Vec<T>, dlda_conj: Vec<T>) -> Result<ComplexDerivatives<T>, GradientError> {
    match previous_act {
      IOType::Matrix(input) => {
        let q_shape = input.get_shape();
        let q = match is_input {
          true => { self.trigger_q(input).get_body().to_vec() },
          false => { self.foward_q(input).get_body().to_vec() }
        };

        /* determine dadq, dadq* */
        let mut dadq = q.clone();
        T::d_activate_mut(dadq.as_mut_slice(), &self.func);
        let mut dadq_conj = q;
        T::d_conj_activate_mut(dadq_conj.as_mut_slice(), &self.func);
        let da_conj_dq = dadq_conj
          .iter()
          .map(|elm| { elm.conj() })
          .collect::<Vec<T>>();
        /* derivative of the RELU is just 1, rethink this! */

        /* you can multiply dlda with dadq and dlda_conj with da_conj_dq */
        let dlda_dadq = dlda.mul_slice(&dadq).unwrap();
        let dlda_conj_da_conj_dq = dlda_conj.mul_slice(&da_conj_dq).unwrap();

        /* dQ/dQ' derivatives throughout layer depth */
        let mut step_derivative = vec![T::unit(); q_shape[0]*q_shape[1]];
        let mut step_conj_derivative = vec![T::unit(); q_shape[0]*q_shape[1]];

        /* new dlda derivatives */
        let mut new_dlda = dlda.clone();
        let mut new_dlda_conj = dlda_conj.clone();

        /* the various dqdw */
        let mut dldkernels = self.kernels.clone();
        let depth = dldkernels.len();
        /* you might need to propagate this backwards :( */
        for (kernel_id, (kernel, dk)) in self.kernels.iter().rev().zip(dldkernels.iter_mut().rev()).enumerate() {
          /* intercept the depth of the kernels */
          /* get the input feature and output feature map of the current kernel */
          let mut feature_map = input.clone();
          let mut input_feature_map = input.clone();
          let mut output_feature_map = input.clone();
          for (intercept_id, intercept_kernel) in self.kernels.iter().enumerate() {
            input_feature_map = feature_map.clone();
            feature_map = feature_map.conv(intercept_kernel).unwrap();
            output_feature_map = feature_map.clone();
            
            if intercept_id == depth - kernel_id - 1 {
              break;
            }
          }
          drop(feature_map);

          let kernel_shape = kernel.get_shape();
          let total_elms = kernel_shape[0]*kernel_shape[1];

          /* go through kernel's elements to compute the derivative (with respect to its elements) */
          for (elm_id, dk_elm) in dk.get_body_as_mut().iter_mut().enumerate() {
            let derivative_kernel = Matrix::from_body(
              T::gen_pred(total_elms, elm_id, &PredictModel::Sparse).unwrap(),
              [kernel_shape[0], kernel_shape[1]]
            );

            /* full derivative (matrix) */
            /* this is an extremely easy convolution */
            /* there might be a much much quicker way of doing it */
            /* derivative of the output feature map with respect to the kernel that generated it! */
            /* this is for the element nm (depending at what elm_id we are) */
            let dqdw_nm = input_feature_map
              .conv(&derivative_kernel)
              .unwrap();

            /* you might want to do things here with the full derivative */
            /* put it in the chain rule */
            /* then determine the dldw matrix using the dlda and dlda_conj */
            /* a dot product is involved */
            /* multiply with the step derivatives */
            let dldw_nm = 
              dlda_dadq.mul_slice(&step_derivative).unwrap().scalar_prod(dqdw_nm.get_body()).unwrap() + 
              dlda_conj_da_conj_dq.mul_slice(&step_conj_derivative).unwrap().scalar_prod(dqdw_nm.get_body()).unwrap();

            *dk_elm = dldw_nm
          }

          /* UPDATE new_dlda and new_dlda_conj */
          /* to propagate backwards */
          /* update step derivative */
          /* deeper in the layer */
          /* POTENTIAL ERROR/MISSCALCULATION */
          /* check this update MIGHT BE WRONG */
          step_derivative.mul_slice_mut(Matrix::from_body(new_dlda.clone(), [q_shape[0], q_shape[1]]).conv(&kernel).unwrap().get_body()).unwrap();
          step_conj_derivative.mul_slice_mut(Matrix::from_body(new_dlda_conj.clone(), [q_shape[0], q_shape[1]]).conv(&kernel).unwrap().get_body()).unwrap();
          /* derivative of the loss with respect to Q' (a level deeper) */
          new_dlda.mul_slice_mut(&step_derivative).unwrap();
          new_dlda_conj.mul_slice_mut(&step_conj_derivative).unwrap();
        }
      },
      _ => { panic!("Something went terribily wrong.") }
    }

    unimplemented!()
  }
}