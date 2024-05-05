use crate::act::ComplexActFunc;
use crate::math::matrix::{Matrix, SliceOps};
use crate::math::{BasicOperations, Complex};
use crate::input::{IOShape, IOType};
use crate::init::InitMethod;
use crate::err::GradientError;

use super::{CLayer, ComplexDerivatives, LayerForwardError, LayerInitError};

#[derive(Debug)]
pub struct DenseCLayer<T> {
  weights: Matrix<T>,
  biases: Vec<T>,
  func: ComplexActFunc
}

/* the activation function is what makes this layer real or complex */
/* the implementations are almost the same tho */
/* all layers should have more or less the same implementations */
impl<T: Complex + BasicOperations<T>> DenseCLayer<T> {
  pub fn is_empty(&self) -> bool {
    if (self.weights.get_shape() == [0_usize, 0]) || (self.biases.len() == 0) {
      true
    } else {
      false
    }
  }

  pub fn is_trainable(&self) -> bool {
    true
  }

  pub fn params_len(&self) -> (usize, usize) {
    let shape = self.weights.get_shape();

    (shape[0] * shape[1], self.biases.len())
  }

  pub fn get_input_shape(&self) -> IOShape {
    let weight_shape = self.weights.get_shape();
    IOShape::Vector(weight_shape[1])
  }

  pub fn get_output_shape(&self) -> IOShape {
    IOShape::Vector(self.biases.len())
  }

  pub fn init(
    input_shape: IOShape, 
    units: usize, 
    func: ComplexActFunc, 
    method: InitMethod, 
    seed: &mut u128
  ) -> Result<Self, LayerInitError> {

    match input_shape {
      IOShape::Vector(inputs) => {
        let mut body = Vec::with_capacity(units * inputs);
        let mut biases = Vec::with_capacity(units);

        match method {
          InitMethod::Random(scale) => {
            for _ in 0..units {
              for _ in 0..inputs {
                body.push(T::gen(seed, scale));
              }
              biases.push(T::gen(seed, scale));
            }
          }     
        }

        Ok(
          DenseCLayer {
            weights: Matrix::from_body(body, [units, inputs]),
            biases,
            func
          }
        )
      },
      _ => { Err(LayerInitError::InvalidInputShape) }
    }
  }

  pub fn forward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    match input_type {
      /* dense layer should receive a vector */
      IOType::Vector(input) => {
        /* instantiate the result (it is going to be a column matrix) */
        let mut res = self.weights
          .mul_vec(input)
          .unwrap();

        let res_mut = &mut res[..];

        /* add the biases to the result */
        res_mut
          .add_slice_mut(&self.biases[..])
          .unwrap();

        /* calculate the activations */
        T::activate_mut(res_mut, &self.func);

        /* layer returns a vector */
        Ok(IOType::Vector(res))
      },
      _ => { Err(LayerForwardError::InvalidInput) }
    }
  }

  pub fn compute_derivatives(&self, previous_act: &IOType<T>, dlda: Vec<T>, dlda_conj: Vec<T>) -> Result<ComplexDerivatives<T>, GradientError> {
    /* check dimensions of every matrix and vector */

    let weight_shape = self.weights.get_shape();
    if dlda.len() != weight_shape[0] { return Err(GradientError::InconsistentShape) }
    if dlda_conj.len() != weight_shape[0] { return Err(GradientError::InconsistentShape) }

    match previous_act {
      IOType::Vector(input) => {
        /* determine q (it is an holomorphic function) */
        /* determine q */
        let q = self.compute_q(input);
        
        let act_func = &self.func;
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
        
        /* determine dqdw */
        /* equal to the previous input and the same for all neurons */
        let dqdw = input;

        /* dqdb is 1 */

        /* determine dqda */
        let dqda = self.weights.rows_as_iter();

        /* current dldw and dldb derivatives */
        let mut dldw = Vec::with_capacity(weight_shape[0] * weight_shape[1]);
        let mut dldb = Vec::with_capacity(weight_shape[0]);

        /* updates on cost function derivatives (propagated backwards) */
        let mut new_dlda: Vec<T> = vec![T::default(); weight_shape[1]];
        
        /* this cycle indirectly goes through the number of neurons */
        for ((val, conj_val), dqda_row) in dlda_dadq.into_iter().zip(dlda_conj_da_conj_dq).zip(dqda) {
          let mult_func = |elm: &T| { *elm * ( val + conj_val ) };
          
          let dldw_row = dqdw
            .iter()
            .map(mult_func)
            .collect::<Vec<_>>();

          dldw.extend(dldw_row);

          /* one neuron bias derivative */
          dldb.push(val + conj_val);

          /* prevsious layer */
          let dlda = dqda_row
            .iter()
            .map(mult_func)
            .collect::<Vec<_>>();

          /* accumulate the sum */
          new_dlda.add_slice_mut(&dlda).unwrap();
        }

        let new_dlda_conj = new_dlda
          .iter()
          .map(|elm| { elm.conj() })
          .collect::<Vec<_>>();

        Ok((dldw, dldb, new_dlda, new_dlda_conj))
      },
      _ => { panic!("Something went terribily wrong.") }
    }
  }

  pub fn neg_conj_adjustment(&mut self, dldw: Vec<T>, dldb: Vec<T>) -> Result<(), GradientError> {
    let dldw_size = dldw.len();
    let dldb_size = dldb.len();
    
    /* if there is an error it can be here */
    let weights = self.weights.get_body_as_mut();
    let n_weights = weights.len();

    if dldb_size != self.biases.len(){
      return Err(GradientError::InconsistentShape)
    } 
    if dldw_size != n_weights {
      return Err(GradientError::InconsistentShape)
    }
    
    /* weights update */
    weights
      .iter_mut()
      .zip(dldw.into_iter())
      .for_each(|(weight, dw)| { *weight -= dw.conj() });
    /* bias update */
    self.biases
      .iter_mut()
      .zip(dldb.into_iter())
      .for_each(|(bias, db)| { *bias -= db.conj() });

    Ok(())
  }

  pub fn wrap(self) -> CLayer<T> {
    CLayer::Dense(self)
  }

  fn compute_q(&self, input: &Vec<T>) -> Vec<T> {
    let mut res = self.weights.mul_slice(input).unwrap();
    res.add_slice_mut(&self.biases).unwrap();

    res
  }
}