use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};

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
          .mul_slice(&input[..])
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

  fn compute_q(&self, input: &Vec<T>) -> Vec<T> {
    let mut res = self.weights.mul_slice(input).unwrap();
    res.add_slice_mut(&self.biases).unwrap();

    res
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
  
        /* determine dadq, dadq* */
        let mut dadq = q.clone();
        T::d_activate_mut(&mut dadq[..], &self.func);
        let mut da_conj_dq = q;
        /* this iteration represents dadq_conj */
        T::d_conj_activate_mut(&mut da_conj_dq[..], &self.func);
        da_conj_dq
          .par_iter_mut()
          .for_each(|elm| { *elm = elm.conj() });
        
        /* determine dqdw */
        /* equal to the previous input and the same for all neurons */
        let dqdw = input.clone();

        /* dqdb is 1 */

        /* determine dqda */
        let mut dqda = self.weights.rows_as_iter();

        /* first two commun derivatives */
        /* dlda * dadq and dlda* * da*dq */
        let vals = dlda
          .iter()
          .zip(&dadq[..])
          .map(|(lhs, rhs)| {*lhs * *rhs});
        let vals_conj = dlda_conj
          .iter()
          .zip(&da_conj_dq[..])
          .map(|(lhs, rhs)| {*lhs * *rhs});

        /* current dldw and dldb derivatives */
        let mut dldw = Vec::with_capacity(weight_shape[0] * weight_shape[1]);
        let mut dldb = Vec::with_capacity(weight_shape[0]);

        /* updates on cost function derivatives (propagated backwards) */
        let mut new_dlda: Vec<T> = vec![T::default(); weight_shape[1]];
        let mut new_dlda_conj: Vec<T> = vec![T::default(); weight_shape[1]];

        /* some aux vars */
        let mut addition: Vec<T>;
        
        /* this cycle indirectly goes through the number of neurons */
        let combined_vals = vals.into_iter().zip(vals_conj.into_iter()).enumerate();
        for (index, (val, conj_val)) in combined_vals {
          let mult_func = |elm: &T| { *elm * ( val + conj_val ) };
          
          let range_iter = dqdw
            .par_iter()
            .map(mult_func);

          dldw.append(&mut range_iter.collect::<Vec<T>>());

          /* one neuron bias derivative */
          dldb.push(val + conj_val);

          addition = dqda
            .next()
            .unwrap()
            .par_iter()
            .map(mult_func)
            .collect();

          /* accumulate the sum */
          new_dlda.add_slice_mut(&addition).unwrap();

          addition.par_iter_mut().for_each(|elm| { *elm = elm.conj() });

          /* accumulate the sum */
          new_dlda_conj.add_slice_mut(&addition).unwrap();
        }

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
    self.weights
      .body_as_par_iter_mut()
      .into_par_iter()
      .zip(dldw.into_par_iter())
      .for_each(|(weight, dw)| { *weight -= dw.conj() });
    /* bias update */
    self.biases
      .par_iter_mut()
      .zip(dldb.into_par_iter())
      .for_each(|(bias, db)| { *bias -= db.conj() });

    Ok(())
  }

  pub fn wrap(self) -> CLayer<T> {
    CLayer::Dense(self)
  }
}