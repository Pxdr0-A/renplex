use crate::act::ComplexActFunc;
use crate::math::matrix::{Matrix, SliceOps};
use crate::math::{BasicOperations, Complex};
use crate::input::{IOShape, IOType};
use crate::init::InitMethod;
use crate::err::GradientError;

use super::{CLayer, LayerForwardError, CLayerLike, LayerInitError};

#[derive(Debug)]
pub struct DenseCLayer<T> {
  weights: Matrix<T>,
  biases: Vec<T>,
  func: ComplexActFunc
}

/* the activation function is what makes this layer real or complex */
/* the implementations are almost the same tho */
impl<T: Complex + BasicOperations<T>> CLayerLike<T> for DenseCLayer<T> {
  fn is_empty(&self) -> bool {
    if (self.weights.get_shape() == [0_usize, 0]) && (self.biases.len() == 0) {
      true
    } else {
      false
    }
  }

  fn is_trainable(&self) -> bool {
    true
  }

  fn get_input_shape(&self) -> IOShape {
    let weight_shape = self.weights.get_shape();
    IOShape::Vector(weight_shape[0] * weight_shape[1])
  }

  fn get_output_shape(&self) -> IOShape {
    IOShape::Vector(self.biases.len())
  }

  fn new(func: ComplexActFunc) -> Self {
    DenseCLayer {
      weights: Matrix::new(),
      biases: Vec::new(),
      func
    }
  }

  fn init(
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

  /// Initializes a [`DenseLayer`] from an empty one.
  fn init_mut(&mut self, 
    input_shape: IOShape, 
    units: usize, 
    method: InitMethod, 
    seed: &mut u128
  ) -> Result<(), LayerInitError> {
    
    if self.is_empty() {
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
    
              self.weights = Matrix::from_body(body, [units, inputs]);
              self.biases = biases;
            }
          }
        },
        _ => { return Err(LayerInitError::InvalidInputShape); }
      }
    } else {
      return Err(LayerInitError::AlreadyInitialized);
    }

    Ok(())
  }

  fn trigger(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    match input_type {
      /* dense layer should receive a vector */
      IOType::Vector(input) => {
        let shape = self.weights.get_shape();

        if input.len() != shape[0] * shape[1] { return Err(LayerForwardError::InvalidInput) }

        /* instantiate the result (it is going to be a column matrix) */
        let mut res = Vec::with_capacity(input.len());

        /* go through units */
        for row in 0..shape[0] {
          res.push(
            self.weights
              .row(row)
              .unwrap()
              .iter()
              .zip(&input[row*shape[1]..row*shape[1]+shape[1]])
              .fold(T::default(), |acc, (weight, input)| { acc + *weight * *input })
          );
        }
        let res_mut = &mut res[..];

        /* add the biases to the result */
        res_mut
          .add_slice(&self.biases[..])
          .unwrap();

        /* calculate the activations */
        T::activate_mut(res_mut, &self.func);

        /* layer returns a vector */
        Ok(IOType::Vector(res))
      },
      _ => { Err(LayerForwardError::InvalidInput) }
    }  
  }

  fn forward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
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
          .add_slice(&self.biases[..])
          .unwrap();

        /* calculate the activations */
        T::activate_mut(res_mut, &self.func);

        /* layer returns a vector */
        Ok(IOType::Vector(res))
      },
      _ => { Err(LayerForwardError::InvalidInput) }
    }
  }

  fn compute_derivatives(&self, is_input: bool, previous_act: &IOType<T>, dlda: Vec<T>, dlda_conj: Vec<T>) -> Result<(Matrix<T>, Matrix<T>, Vec<T>, Vec<T>), GradientError> {
    /* check dimensions of every matrix and vector */

    let weight_shape = self.weights.get_shape();
    if dlda.len() != weight_shape[0] { return Err(GradientError::InconsistentShape) }
    if dlda_conj.len() != weight_shape[0] { return Err(GradientError::InconsistentShape) }

    match previous_act {
      IOType::Vector(input) => {
        /* determine q (it is an holomorphic function) */
        /* determine q */
        let q = match is_input {
          true => { 
            let mut res = Vec::with_capacity(weight_shape[0]);

            /* go through units */
            for row in 0..weight_shape[0] {
              res.push(
                self.weights
                  .row(row)
                  .unwrap()
                  .iter()
                  .zip(&input[row*weight_shape[1]..row*weight_shape[1]+weight_shape[1]])
                  .fold(T::default(), |acc, (weight, input)| { acc + *weight * *input })
              );
            }
            res.add_slice(&self.biases).unwrap();

            res
          },
          false => { 
            let mut res = self.weights.mul_vec(input.clone()).unwrap();
            res.add_slice(&self.biases).unwrap();

            res
          }
        };
  
        /* determine dadq, dadq* */
        let mut dadq = q.clone();
        T::d_activate_mut(&mut dadq[..], &self.func);
        let mut dadq_conj = q;
        T::d_conj_activate_mut(&mut dadq_conj[..], &self.func);
        let da_conj_dq = dadq_conj
          .iter()
          .map(|elm| { elm.conj() })
          .collect::<Vec<T>>();
        
        /* determine dqdw and dq*dw */
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
        let mut dldw = Matrix::with_capacity([weight_shape[0], weight_shape[1]]);
        let mut dldb = Vec::with_capacity(weight_shape[0]);

        /* updates on cost function derivatives (propagated backwards) */
        let mut new_dlda: Vec<T> = vec![T::default(); weight_shape[1]];
        let mut new_dlda_conj: Vec<T> = vec![T::default(); weight_shape[1]];

        /* some aux vars */
        let mut current_dqda_row: &[T];
        let mut addition: Vec<T>;
        
        /* this cycle indirectly goes through the number of neurons */
        for (index, (val, conj_val)) in vals.into_iter().zip(vals_conj).enumerate() {
          if is_input {
            dldw.add_row(
              dqdw[index*weight_shape[1]..index*weight_shape[1]+weight_shape[1]]
                .iter()
                .map(|elm| { *elm * ( val + conj_val ) })
                .collect::<Vec<T>>()
            ).unwrap();
          } else {
            dldw.add_row(
              dqdw
                .iter()
                /* goes through columns or number of previous neurons */
                .map(|elm| { *elm * ( val + conj_val ) })
                .collect::<Vec<T>>()
            ).unwrap();
          }

          /* one neuron bias derivative */
          dldb.push(val + conj_val);

          current_dqda_row = dqda
            .next()
            .unwrap();
          addition = current_dqda_row
            .iter()
            .map(|elm| { *elm * ( val + conj_val ) })
            .collect();
          /* accumulate the sum */
          new_dlda.add_slice(&addition).unwrap();
          for elm in addition.iter_mut() {
            /* happens to be like this */
            *elm = elm.conj();
          }
          /* accumulate the sum */
          new_dlda_conj.add_slice(&addition).unwrap();
        }

        Ok((dldw, Matrix::from_body(dldb, [weight_shape[0], 1]), new_dlda, new_dlda_conj))
      },
      _ => { panic!("Something went terribily wrong.") }
    }
  }

  fn gradient_adjustment(&mut self, dldw: Matrix<T>, dldb: Matrix<T>) -> Result<(), GradientError> {
    let weight_shape = self.weights.get_shape();
    let dldw_shape = dldw.get_shape();
    let dldb_shape = dldb.get_shape();
    if dldb_shape[1] != 1 {
      return Err(GradientError::InvalidBiasShape)
    } 
    if dldb_shape[0] != self.biases.len(){
      return Err(GradientError::InconsistentShape)
    } 
    if dldw_shape != weight_shape {
      return Err(GradientError::InconsistentShape)
    }
    
    for (weights, dw_slice) in self.weights.rows_as_iter_mut().zip(dldw.rows_as_iter()) {
      for (weight, dw) in weights.into_iter().zip(dw_slice) {
        /* see why it is the conjugate */
        *weight -= dw.conj();
      }
    }

    for (bias, db) in self.biases.iter_mut().zip(dldb.get_body()) {
      *bias -= db.conj();
    }

    Ok(())
  }

  fn wrap(self) -> CLayer<T> {
    CLayer::Dense(self)
  }
}