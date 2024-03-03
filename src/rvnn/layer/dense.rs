use crate::act::ActFunc;
use crate::math::matrix::{Matrix, SliceOps};
use crate::math::{BasicOperations, Real};
use crate::input::{IOShape, IOType};

use super::{Layer, LayerForwardError, LayerLike, InitMethod, LayerInitError};

#[derive(Debug)]
pub struct DenseLayer<T> {
  weights: Matrix<T>,
  biases: Vec<T>,
  func: ActFunc
}

/* the activation function is what makes this layer real or complex */
/* the implementations are almost the same tho */
impl<T: Real + BasicOperations<T>> LayerLike<T> for DenseLayer<T> {
  fn is_empty(&self) -> bool {
    if (self.weights.get_shape() == [0_usize, 0]) && (self.biases.len() == 0) {
      true
    } else {
      false
    }
  }

  fn get_input_shape(&self) -> IOShape {
    IOShape::Vector(self.biases.len())
  }

  fn get_output_shape(&self) -> IOShape {
    IOShape::Vector(self.biases.len())
  }

  fn new(func: ActFunc) -> Self {
    DenseLayer {
      weights: Matrix::new(),
      biases: Vec::new(),
      func
    }
  }

  fn init(
    input_shape: IOShape, 
    units: usize, 
    func: ActFunc, 
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
          },
          _ => { return Err(LayerInitError::UnimplementedInitMethod) }
        }

        Ok(
          DenseLayer {
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
            },
            _ => { return Err(LayerInitError::UnimplementedInitMethod); }
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

        /* add the biases to the result */
        res[..]
          .add_slice(&self.biases[..])
          .unwrap();

        /* calculate the activations */
        self.func
          .compute(&mut res[..])
          .unwrap();

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

        /* add the biases to the result */
        res[..]
          .add_slice(&self.biases[..])
          .unwrap();

        /* calculate the activations */
        self.func
          .compute(&mut res[..])
          .unwrap();

        /* layer returns a vector */
        Ok(IOType::Vector(res))
      },
      _ => { Err(LayerForwardError::InvalidInput) }
    }
  }

  fn wrap(self) -> Layer<T> {
    Layer::Dense(self)
  }
}