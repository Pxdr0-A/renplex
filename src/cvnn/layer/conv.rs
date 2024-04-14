use crate::{act::ComplexActFunc, err::{LayerForwardError, LayerInitError}, init::InitMethod, input::{IOShape, IOType}, math::{matrix::Matrix, BasicOperations, Complex}};

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
    unimplemented!()
  }

  pub fn forward(&self, input_type: IOType<T>) -> Result<IOType<T>, LayerForwardError> {
    unimplemented!()
  }
}