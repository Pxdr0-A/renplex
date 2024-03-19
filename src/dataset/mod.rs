mod err;

use std::{fmt::{Debug, Display}, slice::Iter, vec::IntoIter};

use crate::{init::PredictModel, input::IOType, math::{matrix::Matrix, random::lcgi, BasicOperations, Complex, Real}};
use err::DatasetSampleError;
//use crate::math::random::{lcgf32, lcgf64};

/// A very low-level and simple dataset representation.
/// Body contains the input that will be feed directly onto a Network.
/// Target contains the expected output to be directly used in calculating the cost function.
#[derive(Debug, Clone)]
pub struct Dataset<B, T> {
    inputs: Vec<IOType<B>>,
    /// Represents the intended output of the network, i.e. the output layer desired results.
    target: Vec<IOType<T>>
}

impl<T: Display> Dataset<T, T> {
  pub fn to_csv(&self) -> std::io::Result<()> {
    unimplemented!()
  }
}

impl<B, T> Dataset<B, T> {
  pub fn get_n_points(&self) -> usize {
    self.inputs.len()
  }

  pub fn get_point(&self, i: usize) -> (&IOType<B>, &IOType<T>) {
    (&self.inputs[i], &self.target[i])
  }

  pub fn points_as_iter(&self) -> (Iter<'_, IOType<B>>, Iter<'_, IOType<T>>) {
    (self.inputs.iter(), self.target.iter())
  }

  pub fn points_into_iter(self) -> (IntoIter<IOType<B>>, IntoIter<IOType<T>>) {
    (self.inputs.into_iter(), self.target.into_iter())
  }

  pub fn add_point(&mut self, point: (IOType<B>, IOType<T>))  {
    self.inputs.push(point.0);
    self.target.push(point.1);
  }
}

impl<T: Real + BasicOperations<T>> Dataset<T, T> {
  pub fn gen_centers(features: usize, degree: usize, scale: usize, seed: &mut u128) -> Matrix<T> {
    // spray focal points
    let mut centers: Matrix<T> = Matrix::with_capacity([degree, features]);
    let mut center: Vec<T> = Vec::with_capacity(features);
    for _ in 0..degree {
      for _ in 0..features {
        center.push(
          // random point relative to origin
          T::gen(seed, scale)
        );
      }

      // add_row will clean the center vector
      centers.add_mut_row(&mut center).unwrap();
    }

    centers
  }

  /// Samples a [`Dataset`] with real values with vector inputs and vector outputs.
  /// This artificial data has rectangular-like symmetry.
  pub fn sample(
    dims: [usize; 2],
    degree: usize,
    macro_scale: usize,
    micro_scale: usize,
    pred_method: PredictModel,
    seed: &mut u128
  ) -> Result<Dataset<T, T>, DatasetSampleError> {

    if dims[0] < degree { return Err(DatasetSampleError::BellowMinimumSamples) }
      
    // spray focal points
    let centers = Dataset::<T, T>::gen_centers(dims[1], degree, macro_scale, seed);
    

    let mut class_center: &[T];
    let mut selected_class: usize;
    let mut one_hot_vec: Vec<T>;
    let mut sample_body: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut labels: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut added_row: Vec<T> = Vec::with_capacity(dims[1]);
    for _ in 0..dims[0] {
      selected_class = lcgi(seed, degree as u128);
      one_hot_vec = T::gen_pred(degree, selected_class, &pred_method).unwrap();

      class_center = centers.row(selected_class).unwrap();
      for col in 0..dims[1] {
        added_row.push(
          class_center[col] + T::gen(seed, micro_scale)
        );
      }
      
      // add_row will clean the added_row vec
      sample_body.push(IOType::Vector(added_row.clone()));
      labels.push(IOType::Vector(one_hot_vec.clone()));

      added_row.drain(..);
    }

    Ok(
      Dataset {
        inputs: sample_body,
        target: labels
      }
    )
  }

  pub fn gen_batch_sample(
    dims: [usize; 2],
    centers: Matrix<T>,
    micro_scale: usize,
    pred_method: PredictModel,
    seed: &mut u128
  ) -> Result<Dataset<T, T>, DatasetSampleError> {
    
    let centers_shape = centers.get_shape();
    let degree = centers_shape[0];
    let _features = centers_shape[1];
    let mut class_center: &[T];
    let mut selected_class: usize;
    let mut one_hot_vec: Vec<T>;
    let mut sample_body: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut labels: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut added_row: Vec<T> = Vec::with_capacity(dims[1]);
    for _ in 0..dims[0] {
      selected_class = lcgi(seed, degree as u128);
      one_hot_vec = T::gen_pred(degree, selected_class, &pred_method).unwrap();

      class_center = centers.row(selected_class).unwrap();
      for col in 0..dims[1] {
        added_row.push(
          class_center[col] + T::gen(seed, micro_scale)
        );
      }
      
      // add_row will clean the added_row vec
      sample_body.push(IOType::Vector(added_row.clone()));
      labels.push(IOType::Vector(one_hot_vec.clone()));

      added_row.drain(..);
    }

    Ok(
      Dataset {
        inputs: sample_body,
        target: labels
      }
    )
  }
}

impl<T: Complex + BasicOperations<T>> Dataset<T, T> {
  pub fn gen_complex_centers(features: usize, degree: usize, scale: usize, seed: &mut u128) -> Matrix<T> {
    // spray focal points
    let mut centers: Matrix<T> = Matrix::with_capacity([degree, features]);
    let mut center: Vec<T> = Vec::with_capacity(features);
    for _ in 0..degree {
      for _ in 0..features {
        center.push(
          // random point relative to origin
          T::gen(seed, scale)
        );
      }

      // add_row will clean the center vector
      centers.add_mut_row(&mut center).unwrap();
    }

    centers
  }

  /// Samples a [`Dataset`] with complex values with vector inputs and vector outputs.
  /// This artificial data has rectangular-like symmetry.
  pub fn sample_complex(
    dims: [usize; 2],
    degree: usize,
    macro_scale: usize,
    micro_scale: usize,
    pred_method: PredictModel,
    seed: &mut u128
  ) -> Result<Dataset<T, T>, DatasetSampleError> {

    if dims[0] < degree { return Err(DatasetSampleError::BellowMinimumSamples) }
      
    let centers = Dataset::<T, T>::gen_complex_centers(dims[1], degree, macro_scale, seed);

    let mut class_center: &[T];
    let mut selected_class: usize;
    let mut one_hot_vec: Vec<T>;
    let mut sample_body: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut labels: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut added_row: Vec<T> = Vec::with_capacity(dims[1]);
    for _ in 0..dims[0] {
      selected_class = lcgi(seed, degree as u128);
      one_hot_vec = T::gen_pred(degree, selected_class, &pred_method).unwrap();
      
      class_center = centers.row(selected_class).unwrap();
      for col in 0..dims[1] {
        added_row.push(
          class_center[col] + T::gen(seed, micro_scale)
        );
      }
      
      // add_row will clean the added_row vec
      labels.push(IOType::Vector(one_hot_vec.clone()));
      sample_body.push(IOType::Vector(added_row.clone()));

      added_row.drain(..);
    }

    Ok(
      Dataset {
        inputs: sample_body,
        target: labels
      }
    )
  }

  pub fn gen_complex_batch_sample(
    dims: [usize; 2],
    centers: Matrix<T>,
    micro_scale: usize,
    pred_method: PredictModel,
    seed: &mut u128
  ) -> Result<Dataset<T, T>, DatasetSampleError> {
    let centers_shape = centers.get_shape();
    let degree = centers_shape[0];
    let _features = centers_shape[1];
    let mut class_center: &[T];
    let mut selected_class: usize;
    let mut one_hot_vec: Vec<T>;
    let mut sample_body: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut labels: Vec<IOType<T>> = Vec::with_capacity(dims[0]);
    let mut added_row: Vec<T> = Vec::with_capacity(dims[1]);
    for _ in 0..dims[0] {
      selected_class = lcgi(seed, degree as u128);
      one_hot_vec = T::gen_pred(degree, selected_class, &pred_method).unwrap();

      class_center = centers.row(selected_class).unwrap();
      for col in 0..dims[1] {
        added_row.push(
          class_center[col] + T::gen(seed, micro_scale)
        );
      }
      
      // add_row will clean the added_row vec
      sample_body.push(IOType::Vector(added_row.clone()));
      labels.push(IOType::Vector(one_hot_vec.clone()));

      added_row.drain(..);
    }

    Ok(
      Dataset {
        inputs: sample_body,
        target: labels
      }
    )
  }
}
