mod err;

use std::any::TypeId;

use crate::{init::PredictModel, math::{matrix::Matrix, random::lcgi, BasicOperations, Complex, Real}};
use err::DatasetSampleError;
//use crate::math::random::{lcgf32, lcgf64};


/// A very low-level and simple dataset representation.
/// Body contains the input that will be feed directly onto a Network.
/// Target contains the expected output to be directly used in calculating the cost function.
#[derive(Debug)]
pub struct Dataset<B, T> {
    pub body: Matrix<B>,
    /// Represents the intended output of the network, i.e. the output layer desired results.
    pub target: Matrix<T>
}

impl<B: Complex + BasicOperations<B>, T: Real + BasicOperations<T>> Dataset<B, T>
  where <B as Complex>::Precision: 'static, T: 'static {
 
  pub fn sample(
      dims: [usize; 2],
      degree: usize,
      macro_scale: usize,
      micro_scale: usize,
      pred_method: PredictModel,
      seed: &mut u128
  ) -> Result<Dataset<B, T>, DatasetSampleError> {

    if dims[0] < degree { return Err(DatasetSampleError::BellowMinimumSamples) }
    if TypeId::of::<B::Precision>() != TypeId::of::<T>() { return Err(DatasetSampleError::IncompatibleTargetType) }

    let mut sample_matrix: Matrix<B> = Matrix::with_capacity(dims);
      
    // spray focal points
    let mut centers: Matrix<B> = Matrix::with_capacity([degree, dims[1]]);
    let mut center: Vec<B> = Vec::with_capacity(dims[1]);
    for _ in 0..degree {
      for _ in 0..dims[1] {
        center.push(
          // random point relative to origin
          B::gen(seed, macro_scale)
        );
      }

      // add_row will clean the center vector
      centers.add_mut_row(&mut center).unwrap();
    }
    drop(center);

    let mut class_center: &[B];
    let mut selected_class: usize;
    let mut one_hot_vec: Vec<T>;
    let mut labels: Matrix<T> = Matrix::with_capacity([dims[0], degree]);
    let mut added_row: Vec<B> = Vec::with_capacity(dims[1]);
    for _ in 0..dims[0] {
        selected_class = lcgi(seed, degree as u128);
        println!("{}", selected_class);

        one_hot_vec = T::gen_pred(degree, selected_class, &pred_method).unwrap();
        labels.add_mut_row(&mut one_hot_vec).unwrap();
        
        class_center = centers.row(selected_class).unwrap();

        for col in 0..dims[1] {
            added_row.push(
                class_center[col] + B::gen(seed, micro_scale)
            );
        }
        
        // add_row will clean the added_row vec
        sample_matrix.add_mut_row(&mut added_row).unwrap();
    }

    Ok(
      Dataset {
        body: sample_matrix,
        target: labels
      }
    )
  }
}
