use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::Write;
use std::vec::IntoIter;
use super::BasicOperations;
use rayon::prelude::ParallelSlice;

mod err;
use err::{
  AccessError,
  UpdateError,
  DeletionError,
  OperationError
};


/// Simple matrix structure for basic utilities with minimal error handling. Only for 2D Matrices.
#[derive(Debug, Clone)]
pub struct Matrix<T> {
  /// 1D Vector with 2D mapping intention.
  body: Vec<T>,
  /// Dynamic shape of the body.
  shape: [usize; 2],
  /// Static capacity (for now) of the matrix.
  /// 
  /// # Future Updates
  /// 
  /// Capacity is intended to be dynamic in case more memory 
  /// is needed to be allocated for the matrix
  capacity: [usize; 2],
}

impl<T> Matrix<T> {
  pub fn new() -> Matrix<T> {
    let body = Vec::new();
    let shape = [0_usize, 0];
    let capacity = [0_usize, 0];

    Matrix { body, shape, capacity }
  }
  
  /// Returns an empty generic `Matrix<T>` with enough allocated memory given the `capacity`.
  /// 
  /// # Arguments
  /// 
  /// * `capacity` - Array of `usize` integer with two elements, 
  ///                representing the two dimensions of the matrix.
  /// 
  /// # Notes
  /// 
  /// `capacity` for now, cannot be changed. 
  /// It is fixed when `Matrix<T>` is initiated with `new()` method.
  pub fn with_capacity(capacity: [usize; 2]) -> Matrix<T> {
    let body = Vec::with_capacity(capacity[0] * capacity[1]);
    let shape = [0, 0];

    Matrix { body, shape, capacity }
  }

  pub fn is_empty(&self) -> bool {
    if self.body.len() == 0 && self.shape == [0, 0] && self.capacity == [0, 0] {
      true
    } else {
      false
    }
  }

  pub fn from_body(mut body: Vec<T>, shape: [usize; 2]) -> Matrix<T> {
    body.shrink_to_fit();

    Matrix { body, shape, capacity: shape }
  }

  pub fn export_body(self) -> Vec<T> {
    self.body
  }

  pub fn get_body(&self) -> &[T] {
    &self.body[..]
  }

  pub fn get_body_as_mut(&mut self) -> &mut [T] {
    &mut self.body[..]
  }

  pub fn get_shape(&self) -> &[usize] {
    &self.shape[..]
  }

  pub fn get_capacity(&self) -> &[usize] {
    &self.capacity[..]
  }

  pub fn dealloc(&mut self) {
    self.body.shrink_to_fit();
    self.capacity = self.shape;
  }

  /// Returns a reference to the element in position i, j of a `Matrix<T>`.
  /// 
  /// # Arguments
  /// 
  /// * `i` - `usize` representing the row's index.
  /// * `j` - `usize` representing the column's index.
  pub fn elm(&self, i: usize, j: usize) -> Result<&T, AccessError> {
    // i - lines; j - columns
    if i >= self.shape[0] || j >= self.shape[1] {
      return Err(AccessError::OutOfBounds);
    }

    Ok(&self.body[i * self.shape[1] + j])
  }

  /// Returns a mutable reference to the element in position i, j of a `Matrix<T>`.
  /// 
  /// # Arguments
  /// 
  /// * `i` - reference to a `usize` representing the row's index.
  /// * `j` - reference to a `usize` representing the column's index.
  pub fn elm_mut(&mut self, i: usize, j: usize) -> Result<&mut T, AccessError> {
    // i - lines; j - columns
    if i >= self.shape[0] || j >= self.shape[1] {
      return Err(AccessError::OutOfBounds);
    }

    Ok(&mut self.body[i * self.shape[1] + j])
  }

  /// Returns a slice correspondent to the `i`th row of a `Matrix<T>`.
  /// 
  /// # Arguments
  /// 
  /// * `i` - reference to a `usize` representing the row's index.
  pub fn row(&self, i: usize) -> Result<&[T], AccessError> {
    if i >= self.shape[0] {
      return Err(AccessError::OutOfBounds);
    }

    let init = i * self.shape[1];
    let end = i * self.shape[1] + self.shape[1];

    Ok(&self.body[init..end])
  }

  pub fn row_as_mut(&mut self, i: usize) -> Result<&mut [T], AccessError> {
    if i >= self.shape[0] {
      return Err(AccessError::OutOfBounds);
    }

    let init = i * self.shape[1];
    let end = i * self.shape[1] + self.shape[1];

    Ok(&mut self.body[init..end])
  }

  /// Updates the body of a `Matrix<T>` by adding a 
  /// specified row at the last respective axis position.
  /// 
  /// # Arguments
  /// 
  /// * `row` - mutable reference to a generic `Vec<T>`. 
  ///           Gets consumed after the addition of the row to `Matrix<T>`.
  pub fn add_row(&mut self, row: Vec<T>) -> Result<(), UpdateError> {
    if !(row.len() == self.shape[1] || (self.shape == [0, 0])) {
      return Err(UpdateError::InconsistentLength);
    }

    // try "add" the row
    self.shape[0] += 1;
    
    if (self.shape[0] > self.capacity[0]) || (self.shape[1] > self.capacity[1] || row.len() > self.capacity[1]) {
      self.shape[0] -= 1;
      return Err(UpdateError::Overflow);
    }

    self.shape[1] = row.len();
    self.body.extend(row);

    Ok(())
  }

  pub fn add_mut_row(&mut self, row: &mut Vec<T>) -> Result<(), UpdateError> {
    if !(row.len() == self.shape[1] || self.shape[1] == 0) {
      return Err(UpdateError::InconsistentLength);
    }
    
    if (self.shape[0] >= self.capacity[0]) || (self.shape[1] > self.capacity[1]) {
      return Err(UpdateError::Overflow);
    }

    self.shape[0] += 1;
    self.shape[1] = row.len();
    self.body.append(row);

    Ok(())
  }

  pub fn del_row(&mut self, i: usize) -> Result<Vec<T>, DeletionError> {
    if i >= self.shape[0] {
      return Err(DeletionError::OutOfBounds);
    }

    if self.shape[0] == 0 {
      return Err(DeletionError::Empty);
    }

    self.shape[0] -= 1;

    let init = i * self.shape[1];
    let end = i * self.shape[1] + self.shape[1];

    let row: Vec<T> = self.body
      .drain(init..end)
      .collect();

    if self.shape[0] == 0 {
      self.shape[1] = 0;
    }

    self.dealloc();

    Ok(row)
  }

  /// Updates the body of a `Matrix<T>` by adding a 
  /// specified column at the last respective axis position.
  /// 
  /// # Arguments
  /// 
  /// * `col` - mutable reference to a generic `Vec<T>`. 
  ///           Gets consumed after the addition of the column to `Matrix<T>`.
  /// 
  /// # Notes
  /// 
  /// Not very performant. Use this method outside heavy computation, if possible.
  pub fn add_col(&mut self, col: &mut Vec<T>) -> Result<(), UpdateError> {
    // a row must first be added
    if col.len() == self.shape[0] {
      return Err(UpdateError::InconsistentLength);
    }

    if self.shape[1] < self.capacity[1] {
      return Err(UpdateError::Overflow);
    }

    self.shape[1] += 1;
    self.shape[0] = col.len();

    col.reverse();
    let mut last_row_elm: usize;
    for i in 0..self.shape[0] {
      last_row_elm = i * self.shape[1] + self.shape[1] - 1;
      self.body.splice(
        last_row_elm..last_row_elm, 
        col.pop()
      );
    }

    Ok(())
  }

  pub fn rows_as_iter(&self) -> std::slice::Chunks<'_, T> {
    self.body.chunks(self.shape[1])
  }

  pub fn rows_as_iter_mut(&mut self) -> std::slice::ChunksMut<'_, T> {
    self.body.chunks_mut(self.shape[1])
  }

  pub fn get_slider(&self, i: usize, rows: usize) -> std::slice::Chunks<'_, T> {
    let cols = self.shape[1];
    let start = i*cols;
    let end = i*cols + rows*cols;

    self.body[start..end].chunks(cols)
  }

  pub fn body_into_iter(self) -> IntoIter<T> {
    self.body.into_iter()
  }
}

impl<T: Copy + Sync> Matrix<T> {
  pub fn flip(&self) -> Result<Self, AccessError> {
    let shape = self.shape;
    let mut flipped = Vec::with_capacity(shape[0] * shape[1]);

    for i in (0..shape[0]).rev() {
      for j in (0..shape[1]).rev() {
        flipped.push(*self.elm(i, j).unwrap());
      }
    }

    Ok(Matrix::from_body(flipped, shape))
  }

  pub fn rows_as_par_chunks(&self) -> rayon::slice::Chunks<'_, T> {
    self.body.par_chunks(self.shape[1])
  }

  pub fn row_into_slice(&self, i: usize, result: &mut [T]) -> Result<(), OperationError> {
    if i >= self.shape[0] { return Err(OperationError::OutOfBounds); }

    if result.len() != self.shape[1] { return Err(OperationError::InconsistentShape); }

    let init = i * self.shape[1];
    let end = i * self.shape[1] + self.shape[1];

    result.copy_from_slice(&self.body[init..end]);

    Ok(())
  }

  /// Returns a `Vec` of references correspondent 
  /// to the elements of the `j`th column of a `Matrix<T>`.
  /// 
  /// # Arguments
  /// 
  /// * `j` - reference to a `usize` representing the column's index.
  /// 
  /// # Notes
  /// 
  /// Not very performant. Use this method outside heavy computation, if possible.
  pub fn col_into_slice(&self, j: usize, result: &mut [T]) -> Result<(), OperationError> {
    if j >= self.shape[1] { return Err(OperationError::OutOfBounds); }

    if result.len() != self.shape[0] { return Err(OperationError::InconsistentShape); }

    for i in 0..self.shape[0] {
      result[i] = self.body[i * self.shape[1] + j];
    }

    Ok(())
  }
  
  pub fn copy_col_into_vec(&self, j: usize) -> Result<Vec<T>, OperationError> {
    if j >= self.shape[1] { return Err(OperationError::OutOfBounds); }

    let mut result = Vec::with_capacity(self.shape[0]);
    for i in 0..self.shape[0] {
      result.push(self.body[i * self.shape[1] + j]);
    }

    Ok(result)
  }
}

impl<T: BasicOperations<T>> Matrix<T> {
  pub fn pad(self, pad: (usize, usize)) -> Self {
    let shape = self.shape;
    let mut self_body = self.export_body();

    let padded_rows = shape[0] + 2 * pad.0;
    let padded_cols = shape[1] + 2 * pad.1;

    let padded_matrix = (0..padded_rows).into_iter().flat_map(|row_id| {
      if row_id < pad.0 {
        /* outter pad */
        vec![T::default(); padded_cols]
      } else if row_id >= padded_rows-pad.0 {
        /* outter pad */
        vec![T::default(); padded_cols]
      } else {
        /* inner pad */
        let mut row = vec![T::default(); pad.1];
        row.extend(self_body.drain(0..shape[1]));
        self_body.shrink_to_fit();
        row.extend(vec![T::default(); pad.1]);

        row
      }
    }).collect::<Vec<_>>();

    Matrix::from_body(padded_matrix, [padded_rows, padded_cols])
  }

  pub fn add_mut(&mut self, rhs: &Self) -> Result<(), OperationError> {
    
    if self.shape != rhs.shape && (!self.is_empty() && !rhs.is_empty()) { 
      return Err(OperationError::InconsistentShape);
    }

    if self.is_empty() {
      self.body = rhs.body.clone();
      self.shape = [rhs.get_shape()[0], rhs.get_shape()[1]];
      self.capacity = [rhs.get_capacity()[0], rhs.get_capacity()[1]];

      Ok(())
    } else if rhs.is_empty() {

      Ok(())
    } else {
      self.body
        .iter_mut()
        .zip(rhs.body.iter())
        .for_each(|(lhs, rhs)| { *lhs += *rhs });

      Ok(())
    }
  }

  pub fn add_mut_scalar(&mut self, rhs: T) -> Result<(), OperationError> {
    self.body.iter_mut().for_each(|elm| { *elm += rhs; });
    
    Ok(())
  }

  /// Usefull for multiplying columns or rows with a matrix
  pub fn mul_vec(&self, rhs: Vec<T>) -> Result<Vec<T>, OperationError> {
    if self.shape[1] != rhs.len() { return Err(OperationError::InvalidRHS); }

    let res = self.rows_as_iter().map(|elm| {
      elm
        .into_iter()
        .zip(rhs.iter())
        .fold(T::default(), 
        |acc, elm| { acc + (*elm.0 * *elm.1) }
      )
    }).collect::<Vec<_>>();

    Ok(res)
  }

  /// Usefull for multiplying columns or rows with a matrix
  pub fn mul_slice(&self, rhs: &[T]) -> Result<Vec<T>, OperationError> {
    if self.shape[1] != rhs.len() { return Err(OperationError::InvalidRHS); }

    let res = self.rows_as_iter().map(|elm| {
      elm
        .into_iter()
        .zip(rhs.iter())
        .fold(T::default(), 
        |acc, elm| { acc + (*elm.0 * *elm.1) }
      )
    }).collect::<Vec<_>>();

    Ok(res)
  }

  pub fn mul_mut_scalar(&mut self, rhs: T) -> Result<(), OperationError> {
    self.body.iter_mut().for_each(|elm| { *elm *= rhs; });
    
    Ok(())
  }

  pub fn div_mut_scalar(&mut self, rhs: T) -> Result<(), OperationError> {
    self.body.iter_mut().for_each(|elm| { *elm /= rhs; });

    Ok(())
  }

  pub fn convolution(&self, kernel: &Self) -> Result<Self, OperationError> {
    /* Error handling!! */

    let k_shape = kernel.get_shape();
    let initial_shape = self.get_shape();
    let final_shape = [
      initial_shape[0] - (k_shape[0]-1), 
      initial_shape[1] - (k_shape[1]-1)
    ];

    let convolved_body = (0..final_shape[0])
      /* this can now be paralelized! */
      .into_iter()
      .flat_map(|i| {
        let slider = self.get_slider(i, k_shape[0]);

        let conv_row = slider
          .zip(kernel.rows_as_iter())
          .fold(vec![T::default(); final_shape[1]], |mut acc, (full_row, kernel_row)| {
            let reduced_row = full_row
              .windows(k_shape[1])
              .map(|section| { section.scalar_prod(kernel_row) })
              .collect::<Vec<_>>();

            acc.add_slice_mut(&reduced_row).unwrap();

            acc
        });

        conv_row
    }).collect::<Vec<_>>();

    
    Ok(convolved_body.to_matrix(final_shape).unwrap())
  }

  pub fn block_reduce(&self, block_size: &[usize], block_func: impl Fn(&[T]) -> T) -> Result<Self, OperationError> {
    let matrix_shape = self.get_shape();

    if matrix_shape[0] % block_size[0] != 0 || matrix_shape[1] % block_size[1] != 0 {
      return Err(OperationError::InconsistentShape)
    }

    if matrix_shape[0] < block_size[0] || matrix_shape[1] < block_size[1] {
      return Err(OperationError::InconsistentShape)
    }

    let mut matrix_rows = self.rows_as_iter();
    let mut slider_rows = Vec::with_capacity(block_size[0]);

    let n_rows = matrix_shape[0] / block_size[0];
    let n_cols = matrix_shape[1] / block_size[1];
    let mut result_body = Vec::with_capacity(n_rows * n_cols);
    for _ in 0..n_rows {
      /* request another slider */
      /* update slider row */
      slider_rows.drain(..);
      for _ in 0..block_size[0] {
        slider_rows.push(matrix_rows.next().unwrap().chunks(block_size[1]));
      }

      for _ in 0..n_cols {
        let mut block = Vec::with_capacity(block_size[0] * block_size[1]);
        for slider_row in slider_rows.iter_mut() {
          let block_row = slider_row.next().unwrap();
          block.extend_from_slice(block_row);
        }

        result_body.push(block_func(block.as_slice()));
      }
    }

    let result = Matrix::from_body(result_body, [n_rows, n_cols]);

    Ok(result)
  }

  pub fn fractional_upsampling(&self, block_size: &[usize], kernel: &Self) -> Result<Matrix<T>, OperationError> {
    let matrix_shape = self.get_shape();
    let mut matrix_rows = self.rows_as_iter();

    let n_rows = matrix_shape[0];

    let upper: usize = if block_size[0] % 2 == 0 {(block_size[0] / 2) - 1} else { (block_size[0] - 1) / 2 };
    let bottom: usize = block_size[0] - upper - 1;
    let left: usize = if block_size[1] % 2 == 0 {(block_size[1] / 2) - 1} else { (block_size[0] - 1) / 2 };
    let right: usize = block_size[1] - left - 1;

    let final_shape = [matrix_shape[0] * block_size[0], matrix_shape[1] * block_size[1]];

    let mut res = Vec::new();
    for _ in 0..n_rows {
      /* add upper padding */
      for _ in 0..upper {
        /* add as many rows as upper paddings */
        res.extend(vec![T::default(); final_shape[1]]);
      }
      /* add row with padding in between */
      for row_elm in matrix_rows.next().unwrap().iter() {
        res.extend(vec![T::default(); left]);
        res.push(*row_elm);
        res.extend(vec![T::default(); right]);
      }
      /* add lower padding */
      for _ in 0..bottom {
        /* add as many rows as lower paddings */
        res.extend(vec![T::default(); final_shape[1]]);
      }
    }

    if res.len() != final_shape[0] * final_shape[1] { panic!("Something terribily wrong happened.") }

    let matrix_res = Matrix::from_body(res, [final_shape[0], final_shape[1]]);
    let kernel_shape  = kernel.get_shape();
    if kernel_shape[0] % 2 == 0 || kernel_shape[1] % 2 == 0 {
      return Err(OperationError::InvalidRHS)
    }

    let out = matrix_res
      /* pad to conserve the shap on the convolution */
      .pad((kernel_shape[0]-2, kernel_shape[0]-2))
      .convolution(kernel)
      .unwrap();

    Ok(out)
  }
}

pub trait SliceOps<T> {
  fn add_slice_mut(&mut self, rhs: &Self) -> Result<(), OperationError>;

  fn add_slice(&self, rhs: &Self) -> Result<Vec<T>, OperationError>;

  fn mul_slice_mut(&mut self, rhs: &Self) -> Result<(), OperationError>;

  fn mul_slice(&self, rhs: &Self) -> Result<Vec<T>, OperationError>;

  fn mul_mut_scalar(&mut self, rhs: T) -> Result<(), OperationError>;

  fn div_mut_scalar(&mut self, rhs: T) -> Result<(), OperationError>;

  fn scalar_prod(&self, rhs: &Self) -> T;
}

impl<T: BasicOperations<T>> SliceOps<T> for [T] {
  /// Element-wise assignment summation.
  fn add_slice_mut(&mut self, rhs: &Self) -> Result<(), OperationError> {
    if self.len() != rhs.len() { return Err(OperationError::InconsistentShape) }
    
    self
      .iter_mut()
      .zip(rhs.iter())
      .for_each(|(lhs, rhs)| { *lhs += *rhs });

    Ok(())
  }

  fn add_slice(&self, rhs: &Self) -> Result<Vec<T>, OperationError> {
    if self.len() == 0 {
      return Ok(rhs.to_vec())
    }

    if self.len() != rhs.len() { return Err(OperationError::InconsistentShape) }

    let res = self
      .iter()
      .zip(rhs.iter())
      .map(|(lhs, rhs)| { *lhs + *rhs })
      .collect();
    
    Ok(res)
  }

  fn mul_slice_mut(&mut self, rhs: &Self) -> Result<(), OperationError> {
    if self.len() != rhs.len() { return Err(OperationError::InconsistentShape) }
    
    self
      .iter_mut()
      .zip(rhs)
      .for_each(|(lhs, rhs)| { *lhs *= *rhs });

    Ok(())
  }

  fn mul_slice(&self, rhs: &Self) -> Result<Vec<T>, OperationError> {
    if self.len() != rhs.len() { return Err(OperationError::InconsistentShape) }
    
    let res = self
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| { *lhs * *rhs })
        .collect();
    
    Ok(res)
  }

  fn mul_mut_scalar(&mut self, rhs: T) -> Result<(), OperationError> {
    self
      .iter_mut()
      .for_each(|lhs| { *lhs *= rhs });

    Ok(())
  }

  fn div_mut_scalar(&mut self, rhs: T) -> Result<(), OperationError> {
    self
      .iter_mut()
      .for_each(|lhs| { *lhs /= rhs });

    Ok(())
  }

  fn scalar_prod(&self, rhs: &Self) -> T {
    self
      .iter()
      .zip(rhs.iter())
      .fold(T::default(), |acc, (lhs, rhs)| { acc + (*lhs * *rhs) })
  }
}

pub trait SliceToMatrix<T> {
  fn to_matrix(&self, shape: [usize; 2]) -> Result<Matrix<T>, OperationError>;
}

impl<T: Copy> SliceToMatrix<T> for [T] {
  fn to_matrix(&self, shape: [usize; 2]) -> Result<Matrix<T>, OperationError> {
    if self.len() != shape[0] * shape[1] { return Err(OperationError::InconsistentShape); }

    let body = self.to_vec();
    let result = Matrix::from_body(body, shape);

    Ok(result)
  }
}

impl<T: Debug> Display for Matrix<T> {

  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for elm in self.body.chunks(self.shape[1]) {
      writeln!(f, "{:?}", elm)?;
    }

    write!(f, "shape: {:?}, capacity: {:?}", self.shape, self.capacity)
  }
}

impl<T: Display + Copy> Matrix<T> {
  pub fn to_csv(&self, path: String) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    
    let data_row_len = self.get_shape()[1];
    let data_chunks = self.get_body().chunks(data_row_len);

    for row in data_chunks {
      for val in row.into_iter() {
        write!(file, "{},", val)?;
      }

      writeln!(file, "")?;
    }
    
    Ok(())
  }
}

pub struct SparseVec(usize);
