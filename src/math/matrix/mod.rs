use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::Write;
use std::slice::{Chunks, ChunksMut};
use std::thread;
use std::vec::IntoIter;
use super::BasicOperations;

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
  /// * `i` - reference to a `usize` representing the row's index.
  /// * `j` - reference to a `usize` representing the column's index.
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
  pub fn add_row(&mut self, mut row: Vec<T>) -> Result<(), UpdateError> {
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
    self.body.append(&mut row);

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

  pub fn rows_as_iter(&self) -> Chunks<'_, T> {
    self.body.chunks(self.shape[1])
  }

  pub fn rows_as_iter_mut(&mut self) -> ChunksMut<'_, T> {
    self.body.chunks_mut(self.shape[1])
  }

  pub fn into_iter(self) -> IntoIter<T> {
    self.body.into_iter()
  }
}

impl<T: Copy> Matrix<T> {
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
        .zip(&rhs.body)
        .for_each(|(lhs, rhs)| { *lhs += *rhs });

      Ok(())
    }
  }

  /// Usefull for adding intire arrays to the body of a matrix.
  pub fn add_slice(&mut self, rhs: &[T]) -> Result<(), OperationError> {
    if self.body.len() != rhs.len() { 
      return Err(OperationError::InconsistentShape);
    }

    self.body
      .iter_mut()
      .zip(rhs)
      .for_each(|(lhs, rhs)| { *lhs += *rhs });

    Ok(())
  }

  pub fn add_mut_scalar(&mut self, rhs: T) -> Result<(), OperationError> {
    for elm in self.body.iter_mut() {
      *elm += rhs;
    }
    
    Ok(())
  }

  pub fn sub_mut(&mut self, rhs: &Self) -> Result<(), OperationError> {
    if self.shape != rhs.shape { 
      return Err(OperationError::InconsistentShape);
    }

    self.body
      .iter_mut()
      .zip(&rhs.body)
      .for_each(|(lhs, rhs)| { *lhs -= *rhs });

    Ok(())
  }

  pub fn mul(&self, rhs: &Self) -> Result<Matrix<T>, OperationError> {    
    if self.shape[1] != rhs.shape[0] { return Err(OperationError::InvalidRHS); }

    let new_shape: [usize; 2] = [self.shape[0], rhs.shape[1]];
    let mut result_body = vec![T::default(); new_shape[0] * new_shape[1]];
    
    let row_buf = &mut vec![T::default(); self.shape[1]][..];
    let col_buf = &mut vec![T::default(); rhs.shape[0]][..];
    for c in 0..new_shape[1] {
      rhs.col_into_slice(c, col_buf).unwrap();
      for r in 0..new_shape[0] {
        self.row_into_slice(r, row_buf).unwrap();

        result_body[r * new_shape[1] + c] = (0..self.shape[1])
          .fold(T::default(), |acc, k| {
            acc + row_buf[k] * col_buf[k]
        });
      }
    }

    Ok(
      Matrix { 
        body: result_body, 
        shape: new_shape, 
        capacity: new_shape 
      }
    )
  }

  /// Usefull for multiplying columns or rows with a matrix
  pub fn mul_vec(&self, rhs: Vec<T>) -> Result<Vec<T>, OperationError> {
    if self.shape[1] != rhs.len() { return Err(OperationError::InvalidRHS); }

    let mut result_body = vec![T::default(); self.shape[0]];
    
    let row_buf = &mut vec![T::default(); self.shape[1]][..];
    for r in 0..self.shape[0] {
      self.row_into_slice(r, row_buf).unwrap();

      result_body[r] = (0..self.shape[1])
        .fold(T::default(), |acc, k| {
          /* go through columns of matrix and rows of vector (same value) */
          acc + row_buf[k] * rhs[k]
      });
    }

    Ok(result_body)
  }

  pub fn mul_elm_vec(&mut self, rhs: &Vec<T>) -> Result<(), OperationError> {
    if self.shape[1] != rhs.len() { return Err(OperationError::InvalidRHS); }

    for row in self.rows_as_iter_mut() {
      for (elm, other) in row.into_iter().zip(rhs) {
        *elm *= *other;
      }
    }

    Ok(())
  }

  /// Usefull for multiplying columns or rows with a matrix
  pub fn mul_slice(&self, rhs: &[T]) -> Result<Vec<T>, OperationError> {
    if self.shape[1] != rhs.len() { return Err(OperationError::InvalidRHS); }

    let mut result_body = vec![T::default(); self.shape[0]];
    
    let row_buf = &mut vec![T::default(); self.shape[1]][..];
    for r in 0..self.shape[0] {
      self.row_into_slice(r, row_buf).unwrap();

      result_body[r] = (0..self.shape[1])
        .fold(T::default(), |acc, k| {
          /* go through columns of matrix and rows of vector (same value) */
          acc + row_buf[k] * rhs[k]
      });
    }

    Ok(result_body)
  }

  pub fn mul_mut_scalar(&mut self, rhs: T) -> Result<(), OperationError> {
    for elm in self.body.iter_mut() {
      *elm *= rhs;
    }
    
    Ok(())
  }

  pub fn mul_point(&self, rhs: &Self) -> Result<Self, OperationError> {
    let rhs_shape = rhs.get_shape();
    let lhs_shape = self.get_shape();
    if lhs_shape != rhs_shape {
      return Err(OperationError::InconsistentShape)
    }

    let mut body = Vec::with_capacity(lhs_shape[0] * lhs_shape[1]);
    for (lhs_elm, rhs_elm) in self.get_body().iter().zip(rhs.get_body()) {
      body.push(*lhs_elm * *rhs_elm);
    }

    Ok(Matrix::from_body(body, [lhs_shape[0], lhs_shape[1]]))
  }

  pub fn mul_point_mut(&mut self, rhs: &Self) -> Result<(), OperationError> {
    let rhs_shape = rhs.get_shape();
    let mut lhs_shape = [0_usize, 0];
    lhs_shape.copy_from_slice(self.get_shape());

    if lhs_shape != rhs_shape {
      return Err(OperationError::InconsistentShape)
    }

    for (lhs_elm, rhs_elm) in self.get_body_as_mut().iter_mut().zip(rhs.get_body()) {
      *lhs_elm *= *rhs_elm;
    }

    Ok(())
  }

  pub fn div_mut_scalar(&mut self, rhs: T) -> Result<(), OperationError> {
    for elm in self.body.iter_mut() {
      *elm /= rhs;
    }

    Ok(())
  }

  pub fn conv(&self, kernel: &Self) -> Result<Self, OperationError> {
    /* define target block */
    let kernel_shape = kernel.get_shape();
    let matrix_shape = self.get_shape();
    /* check if kernel is to big */
    if ((kernel_shape[0] - 1) / 2) >= matrix_shape[0] || ((kernel_shape[1] - 1) / 2) >= matrix_shape[1] {
      return Err(OperationError::OutOfBounds)
    }
    /* shape needs to be odd numbers */
    if kernel_shape[0] % 2 == 0 || kernel_shape[1] % 2 == 0 {
      return Err(OperationError::InvalidRHS)
    }
    /* and also greater or equal to 3 */
    if kernel_shape[0] < 3 || kernel_shape[1] < 3 {
      return Err(OperationError::InvalidRHS)
    }
    
    let mut matrix_rows = self.rows_as_iter();
    
    let max_pad_col = (kernel_shape[1] - 1) / 2;
    let max_pad_row = (kernel_shape[0] - 1) / 2;

    let mut slider = Vec::with_capacity(kernel_shape[0]);
    let mut slide_row = Vec::with_capacity(matrix_shape[1] + (kernel_shape[1] - 1));

    let row_pad = vec![T::default(); matrix_shape[1] + (kernel_shape[1] - 1)];
    /* create current slide_rows with padding by chaining iterators */
    /* you can try to improve memory here */
    /* this is valid for the first iteration! */
    for _ in 0..max_pad_row { slider.push(row_pad.clone()); }
    for _ in 0..(kernel_shape[0] - max_pad_row) {
      let matrix_row = matrix_rows.next().unwrap();

      /* intial col pad */
      slide_row.append(&mut vec![T::default(); max_pad_col]);
      /* core vals */
      slide_row.append(&mut matrix_row.to_vec());
      /* final col pad */
      slide_row.append(&mut vec![T::default(); max_pad_col]);

      slider.push(slide_row.clone());

      slide_row.drain(..);
    }

    let mut thread_results = Vec::new();
    for row_id in 0..matrix_shape[0] {
      let current_slider = slider.clone();
      let kernel_copy = kernel.clone();
      let handle = thread::spawn(move || { 
        let kernel_shape = kernel_copy.get_shape();
        current_slider
          .iter()
          .map(|row| {row.windows(kernel_shape[1])})
          .enumerate()
          .map(|(row_id, windows)| {
            /* slide the kernel row through the image row */
            /* vec that now has dim[0] = matrix[0] */
            /* you can try to create threads here */
            /* each thread will give a Vec<T> */
            /* you can use drain instead */
            let kernel_row = kernel_copy.row(row_id).unwrap();
            windows
              .map(|window| { window.scalar_prod(kernel_row).unwrap() })
              .collect::<Vec<T>>()
          })
          .reduce(|acc, row| {
            /* you can try to create threads here */
            /* each thread will give a Vec<T> */
            acc.add_slice(&row).unwrap()
          })
          .unwrap()
      });

      thread_results.push(handle);
      
      slider.drain(0..1);
      if row_id+1 + max_pad_row >= matrix_shape[0] {
        /* kernel overflowed bottom pixels */
        slider.push(row_pad.clone());
      } else {
        /* kernel is still within the matrix */
        /* with the zeros added (col padding) */
        /* intial col pad */
        slide_row.append(&mut vec![T::default(); max_pad_col]);
        /* core vals */
        slide_row.append(&mut matrix_rows.next().unwrap().to_vec());
        /* final col pad */
        slide_row.append(&mut vec![T::default(); max_pad_col]);

        slider.push(slide_row.clone());

        slide_row.drain(..);
      }
    }

    let mut out = Matrix::with_capacity([matrix_shape[0], matrix_shape[1]]);
    let mut convolved_row;
    for thread_result in thread_results.into_iter() {
      convolved_row = thread_result.join().unwrap();
      out.add_mut_row(&mut convolved_row).unwrap();
    }

    Ok(out)
  }

  pub fn rev_conv(&self, kernel: &Self) -> Result<Self, OperationError> {
    /* define target block */
    let kernel_shape = kernel.get_shape();
    let matrix_shape = self.get_shape();
    /* check if kernel is to big */
    if ((kernel_shape[0] - 1) / 2) >= matrix_shape[0] || ((kernel_shape[1] - 1) / 2) >= matrix_shape[1] {
      return Err(OperationError::OutOfBounds)
    }
    /* shape needs to be odd numbers */
    if kernel_shape[0] % 2 == 0 || kernel_shape[1] % 2 == 0 {
      return Err(OperationError::InvalidRHS)
    }
    /* and also greater or equal to 3 */
    if kernel_shape[0] < 3 || kernel_shape[1] < 3 {
      return Err(OperationError::InvalidRHS)
    }
    
    let mut matrix_rows = self.rows_as_iter().rev();
    
    let max_pad_col = (kernel_shape[1] - 1) / 2;
    let max_pad_row = (kernel_shape[0] - 1) / 2;

    let mut slider = Vec::with_capacity(kernel_shape[0]);
    let mut slide_row = Vec::with_capacity(matrix_shape[1] + (kernel_shape[1] - 1));

    let row_pad = vec![T::default(); matrix_shape[1] + (kernel_shape[1] - 1)];
    /* create current slide_rows with padding by chaining iterators */
    /* you can try to improve memory here */
    /* this is valid for the first iteration! */
    for _ in 0..(kernel_shape[0] - max_pad_row) {
      let matrix_row = matrix_rows.next().unwrap();

      /* intial col pad */
      slide_row.append(&mut vec![T::default(); max_pad_col]);
      /* core vals */
      slide_row.append(&mut matrix_row.to_vec());
      /* final col pad */
      slide_row.append(&mut vec![T::default(); max_pad_col]);

      slider.push(slide_row.clone());

      slide_row.drain(..);
    }
    for _ in 0..max_pad_row { slider.push(row_pad.clone()); }

    let mut thread_results = Vec::new();
    for row_id in 0..matrix_shape[0] {
      let kernel_copy = kernel.clone();
      let slider_copy = slider.clone();
      let handle = thread::spawn(move || {
        let kernel_shape = kernel_copy.get_shape();
        slider_copy
          .iter()
          .rev()
          /* create windows to slide the filter through */
          .map(|row| { row.windows(kernel_shape[1]) })
          .enumerate()
          .map(|(row, windows)| {
            /* slide the kernel row through the image row */
            let kernel_row = kernel_copy.row(row).unwrap();
            windows
              .map(|window| { window.scalar_prod(kernel_row).unwrap() })
              .collect::<Vec<T>>()
          })
          .reduce(|acc, row| {
            acc.add_slice(&row).unwrap()
          })
          .unwrap()
      });

      thread_results.push(handle);

      /* update slide_rows by draining the first element and adding another new one */
      /* remove last element */
      slider.pop();
      /* add element to the last position */
      if row_id+1 + max_pad_row >= matrix_shape[0] {
        /* kernel overflowed bottom pixels */
        slider.push(row_pad.clone());
      } else {
        /* kernel is still within the matrix */
        /* with the zeros added (col padding) */
        /* intial col pad */
        slide_row.append(&mut vec![T::default(); max_pad_col]);
        /* core vals */
        slide_row.append(&mut matrix_rows.next().unwrap().to_vec());
        /* final col pad */
        slide_row.append(&mut vec![T::default(); max_pad_col]);

        slider.push(slide_row.clone());

        slide_row.drain(..);
      }
      /* to get the element that was added last, to the first position */
      slider.rotate_right(1);
    }

    let mut out = Matrix::with_capacity([matrix_shape[0], matrix_shape[1]]);
    let mut convolved_row;
    for thread_result in thread_results.into_iter() {
      convolved_row = thread_result.join().unwrap();
      out.add_mut_row(&mut convolved_row).unwrap();
    } 

    Ok(out)
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
          block.append(&mut block_row.to_vec());
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

    // even
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
        res.append(&mut vec![T::default(); final_shape[1]]);
      }

      /* add row with padding in between */
      for row_elm in matrix_rows.next().unwrap().iter() {
        res.append(&mut vec![T::default(); left]);
        res.push(*row_elm);
        res.append(&mut vec![T::default(); right]);
      }

      /* add lower padding */
      for _ in 0..bottom {
        /* add as many rows as lower paddings */
        res.append(&mut vec![T::default(); final_shape[1]]);
      }
    }

    if res.len() != final_shape[0] * final_shape[1] { panic!("Something terribily wrong happened.") }

    let out = Matrix::from_body(res, [final_shape[0], final_shape[1]])
      .conv(kernel)
      .unwrap();

    Ok(out)
  }
}

pub trait SliceOps<T> {
  fn add_slice_mut(&mut self, rhs: &Self) -> Result<(), OperationError>;

  fn add_slice(&self, rhs: &Self) -> Result<Vec<T>, OperationError>;

  fn sub_slice_mut(&mut self, rhs: &Self) -> Result<(), OperationError>;

  fn mul_slice_mut(&mut self, rhs: &Self) -> Result<(), OperationError>;

  fn mul_slice(&self, rhs: &Self) -> Result<Vec<T>, OperationError>;

  fn mul_mut_scalar(&mut self, rhs: T) -> Result<(), OperationError>;

  fn scalar_prod(&self, rhs: &Self) -> Result<T, OperationError>;
}

impl<T: BasicOperations<T>> SliceOps<T> for [T] {
  /// Element-wise assignment summation.
  fn add_slice_mut(&mut self, rhs: &Self) -> Result<(), OperationError> {
    if self.len() != rhs.len() { return Err(OperationError::InconsistentShape) }
    
    self
      .iter_mut()
      .zip(rhs)
      .for_each(|(lhs, rhs)| { *lhs += *rhs });

    Ok(())
  }

  fn add_slice(&self, rhs: &Self) -> Result<Vec<T>, OperationError> {
    if self.len() != rhs.len() { return Err(OperationError::InconsistentShape) }
    
    Ok(
      self
        .iter()
        .zip(rhs)
        .map(|(lhs, rhs)| { *lhs + *rhs })
        .collect()
    )
  }

  fn sub_slice_mut(&mut self, rhs: &Self) -> Result<(), OperationError> {
    if self.len() != rhs.len() { return Err(OperationError::InconsistentShape) }
    
    self
      .iter_mut()
      .zip(rhs)
      .for_each(|(lhs, rhs)| { *lhs -= *rhs });

    Ok(())
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
    
    Ok(
      self
        .iter()
        .zip(rhs)
        .map(|(lhs, rhs)| { *lhs * *rhs })
        .collect()
    )
  }

  fn mul_mut_scalar(&mut self, rhs: T) -> Result<(), OperationError> {
    self
      .iter_mut()
      .for_each(|lhs| { *lhs *= rhs });

    Ok(())
  }

  fn scalar_prod(&self, rhs: &Self) -> Result<T, OperationError> {
    if self.len() != rhs.len() { return Err(OperationError::InconsistentShape) }

    let res = self
      .iter()
      .zip(rhs)
      .fold(T::default(), |acc, (lhs, rhs)| { acc + (*lhs * *rhs) });

    Ok(res)
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

impl<T: Debug + Copy> Matrix<T> {
  pub fn to_csv(&self, path: &'static str) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    
    let data_row_len = self.get_shape()[1];
    let data_chunks = self.get_body().chunks(data_row_len);

    let mut string_val;
    for row in data_chunks {
      string_val = format!("{:?}", row)
        .replace(" ", "")
        .replace("[", "")
        .replace("]", "");
      
      writeln!(file, "{}", string_val)?;
    }
    
    Ok(())
  }
}

pub struct SparseVec(usize);
