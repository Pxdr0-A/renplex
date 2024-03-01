use std::fmt::{Debug, Display};
use std::ops::AddAssign;
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

impl<T: Copy> Matrix<T> {
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
  pub fn new(capacity: [usize; 2]) -> Matrix<T> {
    let body = Vec::with_capacity(capacity[0] * capacity[1]);
    let shape = [0, 0];

    Matrix { body, shape, capacity }
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

  /// Returns a reference to the generic element in position i, j of a `Matrix<T>`.
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

  pub fn row_into_slice(&self, i: usize, result: &mut [T]) -> Result<(), OperationError> {
    if i >= self.shape[0] { return Err(OperationError::OutOfBounds); }

    if result.len() != self.shape[1] { return Err(OperationError::InconsistentShape); }

    let init = i * self.shape[1];
    let end = i * self.shape[1] + self.shape[1];

    result.copy_from_slice(&self.body[init..end]);

    Ok(())
  }

  pub fn row_as_mut(&mut self, i: usize) -> Result<&mut [T], AccessError> {
    if i >= self.shape[0] {
      return Err(AccessError::OutOfBounds);
    }

    let init = i * self.shape[1];
    let end = i * self.shape[1] + self.shape[1];

    Ok(&mut self.body[init..end])
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
}

impl<T> Matrix<T> 
  where 
    T: BasicOperations<T> {

  pub fn add_mut(&mut self, rhs: &Self) -> Result<(), OperationError> {
    if self.shape != rhs.shape { 
      return Err(OperationError::InconsistentShape);
    }

    self.body
      .iter_mut()
      .zip(&rhs.body)
      .for_each(|(lhs, rhs)| { *lhs += *rhs });

    Ok(())
  }

  /// Usefull for adding columns with columns or rows with rows
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
}

pub trait SliceOps<T> {
  fn add_slice(&mut self, rhs: &Self) -> Result<(), OperationError>;
}

impl<T: Copy + AddAssign> SliceOps<T> for [T] {
  fn add_slice(&mut self, rhs: &Self) -> Result<(), OperationError> {
    if self.len() != rhs.len() { return Err(OperationError::InconsistentShape) }
    
    self
      .iter_mut()
      .zip(rhs)
      .for_each(|(lhs, rhs)| { *lhs += *rhs });

    Ok(())
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

impl<T: Display + Debug> Display for Matrix<T> {

  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for elm in self.body.chunks(self.shape[1]) {
      writeln!(f, "{:?}", elm)?;
    }

    write!(f, "shape: {:?}, capacity: {:?}", self.shape, self.capacity)
  }
}

pub struct SparseVec(usize);
