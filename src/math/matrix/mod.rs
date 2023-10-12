pub mod dataset;


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

    /// Returns a reference to the generic element in position i, j of a `Matrix<T>`.
    /// 
    /// # Arguments
    /// 
    /// * `i` - reference to a `usize` representing the row's index.
    /// * `j` - reference to a `usize` representing the column's index.
    pub fn elm(&self, i: &usize, j: &usize) -> &T {
        // i - lines; j - columns
        assert!(
            i < &self.shape[0] && j < &self.shape[1],
            "Index Error: Some axis is out of bounds."
        );

        &self.body[i * self.shape[1] + j]
    }

    /// Returns a slice correspondent to the `i`th row of a `Matrix<T>`.
    /// 
    /// # Arguments
    /// 
    /// * `i` - reference to a `usize` representing the row's index.
    pub fn row(&self, i: &usize) -> &[T] {
        assert!(
            i < &self.shape[0],
            "Index Error: Row index is out of bounds."
        );

        let init = i * self.shape[1];
        let end = i * self.shape[1] + self.shape[1];

        &self.body[init..end]
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
    pub fn column(&self, j: &usize) -> Vec<&T> {
        assert!(
            j < &self.shape[1],
            "Index Error. Column index is out of bounds."
        );

        let mut column: Vec<&T> = Vec::with_capacity(self.shape[0]);
        for i in 0..self.shape[0] {
            column.push(&self.body[i * self.shape[1] + j]);
        }

        column
    }

    /// Updates the body of a `Matrix<T>` by adding a 
    /// specified row at the last respective axis position.
    /// 
    /// # Arguments
    /// 
    /// * `row` - mutable reference to a generic `Vec<T>`. 
    ///           Gets consumed after the addition of the row to `Matrix<T>`.
    pub fn add_row(&mut self, row: &mut Vec<T>) {
        assert!(
            (row.len() == self.shape[1]) || self.shape[1] == 0,
            "Invalid Addition: Inconsistent row length."
        );

        assert!(
            self.capacity[0] > self.shape[0],
            "Invalid Addition: Attempting to exceed allocated memory."
        );

        self.shape[0] += 1;
        self.shape[1] = row.len();
        self.body.append(row);
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
    pub fn add_col(&mut self, col: &mut Vec<T>) {
        // a row must first be added
        assert!(
            (col.len() == self.shape[0]),
            "Invalid Addition: Inconsistent column length."
        );

        assert!(
            self.capacity[1] > self.shape[1],
            "Invalid Addition: Attempting to exceed allocated memory."
        );

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
    }
}
