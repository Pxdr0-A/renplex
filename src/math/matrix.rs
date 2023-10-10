use super::random::lcgf32;


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


// pass this to a macro for f64
impl Matrix<f32> {
    pub fn sample(
        dims: [usize; 2],
        degree: usize,
        seed: &mut u128

    ) -> (Matrix<f32>, Vec<f32>) {

        assert!(
            dims[0] > degree,
            "Number of instances needs to be larger than the number of classes."
        );

        let mut sample_matrix: Matrix<f32> = Matrix::new(dims);

        let macro_scale: f32 = 100.0;
        let micro_scale: f32 = macro_scale / 50.0;

        let mut rand_val: f32 = lcgf32(seed);

        // spray focal points
        let mut centers: Matrix<f32> = Matrix::new([degree, dims[1]]);
        let mut center: Vec<f32> = Vec::with_capacity(dims[1]);
        for _ in 0..degree {
            for _ in 0..dims[1] {
                center.push(
                    // random point relative to origin
                    rand_val * macro_scale - (macro_scale / 2.0)
                );

                rand_val = lcgf32(seed);
            }

            // add_row will clean the center vector
            centers.add_row(&mut center);
        }

        let mut class_center: &[f32];
        let mut selected_class: f32;
        let mut labels: Vec<f32> = Vec::with_capacity(dims[0]);
        let mut added_row: Vec<f32> = Vec::with_capacity(dims[1]);
        for _ in 0..dims[0] {
            selected_class = (rand_val * (degree as f32) - 1.0).round();
            labels.push(selected_class);
            
            class_center = centers.row(
                &(
                    selected_class as usize
                )
            );

            for col in 0..dims[1] {
                rand_val = lcgf32(seed);

                added_row.push(
                    class_center[col] + rand_val * micro_scale - (micro_scale / 2.0)
                );
            }
            
            // add_row will clean the added_row vec
            sample_matrix.add_row(&mut added_row);

            rand_val = lcgf32(seed);
        }

        (sample_matrix, labels)
    }
}