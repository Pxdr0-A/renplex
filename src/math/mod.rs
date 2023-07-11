/* Math: Arithmetic, Operations and Structures

 */

pub mod backpropagation;

use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    body: Vec<T>,
    pub shape: [usize; 2],
    capacity: [usize; 2]
}

impl<T> Matrix<T> {
    pub fn new(capacity: [usize; 2]) -> Matrix<T> {
        // allocates enough memory
        let body = Vec::with_capacity(capacity[0] * capacity[1]);
        let shape = [0, 0];

        Matrix { body, shape, capacity }
    }

    pub fn elm(&self, i: &usize, j: &usize) -> &T {
        // i - lines; j - columns
        assert!(i < &self.shape[0] && j < &self.shape[1],
                "Index Error: Some axis is out of bounds."
        );

        &self.body[i * self.shape[1] + j]
    }

    pub fn row(&self, i: &usize) -> &[T] {
        assert!(i < &self.shape[0],
                "Index Error: Row index is out of bounds."
        );

        let init = i * self.shape[1];
        let end = i * self.shape[1] + self.shape[1];

        &self.body[init..end]
    }

    pub fn column(&self, j: &usize) -> Vec<&T> {
        assert!(j < &self.shape[1],
                "Index Error. Column index is out of bounds."
        );

        let mut column: Vec<&T> = Vec::with_capacity(self.shape[0]);
        for i in 0..self.shape[0] {
            column.push(&self.body[i * self.shape[1] + j]);
        }

        column
    }

    pub fn add_row(&mut self, row: &mut Vec<T>) {
        assert!((row.len() == self.shape[1]) || self.shape[1] == 0,
                   "Invalid Addition: Inconsistent row length."
        );

        assert!(self.capacity[0] > self.shape[0],
                   "Invalid Addition: Attempting to exceed allocated memory."
        );

        self.shape[0] += 1;
        self.shape[1] = row.len();
        self.body.append(row);
    }

    pub fn add_col(&mut self, col: &mut Vec<T>) {
        // a row must first be added
        assert!((col.len() == self.shape[0]),
                   "Invalid Addition: Inconsistent column length."
        );

        assert!(self.capacity[1] > self.shape[1],
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

pub mod random {
    pub fn lcg(seed: &mut u128) -> f64 {
        // IBM C/C++ convention params
        let a: u128 = 1103515245;
        let b: u128 = 12345;
        let m: u128 = 2u128.pow(31);

        *seed = (a * *seed + b) % (m - 1);
        let rand = (*seed as f64) / (m as f64);

        rand
    }
}