// std

use crate::prelude::neuron::param::Param;

// local
use super::Matrix;


impl<P: Param + Clone> Matrix<P> {

    pub fn add(self, rhs: Matrix<P>) -> Matrix<P> {

        assert!(
            self.shape == rhs.shape,
            "Matrix shapes do not match in Addition."
        );

        let body = self.body
            .into_iter()
            .zip(rhs.body)
            .map(|(lhs, rhs)| { lhs.add(rhs) })
            .collect::<Vec<P>>();

        Matrix {
            body,
            shape: self.shape,
            capacity: self.capacity
        }

    }

    pub fn sub(self, rhs: Matrix<P>) -> Matrix<P> {

        assert!(
            self.shape == rhs.shape,
            "Matrix shapes do not match in Addition."
        );

        let body = self.body
            .into_iter()
            .zip(rhs.body)
            .map(|(lhs, rhs)| { lhs.sub(rhs) })
            .collect::<Vec<P>>();

        Matrix { 
            body, 
            shape: self.shape, 
            capacity: self.capacity 
        }

    }

    pub fn powi(self, n: i32) -> Matrix<P> {

        let body = self.body
            .into_iter()
            .map(|base| { base.powi(n) })
            .collect::<Vec<P>>();

        Matrix { 
            body, 
            shape: self.shape,
            capacity: self.capacity 
        }

    }

}
