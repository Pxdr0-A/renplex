// std

use crate::lite::real::Param;
use crate::lite::complex::ComplexParam;

// local
use super::Matrix;


impl<P: Param + Copy> Matrix<P> {

    pub fn add(self, rhs: Matrix<P>) -> Matrix<P> {

        if self.shape != rhs.shape { panic!("Matrix shapes do not match in Subtraction.") }

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

        if self.shape != rhs.shape { panic!("Matrix shapes do not match in Subtraction.") }

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

impl<CP: ComplexParam + Copy> Matrix<CP> {

    pub fn add_cp(self, rhs: Matrix<CP>) -> Matrix<CP> {

        if self.shape != rhs.shape { panic!("Matrix shapes do not match in Subtraction.") }

        let body = self.body
            .into_iter()
            .zip(rhs.body)
            .map(|(lhs, rhs)| { lhs.add(rhs) })
            .collect::<Vec<CP>>();

        Matrix {
            body,
            shape: self.shape,
            capacity: self.capacity
        }

    }

    pub fn sub_cp(self, rhs: Matrix<CP>) -> Matrix<CP> {

        if self.shape != rhs.shape { panic!("Matrix shapes do not match in Subtraction.") }

        let body = self.body
            .into_iter()
            .zip(rhs.body)
            .map(|(lhs, rhs)| { lhs.sub(rhs) })
            .collect::<Vec<CP>>();

        Matrix { 
            body, 
            shape: self.shape, 
            capacity: self.capacity 
        }

    }

    pub fn abs_sq(self) -> Matrix<CP> {
        let body = self.body
            .into_iter()
            .map(|elm| { elm.conj().mul(elm) })
            .collect::<Vec<CP>>();

        Matrix { 
            body, 
            shape: self.shape, 
            capacity: self.capacity 
        }
    }
}
