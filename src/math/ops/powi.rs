use std::ops::{Add, Sub, Neg, Mul, Div};
use std::iter::Sum;

use crate::math::cfloat::Cfloat;
use crate::math::matrix::Matrix;

use super::base::Complex;
use super::sqrt::SquareRootable;
use super::{
    trig::Trignometricable, 
    arc::Arcable
};


pub trait Powerable {
    fn powi(self, n: i32) -> Self;
}

macro_rules! powerfy {
    ( $( $t: ty ), * ) => {
        $(
            impl Powerable for $t {
                fn powi(self, n: i32) -> $t {
                    self.powi(n)
                }
            }
        )*
    };
}
powerfy!{f32, f64}


impl<P> Powerable for Cfloat<P> 
    where
        Cfloat<P>: Complex<P>,
        P: Trignometricable + Arcable + Powerable + SquareRootable,
        P: Add<Output=P> + Sub<Output=P> + Neg<Output=P> + Mul<Output=P> + Div<Output=P>,
        P: Sum,
        P: Copy {

    fn powi(self, n: i32) -> Self {
        
        let power_norm = self
            .norm()
            .powi(n);
        
        let phase = self.phase();
        // phase * n
        let prod_phase: P = vec![phase; n as usize].into_iter().sum();

        Cfloat { 
            x: power_norm * prod_phase.cos(), 
            y: power_norm * prod_phase.sin() 
        }
    }
}

impl<T> Powerable for Matrix<T> 
    where 
        T: Powerable {

    fn powi(self, n: i32) -> Self {

        let result: Vec<T> = self.body
            .into_iter()
            .map( |x| { x.powi(n) } )
            .collect();
        
        Matrix { 
            body: result, 
            shape: self.shape, 
            capacity: self.capacity 
        }
    }
}