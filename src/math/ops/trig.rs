use std::ops::{Add, Sub, Mul, Div, Neg};

use crate::math::cfloat::Cfloat;

use super::base::Number;


pub trait Trignometricable {
    fn sin(self) -> Self;

    fn cos(self) -> Self;

    fn tan(self) -> Self; 

    fn sinh(self) -> Self;

    fn cosh(self) -> Self;

    fn tanh(self) -> Self;
}

macro_rules! trignomify {
    ( $( $t: ty ), * ) => {
        $(
            impl Trignometricable for $t {
                fn sin(self) -> Self {
                    self.sin()
                }
            
                fn cos(self) -> Self {
                    self.cos()
                }
            
                fn tan(self) -> Self {
                    self.tan()
                }

                fn sinh(self) -> Self {
                    self.sinh()
                }

                fn cosh(self) -> Self {
                    self.cosh()
                }
            
                fn tanh(self) -> Self {
                    self.tanh()
                }
            }
        )*
    };
}
// All types that have trignometric functions
trignomify!{f32, f64}


impl<P> Trignometricable for Cfloat<P> 
    where 
        P: Add<Output=P> + Sub<Output=P> + Mul<Output=P> + Div<Output=P> + Neg<Output=P>,
        P: PartialEq,
        P: Copy,
        P: Trignometricable + Number {

    fn sin(self) -> Self {

        Cfloat {
            x: self.x.sin() * self.y.cosh(),
            y: self.x.cos() * self.y.sinh()
        }
    }

    fn cos(self) -> Self {
        
        Cfloat {
            x: self.x.cos() * self.y.cosh(),
            y: -(self.x.sin() * self.y.sinh())
        }
    }

    fn tan(self) -> Self {
        let den = self.cos();

        // Potential vulnerability: Search an alternative.
        if den != Cfloat::null() { self.sin() / den } else { Cfloat::inf() }
    }

    fn sinh(self) -> Self {

        Cfloat {
            x: self.x.sinh() * self.y.cos(),
            y: self.x.cosh() * self.y.sin()
        }
    }

    fn cosh(self) -> Self {

        Cfloat {
            x: self.x.cosh() * self.y.cos(),
            y: -(self.x.sinh() * self.y.sin())
        }
    }

    fn tanh(self) -> Self {

        if self != Cfloat::null() { self.sinh() / self.cosh() } else { self }
    }
}