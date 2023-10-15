use std::ops::Mul;

use crate::math::complex::Cfloat;

use super::trig::Trignometricable;

pub trait Exponentiable {
    fn exp(self) -> Self; 
}

macro_rules! exponensify {
    ( $( $t: ty ), * ) => {
        $(
            impl Exponentiable for $t {
                fn exp(self) -> $t {
                    self.exp()
                }
            }
        )*
    };
}
exponensify!{f32, f64}


impl<P> Exponentiable for Cfloat<P> 
    where 
        P: Trignometricable + Exponentiable, 
        P: Mul<Output=P>,
        P: Copy {

    fn exp(self) -> Self {
        
        Cfloat {
            x: self.x.exp() * self.y.cos(),
            y: self.x.exp() * self.y.sin()
        }
    }
}