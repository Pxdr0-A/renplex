//! Complex properties, operations and castings.
//! 
//! Provides simple tools to handle complex numbers with generic types. Nevertheless, 
//! these tools are based on f32 and f64 which are the only primitives that implement
//! the traits requested by the tools.


pub mod ops;
pub mod casts;


// std
use std::ops::{Add, Neg, Mul, Div};

//local
use super::ops::{
    arc::Arcable, 
    sqrt::SquareRootable
};


#[derive(Debug, Clone, Copy, PartialEq)]
/// Generic structure for a complex number.
/// 
/// Complex numbers here were defined in their cartesian form.
pub struct Cfloat<P> {
    pub x: P,
    pub y: P
}

impl<P> Cfloat<P> 
    where 
        P: Add<Output=P> + Mul<Output=P> + Div<Output=P> + Neg<Output=P>,
        P: SquareRootable + Arcable,
        P: Copy  {

    pub fn new(x: P, y: P) -> Cfloat<P> {

        Cfloat {x, y}
    }

    pub fn re(&self) -> P {

        self.x
    }

    pub fn im(&self) -> P {
                
        self.y
    }

    pub fn norm(&self) -> P {
        
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn phase(&self) -> P {

        (self.y / self.x).atan()
    }

    pub fn conj(&self) -> Cfloat<P> {
        
        Cfloat {
            x: self.x,
            y: -self.y
        }
    }

    pub fn inv(&self) -> Cfloat<P> {

        Cfloat {
            x: -self.x,
            y: -self.y
        }
    }
}