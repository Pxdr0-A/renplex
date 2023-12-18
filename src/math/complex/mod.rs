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
    sqrt::SquareRootable, base::Complex
};


#[derive(Debug, Clone, Copy, PartialEq)]
/// Generic structure for a complex number.
/// 
/// Complex numbers here were defined in their cartesian form.
pub struct Cfloat<T> {
    pub x: T,
    pub y: T
}

impl<T> Complex<T> for Cfloat<T> 
    where 
        T: Add<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T>,
        T: SquareRootable + Arcable,
        T: Copy  {

    fn new(x: T, y: T) -> Cfloat<T> {

        Cfloat {x, y}
    }

    fn re(&self) -> T {

        self.x
    }

    fn im(&self) -> T {
                
        self.y
    }

    fn norm(&self) -> T {
        
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn phase(&self) -> T {

        (self.y / self.x).atan()
    }

    fn conj(&self) -> Cfloat<T> {
        
        Cfloat {
            x: self.x,
            y: -self.y
        }
    }

    fn inv(&self) -> Cfloat<T> {

        Cfloat {
            x: -self.x,
            y: -self.y
        }
    }

}