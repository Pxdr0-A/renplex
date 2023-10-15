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
pub struct Cfloat<P> {
    pub x: P,
    pub y: P
}

impl<P> Cfloat<P> {
    pub fn new(x: P, y: P) -> Cfloat<P> {
        Cfloat {x, y}
    }

    pub fn re(&self) -> P 
        where 
            P: Copy {

        self.x
    }

    pub fn im(&self) -> P 
        where
            P: Copy {
                
        self.y
    }

    pub fn norm(&self) -> P 
        where 
            P: Add<Output=P> + Mul<Output=P>,
            P: SquareRootable,
            P: Copy {
        
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn phase(&self) -> P
        where 
            P: Div<Output=P>,
            P: Arcable,
            P: Copy {

        (self.y / self.x).atan()
    }

    pub fn conj(&self) -> Cfloat<P> 
        where 
            P: Neg<Output=P>,
            P: Copy {
        
        Cfloat {
            x: self.x,
            y: -self.y
        }
    }

    pub fn inv(&self) -> Cfloat<P> 
        where 
            P: Neg<Output = P>,
            P: Copy {

        Cfloat {
            x: -self.x,
            y: -self.y
        }
    }
}