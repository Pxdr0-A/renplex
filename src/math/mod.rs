/* Math: Arithmetic, Operations and Structures

*/

// std
use std::ops::{Add, Sub, Mul, Div};

// local
pub mod matrix;
pub mod random;
pub mod backpropagation;


#[derive(Debug, Clone, Copy)]
pub struct Cfloat<P> {
    x: P,
    y: P
}

impl<P> Cfloat<P> {
    pub fn new(x: P, y: P) -> Cfloat<P> {
        Cfloat {x, y}
    }
}

// Generic addition for Cfloat
impl<P> Add for Cfloat<P> where 
    P: Add<Output = P> {
    
    type Output = Cfloat<P>;

    fn add(self, rhs: Cfloat<P>) -> Cfloat<P> {
        Cfloat { 
            x: self.x + rhs.x,
            y: self.y + rhs.y
        }
    }
}

// Generic subtraction for Cfloat
impl<P> Sub for Cfloat<P> where 
    P: Sub<Output = P> {
    
    type Output = Cfloat<P>;

    fn sub(self, rhs: Self) -> Self::Output {
        Cfloat {
            x: self.x - rhs.x,
            y: self.y - rhs.y
        }
    }
}

// Generic multiplication
impl<P> Mul for Cfloat<P> where 
    P: Mul<Output = P> + Add<Output = P> + Sub<Output = P>, 
    P: Copy {
    
    type Output = Cfloat<P>;

    fn mul(self, rhs: Self) -> Cfloat<P> {

        Cfloat { 
            x: self.x * rhs.x - self.y * rhs.y, 
            y: self.x * rhs.y + self.y * rhs.x
        }
    }
}

// Generic division
impl<P> Div for Cfloat<P> where 
    P: Mul<Output = P> + Div<Output = P> + Add<Output = P> + Sub<Output = P>, 
    P: Copy {
    
    type Output = Cfloat<P>;

    fn div(self, rhs: Cfloat<P>) -> Cfloat<P> {
        let den = rhs.x * rhs.x - rhs.y * rhs.y;

        Cfloat { 
            x: (self.x * rhs.x + self.y * rhs.y) / den, 
            y: (self.y * rhs.x - self.x * rhs.y) / den
        }
    }
}


// You may need to implement these ops for &Cfloat
// Generic addition for &Cfloat
// Review this!!
impl<P> Add for &Cfloat<P> where 
    P: Add<Output = P>,
    P: Copy {
    
    type Output = Cfloat<P>;

    fn add(self, rhs: &Cfloat<P>) -> Cfloat<P> {
        Cfloat { 
            x: self.x + rhs.x,
            y: self.y + rhs.y
        }
    }
}


// Please build this with a macro
impl Cfloat<f32>  {
    
}

impl Cfloat<f64>  {
    pub fn phase(&self) -> f64 {
        (self.y / self.x).atan()
    }

    pub fn norm(&self) -> f64 {
        self.x.powi(2) + self.y.powi(2)
    }

    pub fn exp(&self) -> Cfloat<f64> {
        Cfloat { 
            x: self.x.exp() * self.y.cos(), 
            y: self.x.exp() * self.y.sin()
        }
    }

    pub fn tanh(&self) -> Cfloat<f64> {
        let den = 
            self.y.cos().powi(2) * self.x.cosh().powi(2) - 
            self.y.sin().powi(2) * self.x.sinh().powi(2);

        Cfloat {
            x: self.x.sinh() * self.x.cosh() / den,
            y: self.y.sin() * self.y.cos() / den
        }
    }

    pub fn is_sign_positive(&self) -> bool {
        if self.phase().is_sign_positive() { true } else { false }
    }
}