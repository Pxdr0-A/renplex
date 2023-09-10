// std
use std::ops::{Add, Sub, Mul, Div, AddAssign};


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

    fn add(self, rhs: Self) -> Cfloat<P> {
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

    fn div(self, rhs: Self) -> Cfloat<P> {
        let den = rhs.x * rhs.x - rhs.y * rhs.y;

        Cfloat { 
            x: (self.x * rhs.x + self.y * rhs.y) / den, 
            y: (self.y * rhs.x - self.x * rhs.y) / den
        }
    }
}

// Generic addition for &Cfloat
impl<'a, 'b, P> Add<&'b Cfloat<P>> for &'a Cfloat<P> where 
    P: Add<Output = P>,
    P: Copy {

    type Output = Cfloat<P>;

    fn add(self, rhs: &'b Cfloat<P>) -> Cfloat<P> {
        Cfloat {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

// Generic addition for &Cfloat
impl<'a, 'b, P> Sub<&'b Cfloat<P>> for &'a Cfloat<P> where 
    P: Sub<Output = P>,
    P: Copy {

    type Output = Cfloat<P>;

    fn sub(self, rhs: &'b Cfloat<P>) -> Cfloat<P> {
        Cfloat {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

// Generic multiplication &Cfloat
impl<'a, 'b, P> Mul<&'b Cfloat<P>> for &'a Cfloat<P> where 
    P: Mul<Output = P> + Add<Output = P> + Sub<Output = P>, 
    P: Copy {
    
    type Output = Cfloat<P>;

    fn mul(self, rhs: &'b Cfloat<P>) -> Cfloat<P> {

        Cfloat { 
            x: self.x * rhs.x - self.y * rhs.y, 
            y: self.x * rhs.y + self.y * rhs.x
        }
    }
}

// Generic division &Cfloat
impl<'a, 'b, P> Div<&'b Cfloat<P>> for &'a Cfloat<P> where 
    P: Div<Output = P> + Mul<Output = P> + Add<Output = P> + Sub<Output = P>, 
    P: Copy {
    
    type Output = Cfloat<P>;

    fn div(self, rhs: &'b Cfloat<P>) -> Cfloat<P> {
        let den = rhs.x * rhs.x - rhs.y * rhs.y;

        Cfloat { 
            x: (self.x * rhs.x + self.y * rhs.y) / den, 
            y: (self.y * rhs.x - self.x * rhs.y) / den
        }
    }
}

// Generic Assign Addition for Cfloat
impl<P> AddAssign for Cfloat<P> where 
    P: AddAssign {

    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}


// Operations for each usable type
macro_rules! make_complex_ops {
    ( $( $t:ty ),* ) => {
        $(
            impl Cfloat<$t>  {
                pub fn phase(&self) -> $t {
                    (self.y / self.x).atan()
                }
            
                pub fn norm(&self) -> $t {
                    self.x.powi(2) + self.y.powi(2)
                }

                pub fn conj(&self) -> Cfloat<$t> {
                    Cfloat {
                        x: self.x,
                        y: -self.y
                    }
                }

                pub fn inv(&self) -> Cfloat<$t> {
                    Cfloat {
                        x: -self.x,
                        y: -self.y
                    }
                }
            
                pub fn exp(&self) -> Cfloat<$t> {
                    Cfloat { 
                        x: self.x.exp() * self.y.cos(), 
                        y: self.x.exp() * self.y.sin()
                    }
                }
            
                pub fn tanh(&self) -> Cfloat<$t> {
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
        )*
    };
}

make_complex_ops!{f32, f64}