// std
use std::ops::{Add, Sub, Neg, Mul, Div, AddAssign, SubAssign};

// local
use super::Cfloat;


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

impl<P> Neg for Cfloat<P> where
    P: Neg<Output = P> {

    type Output = Cfloat<P>;

    fn neg(self) -> Self::Output {
        Cfloat {
            x: - self.x,
            y: - self.y
        }
    }
}

// Generic multiplication for Cfloat
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

// Generic division for Cfloat
impl<P> Div for Cfloat<P> where 
    P: Mul<Output = P> + Div<Output = P> + Add<Output = P> + Sub<Output = P>,
    P: PartialEq,
    P: Copy {
    
    type Output = Cfloat<P>;

    fn div(self, rhs: Self) -> Cfloat<P> {
        let den = rhs.x * rhs.x - rhs.y * rhs.y;

        assert!(
            den != den + den,
            "Division by zero encountered in complex numbers."
        );

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

// Generic subtraction for &Cfloat
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

// Generic Assign Addition for Cfloat
impl<P> SubAssign for Cfloat<P> where 
    P: SubAssign {

    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}