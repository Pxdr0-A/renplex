//! Complex properties, operations and castings.
//! 
//! Provides simple tools to handle complex numbers with generic types. Nevertheless, 
//! these tools are based on f32 and f64 which are the only primitives that implement
//! the traits requested by the tools.

use std::ops::{
    Add, Sub, Mul, Div,
    AddAssign, SubAssign, MulAssign, DivAssign
};

#[derive(Copy, Clone)]
pub struct Cf32 {
    pub x: f32,
    pub y: f32
}

#[derive(Copy, Clone)]
pub struct Cf64 {
    pub x: f64,
    pub y: f64
}

// General operations for Cf32
impl Cf32 {
    pub fn re(&self) -> f32 {
        self.x
    }

    pub fn im(&self) -> f32 {
        self.y
    }

    pub fn norm(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }

    pub fn phase(&self) -> f32 {
        (self.y / self.x).tan()
    }
}

// General operations for Cf64
impl Cf64 {
    pub fn re(&self) -> f64 {
        self.x
    }

    pub fn im(&self) -> f64 {
        self.y
    }

    pub fn norm(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }

    pub fn phase(&self) -> f64 {
        (self.y / self.x).tan()
    }
} 


// Arithmetic Operations for Cf32

// Normal operations
impl Add for Cf32 {
            
    type Output = Cf32;

    fn add(self, rhs: Self) -> Self::Output {
        Cf32 { 
            x: self.x + rhs.x,
            y: self.y + rhs.y
        }
    }
}

impl Sub for Cf32 {
    
    type Output = Cf32;

    fn sub(self, rhs: Self) -> Self::Output {
        Cf32 {
            x: self.x - rhs.x,
            y: self.y - rhs.y
        }
    }
}

impl Mul for Cf32 {
    
    type Output = Cf32;

    fn mul(self, rhs: Self) -> Self::Output {
        Cf32 { 
            x: self.x * rhs.x - self.y * rhs.y, 
            y: self.x * rhs.y + self.y * rhs.x
        }
    }
}

impl Div for Cf32 {
    
    type Output = Cf32;

    fn div(self, rhs: Self) -> Self::Output {
        let den = rhs.x * rhs.x - rhs.y * rhs.y;

        if den != 0.0 { panic!("Division by zero encountered in complex numbers.") }

        Cf32 { 
            x: (self.x * rhs.x + self.y * rhs.y) / den, 
            y: (self.y * rhs.x - self.x * rhs.y) / den
        }
    }
}

// Assignement operations
impl AddAssign for Cf32 {

    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl SubAssign for Cf32 {

    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl MulAssign for Cf32 {

    fn mul_assign(&mut self, rhs: Self) {
        self.x = self.x * rhs.x - self.y * rhs.y;
        self.y = self.x * rhs.y + self.y * rhs.x;
    }
}

impl DivAssign for Cf32 {
    
    fn div_assign(&mut self, rhs: Self) {
        let den = rhs.x * rhs.x - rhs.y * rhs.y;

        if den != 0.0 { panic!("Division by zero encountered in complex numbers.") }

        self.x = (self.x * rhs.x + self.y * rhs.y) / den;
        self.y = (self.y * rhs.x - self.x * rhs.y) / den;
    }
}

// Operations with references (not complete)
impl<'a, 'b> Add<&'b Cf32> for &'a Cf32 {

    type Output = Cf32;

    fn add(self, rhs: &'b Cf32) -> Self::Output {
        Cf32 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<'a, 'b> Sub<&'b Cf32> for &'a Cf32 {

    type Output = Cf32;

    fn sub(self, rhs: &'b Cf32) -> Self::Output {
        Cf32 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<'a, 'b> Mul<&'b Cf32> for &'a Cf32 {
    
    type Output = Cf32;

    fn mul(self, rhs: &'b Cf32) -> Self::Output {

        Cf32 { 
            x: self.x * rhs.x - self.y * rhs.y, 
            y: self.x * rhs.y + self.y * rhs.x
        }
    }
}

impl<'a, 'b> Div<&'b Cf32> for &'a Cf32 {
    
    type Output = Cf32;

    fn div(self, rhs: &'b Cf32) -> Self::Output{
        let den = rhs.x * rhs.x - rhs.y * rhs.y;

        if den != 0.0 { panic!("Division by zero encountered in complex numbers.") }

        Cf32 { 
            x: (self.x * rhs.x + self.y * rhs.y) / den, 
            y: (self.y * rhs.x - self.x * rhs.y) / den
        }
    }
}


// Operations for Cf64

// Normal operations
impl Add for Cf64 {
            
    type Output = Cf64;

    fn add(self, rhs: Self) -> Self::Output {
        Cf64 { 
            x: self.x + rhs.x,
            y: self.y + rhs.y
        }
    }
}

impl Sub for Cf64 {
    
    type Output = Cf64;

    fn sub(self, rhs: Self) -> Self::Output {
        Cf64 {
            x: self.x - rhs.x,
            y: self.y - rhs.y
        }
    }
}

impl Mul for Cf64 {
    
    type Output = Cf64;

    fn mul(self, rhs: Self) -> Self::Output {
        Cf64 { 
            x: self.x * rhs.x - self.y * rhs.y, 
            y: self.x * rhs.y + self.y * rhs.x
        }
    }
}

impl Div for Cf64 {
    
    type Output = Cf64;

    fn div(self, rhs: Self) -> Self::Output {
        let den = rhs.x * rhs.x - rhs.y * rhs.y;

        if den != 0.0 { panic!("Division by zero encountered in complex numbers.") }

        Cf64 { 
            x: (self.x * rhs.x + self.y * rhs.y) / den, 
            y: (self.y * rhs.x - self.x * rhs.y) / den
        }
    }
}

// Assignement operations
impl AddAssign for Cf64 {

    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

// Assignement Operations
impl SubAssign for Cf64 {

    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl MulAssign for Cf64 {

    fn mul_assign(&mut self, rhs: Self) {
        self.x = self.x * rhs.x - self.y * rhs.y;
        self.y = self.x * rhs.y + self.y * rhs.x;
    }
}

impl DivAssign for Cf64 {
    
    fn div_assign(&mut self, rhs: Self) {
        let den = rhs.x * rhs.x - rhs.y * rhs.y;

        if den != 0.0 { panic!("Division by zero encountered in complex numbers.") }

        self.x = (self.x * rhs.x + self.y * rhs.y) / den;
        self.y = (self.y * rhs.x - self.x * rhs.y) / den;
    }
}

// Operations with references (not complete)
impl<'a, 'b> Add<&'b Cf64> for &'a Cf64 {

    type Output = Cf64;

    fn add(self, rhs: &'b Cf64) -> Self::Output {
        Cf64 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<'a, 'b> Sub<&'b Cf64> for &'a Cf64 {

    type Output = Cf64;

    fn sub(self, rhs: &'b Cf64) -> Self::Output {
        Cf64 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<'a, 'b> Mul<&'b Cf64> for &'a Cf64 {
    
    type Output = Cf64;

    fn mul(self, rhs: &'b Cf64) -> Self::Output {

        Cf64 { 
            x: self.x * rhs.x - self.y * rhs.y, 
            y: self.x * rhs.y + self.y * rhs.x
        }
    }
}

impl<'a, 'b> Div<&'b Cf64> for &'a Cf64 {
    
    type Output = Cf64;

    fn div(self, rhs: &'b Cf64) -> Self::Output{
        let den = rhs.x * rhs.x - rhs.y * rhs.y;

        if den != 0.0 { panic!("Division by zero encountered in complex numbers.") }

        Cf64 { 
            x: (self.x * rhs.x + self.y * rhs.y) / den, 
            y: (self.y * rhs.x - self.x * rhs.y) / den
        }
    }
}