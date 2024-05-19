//! Complex properties, operations and castings.
//! 
//! Provides simple tools to handle complex numbers with generic types. Nevertheless, 
//! these tools are based on f32 and f64 which are the only primitives that implement
//! the traits requested by the tools.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::fmt::Display;

use super::Complex;


#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Cf32 {
    pub x: f32,
    pub y: f32
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Cf64 {
    pub x: f64,
    pub y: f64
}

impl Display for Cf32 {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /*
    let precision = 2;
    let multi = 10_f32.powi(precision);
    let norm = ( self.norm() * multi ).round() / multi;
    let phase = ( self.phase() * multi ).round() / multi;
    write!(f, "{}ø{}", norm, phase)
    */
    //let precision = 2;
    //let multi = 10_f32.powi(precision);
    write!(f, "{:?}", [self.x, self.y])
  }
}

impl Display for Cf64 {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /*
    let precision = 2;
    let multi = 10_f64.powi(precision);
    let norm = ( self.norm() * multi ).round() / multi;
    let phase = ( self.phase() * multi ).round() / multi;
    write!(f, "{}ø{}", norm, phase)
    */
    //let precision = 2;
    //let multi = 10_f64.powi(precision);
    write!(f, "{:?}", [self.x, self.y])  
  }
}

impl Default for Cf32 {
  fn default() -> Self {
    Cf32 {
      x: 0.0,
      y: 0.0
    }
  }
}

impl Default for Cf64 {
  fn default() -> Self {
    Cf64 {
      x: 0.0,
      y: 0.0
    }
  }
}

impl PartialOrd for Cf64 {
  fn ge(&self, other: &Self) -> bool {
    let norm_lhs = self.norm_sq();
    let norm_rhs = other.norm_sq();

    norm_lhs >= norm_rhs
  }

  fn gt(&self, other: &Self) -> bool {
    let norm_lhs = self.norm_sq();
    let norm_rhs = other.norm_sq();

    norm_lhs > norm_rhs
  }

  fn le(&self, other: &Self) -> bool {
    let norm_lhs = self.norm_sq();
    let norm_rhs = other.norm_sq();

    norm_lhs <= norm_rhs
  }

  fn lt(&self, other: &Self) -> bool {
    let norm_lhs = self.norm_sq();
    let norm_rhs = other.norm_sq();

    norm_lhs < norm_rhs
  }

  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    let norm_lhs = self.norm_sq();
    let norm_rhs = other.norm_sq();

    norm_lhs.partial_cmp(&norm_rhs)
  }
}

impl PartialOrd for Cf32 {
  fn ge(&self, other: &Self) -> bool {
    let norm_lhs = self.norm_sq();
    let norm_rhs = other.norm_sq();

    norm_lhs >= norm_rhs
  }

  fn gt(&self, other: &Self) -> bool {
    let norm_lhs = self.norm_sq();
    let norm_rhs = other.norm_sq();

    norm_lhs > norm_rhs
  }

  fn le(&self, other: &Self) -> bool {
    let norm_lhs = self.norm_sq();
    let norm_rhs = other.norm_sq();

    norm_lhs <= norm_rhs
  }

  fn lt(&self, other: &Self) -> bool {
    let norm_lhs = self.norm_sq();
    let norm_rhs = other.norm_sq();

    norm_lhs < norm_rhs
  }

  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    let norm_lhs = self.norm_sq();
    let norm_rhs = other.norm_sq();

    norm_lhs.partial_cmp(&norm_rhs)
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
    let den = rhs.x.powi(2) + rhs.y.powi(2);

    if den == 0.0 { panic!("Division by zero encountered in complex numbers.") }

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
    let den = rhs.x.powi(2) + rhs.y.powi(2);

    if den == 0.0 { panic!("Division by zero encountered in complex numbers.") }

    self.x = (self.x * rhs.x + self.y * rhs.y) / den;
    self.y = (self.y * rhs.x - self.x * rhs.y) / den;
  }
}

// Operations with references (incomplete)
impl<'a, 'b> Add<&'b Cf32> for &'a Cf32 {
  type Output = Cf32;

  fn add(self, rhs: &'b Cf32) -> Self::Output {
    Cf32 {
      x: self.x + rhs.x,
      y: self.y + rhs.y
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
    let den = rhs.x.powi(2) + rhs.y.powi(2);

    if den == 0.0 { panic!("Division by zero encountered in complex numbers.") }

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
    let den = rhs.x.powi(2) + rhs.y.powi(2);

    if den == 0.0 { panic!("Division by zero encountered in complex numbers.") }

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
    let den = rhs.x.powi(2) + rhs.y.powi(2);

    if den == 0.0 { panic!("Division by zero encountered in complex numbers.") }

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
      y: self.y + rhs.y
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
    let den = rhs.x.powi(2) + rhs.y.powi(2);

    if den == 0.0 { panic!("Division by zero encountered in complex numbers.") }

    Cf64 { 
      x: (self.x * rhs.x + self.y * rhs.y) / den, 
      y: (self.y * rhs.x - self.x * rhs.y) / den
    }
  }
}
