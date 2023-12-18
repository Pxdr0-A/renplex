use std::ops::{Mul, Add, Div, Rem, AddAssign, Sub};

use super::ops::base::Number;

#[derive(Debug)]
pub struct Lcg<T> {
    a: T,
    b: T,
    m: T,
    pub range: T
}

impl<T> Lcg<T>
    where 
        T: Number,
        T: Copy + AddAssign + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Rem<Output = T> {
    

    pub fn new(a: T, b: T, m: T, range: T) -> Lcg<T> {

        Lcg { a, b, m, range }
    
    }

    pub fn setup() -> Lcg<T> {

        let mut a: T = T::null();
        for _ in 0..1103515245 {
            a += T::unit();
        }

        let mut b: T = T::null();
        for _ in 0..12345 {
            b += T::unit();
        }

        let mut m: T = T::null();
        for _ in 0..2usize.pow(31) {
            m += T::unit();
        }

        Lcg { a, b, m, range: m }
    }

    pub fn set_range(&mut self, range: T) {

        self.range = range;

    }

    pub fn gen(&self, seed: &mut T) -> T {
        
        *seed = (self.a * *seed + self.b) % self.m;

        *seed / self.range

    }

}

/// Returns a random number between 0 and 1 with 2^31 different equal spaced values.
/// 
/// # Arguments
/// 
/// * `seed` - a mutable `u128` with a certain initial value (seed), that will be changing throughout calls.
///
/// # Example
/// ```
/// use renplex::math::random::lcg;
/// 
/// let mut seed = 12345u128;
/// let mut val: f64;
/// for _ in 0..10 {
///     val = lcg(&mut seed);
///     println!("Value Update: {}", val);
///     println!("Seed Update: {}", seed);
/// }
/// ```
pub fn lcgf64(seed: &mut u128) -> f64 {
    // IBM C/C++ convention params
    let a: u128 = 1103515245;
    let b: u128 = 12345;
    let m: u128 = 2u128.pow(31);

    *seed = (a * *seed + b) % (m - 1);
    
    (*seed as f64) / (m as f64)

}

pub fn lcgf32(seed: &mut u128) -> f32 {
    // IBM C/C++ convention params
    let a: u128 = 1103515245;
    let b: u128 = 12345;
    let m: u128 = 2u128.pow(31);

    *seed = (a * *seed + b) % (m - 1);
    
    (*seed as f32) / (m as f32)

}


pub fn lcgi(seed: &mut u128, m: u128) -> usize {
    // IBM C/C++ convention params
    let a: u128 = 1103515245;
    let b: u128 = 12345;

    *seed = (a * *seed + b) % (m - 1);
    
    (*seed as usize) / (m as usize)

}