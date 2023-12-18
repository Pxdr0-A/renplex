use std::ops::{Add, Mul, Neg, Sub, Div, AddAssign, SubAssign};

use crate::math::ops::{base::Number, exp::Exponentiable, trig::Trignometricable, sign::Signable, powi::Powerable};

use super::ActivationFunction;


pub trait Param {

    fn new() -> Self;

    fn null() -> Self;

    fn unit() -> Self;

    fn neg(self) -> Self;

    fn add(self, rhs: Self) -> Self;

    fn add_mut(&mut self, rhs: Self);

    fn sub(self, rhs: Self) -> Self;
    
    fn sub_mut(&mut self, rhs: Self);

    fn mul(self, rhs: Self) -> Self;

    fn powi(self, n: i32) -> Self;

    fn act(self, act_func: &ActivationFunction) -> Self;

}


impl<P> Param for P 
    where 
        P: AddAssign + SubAssign + Add<Output=P> + Sub<Output=P> + Mul<Output=P> + Div<Output=P> + Neg<Output=P>,
        P: Copy,
        P: Powerable + Exponentiable + Trignometricable + Signable + Number {

    fn new() -> Self {

        P::unit()
    
    }

    fn null() -> Self {

        P::null()

    }

    fn unit() -> Self {

        P::unit()

    }

    fn add(self, rhs: Self) -> Self {
        
        self + rhs

    }

    fn add_mut(&mut self, rhs: Self) {
        
        *self += rhs;

    }

    fn sub(self, rhs: Self) -> Self {
        
        self - rhs

    }

    fn sub_mut(&mut self, rhs: Self) {
        
        *self -= rhs;

    }

    fn mul(self, rhs: Self) -> Self {

        self * rhs

    }

    fn powi(self, n: i32) -> Self {

        self.powi(n)

    }

    fn neg(self) -> Self {

        - self
         
    }

    fn act(self, act_func: &ActivationFunction) -> Self {
        
        match act_func {
            ActivationFunction::SIGMOID => { self.exp() / (P::unit() + self.exp()) },
            ActivationFunction::RELU => { if self.is_sign_positive() { self } else { P::null() } },
            ActivationFunction::TANH => { self.tanh() }
        }

    }
}
