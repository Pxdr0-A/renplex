// std
use std::ops::{Add, Sub, Mul, Div, Neg};

// local
use crate::math::complex::Cfloat;
use crate::math::ops::base::Number;
use crate::math::ops::{
    exp::Exponentiable, 
    trig::Trignometricable, 
    sign::Signable
};

#[derive(Debug, Clone)]
pub enum ActivationFunction {
    SIGMOID,
    TANH,
    RELU,
}

pub trait Activatable {
    fn activation(self, act_func: &ActivationFunction) -> Self;
}


// for primitives (that obey the where condition defined bellow)
impl<P> Activatable for P 
    where
        P: Add<Output=P> + Sub<Output=P> + Mul<Output=P> + Div<Output=P> + Neg<Output=P>,
        P: PartialEq,
        P: Exponentiable + Trignometricable + Signable + Number,
        P: Copy {

    fn activation(self, act_func: &ActivationFunction) -> Self {
        assert!(
            self != self + self,
            "Division by zero encountered in primitives."
        );

        match act_func {
            ActivationFunction::SIGMOID => { self.exp() / (self.unit() + self.exp()) },
            ActivationFunction::RELU => { if self.is_sign_positive() { self } else { self.null() } },
            ActivationFunction::TANH => { self.tanh() }
        }
    }
}

// for complex numbers
impl<P> Activatable for Cfloat<P> 
    where
        P: Add<Output=P> + Sub<Output=P> + Mul<Output=P> + Div<Output=P> + Neg<Output=P>,
        P: PartialEq, 
        P: Exponentiable + Trignometricable + Signable + Number,
        P: Copy {
            
    fn activation(self, act_func: &ActivationFunction) -> Self {

        match act_func {
            ActivationFunction::SIGMOID => { self.exp() / (self.unit() + self.exp()) },
            ActivationFunction::RELU => { if self.is_sign_positive() { self } else { self.null() } },
            ActivationFunction::TANH => { self.tanh() }
        }
    }
}
