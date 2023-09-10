use crate::math::complex::Cfloat;


pub enum ActivationFunction {
    SIGMOID,
    TANH,
    RELU,
}

pub trait Activation {
    fn activation(&self, func: &ActivationFunction) -> Self;
}

macro_rules! activate_float {
    ($value:ident, $func:ident, $t:ty) => {
        match $func {
            ActivationFunction::SIGMOID => {
                let one: $t = 1.0;
                one / (one + (-$value).exp())
            },
            ActivationFunction::TANH => $value.tanh(),
            ActivationFunction::RELU => {
                if $value.is_sign_positive() {
                    *$value
                } else {
                    let zero: $t = 0.0;
                    zero
                }
            }
        }
    };
}

macro_rules! activate_complex {
    ($value:ident, $func:ident, $t:ty) => {
        match $func {
            ActivationFunction::SIGMOID => {
                // hard coded float n.d (I do not think there is a integer solution)
                let x: $t = 1.0;
                let y: $t = 0.0;
                Cfloat::new(x, y) / (Cfloat::new(x, y) + ($value.inv()).exp())
            },
            ActivationFunction::TANH => $value.tanh(),
            ActivationFunction::RELU => {
                if $value.is_sign_positive() {
                    *$value
                } else {
                    let zero: $t = 0.0;
                    Cfloat::new(zero, zero)
                }
            }
        }
    };
}

macro_rules! activate_all {
    ($( $t:ty ),*) => {
        $(
            impl Activation for $t {
                fn activation(&self, func: &ActivationFunction) -> $t {
                    activate_float!(self, func, $t)
                }
            }  

            impl Activation for Cfloat<$t> {
                fn activation(&self, func: &ActivationFunction) -> Cfloat<$t> {
                    activate_complex!(self, func, $t)
                }
            }
        )*         
    };
}
 
// defines activation method for f32, f64, Cfloat32 and Cfloat64
activate_all!{f32, f64}
