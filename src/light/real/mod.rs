pub mod unit;

pub enum ActivationFunction {
    SIGMOID,
    TANH,
    RELU
}

/// Something that can be updated in the network for the learning stages.
pub trait Param {

    fn null() -> Self;

    fn unit() -> Self;

    fn neg(self) -> Self;

    fn add(self, rhs: Self) -> Self;

    fn add_mut(&mut self, rhs: Self);

    fn sub(self, rhs: Self) -> Self;
    
    fn sub_mut(&mut self, rhs: Self);

    fn mul(self, rhs: Self) -> Self;

    fn mul_mut(&mut self, rhs: Self);

    fn div(self, rhs: Self) -> Self;
    
    fn div_mut(&mut self, rhs: Self);

    fn powi(self, n: i32) -> Self;

    fn act(self, act_func: &ActivationFunction) -> Self;

}

impl Param for f32 {

    fn null() -> Self {
        0.0
    }

    fn unit() -> Self {
        1.0
    }

    fn neg(self) -> Self {
        -self
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

    fn mul_mut(&mut self, rhs: Self) {
        *self *= rhs;
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn div_mut(&mut self, rhs: Self) {
        *self /= rhs;
    }

    fn powi(self, n: i32) -> Self {
        self.powi(n)
    }

    fn act(self, act_func: &ActivationFunction) -> Self {
        match act_func {
            ActivationFunction::SIGMOID => { self.exp() / (1.0 + self.exp()) },
            ActivationFunction::RELU => { if self.is_sign_positive() { self } else { 0.0 } },
            ActivationFunction::TANH => { self.tanh() }
        }
    }

}