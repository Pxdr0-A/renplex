pub mod unit;
pub mod layer;
pub mod network;

#[derive(Debug, Clone)]
pub enum ActFunction {
    SIGMOID,
    TANH,
    RELU
}

impl ActFunction {
    pub fn compute_f32(&self, val: f32) -> f32 {
        match self {
            ActFunction::SIGMOID => {
                val.exp() / (1.0 + val.exp())
            },

            ActFunction::TANH => {
                val.tanh()
            },
            
            ActFunction::RELU => {
                if val.is_sign_positive() { val } else { 0.0 }
            }
        }
    }

    pub fn compute_f64(&self, val: f64) -> f64 {
        match self {
            ActFunction::SIGMOID => {
                val.exp() / (1.0 + val.exp())
            },

            ActFunction::TANH => {
                val.tanh()
            },
            
            ActFunction::RELU => {
                if val.is_sign_positive() { val } else { 0.0 }
            }
        }
    }
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

    fn act(self, act_func: &ActFunction) -> Self;

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

    fn act(self, act_func: &ActFunction) -> Self {
        act_func.compute_f32(self)
    }

}

impl Param for f64 {

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

    fn act(self, act_func: &ActFunction) -> Self {
        act_func.compute_f64(self)
    }

}