pub mod unit;
pub mod layer;
pub mod network;

use crate::math::cfloat::{Cf32, Cf64};

#[derive(Debug, Clone)]
pub enum ComplexActFunction {
    RITSIGMOID,
    RITTANH
}

impl ComplexActFunction {
    pub fn compute_f32(&self, val: Cf32) -> Cf32 {
        match self {
            ComplexActFunction::RITSIGMOID => {
                Cf32 {
                    x: val.x.exp() / (1.0 + val.x.exp()),
                    y: val.y.exp() / (1.0 + val.y.exp()) 
                }
            },

            ComplexActFunction::RITTANH => {
                Cf32 {
                    x: val.x.tanh(),
                    y: val.y.tanh() 
                }
            }
        }
    }

    pub fn compute_f64(&self, val: Cf64) -> Cf64 {
        match self {
            ComplexActFunction::RITSIGMOID => {
                Cf64 {
                    x: val.x.exp() / (1.0 + val.x.exp()),
                    y: val.y.exp() / (1.0 + val.y.exp()) 
                }
            },

            ComplexActFunction::RITTANH => {
                Cf64 {
                    x: val.x.tanh(),
                    y: val.y.tanh() 
                }
            }
        }
    }

    pub fn derive_bias<CP: ComplexParam>(&self, val: CP) -> CP {
        match self {
            ComplexActFunction::RITSIGMOID => {
                val.inv()
            },
            
            ComplexActFunction::RITTANH => {
                val.inv()
            }
        }
    }
}

pub trait ComplexParam {

    fn null() -> Self;

    fn unit() -> Self;

    fn iunit() -> Self;

    fn conj(self) -> Self;
    
    fn inv(self) -> Self;

    fn add(self, rhs: Self) -> Self;

    fn add_mut(&mut self, rhs: Self);

    fn sub(self, rhs: Self) -> Self;
    
    fn sub_mut(&mut self, rhs: Self);

    fn mul(self, rhs: Self) -> Self;

    fn mul_mut(&mut self, rhs: Self);

    fn div(self, rhs: Self) -> Self;
    
    fn div_mut(&mut self, rhs: Self);

    fn act(self, act_func: &ComplexActFunction) -> Self;

}

// Param implementations for Cf32
impl ComplexParam for Cf32 {
    fn null() -> Self {
        Self {
            x: 0.0,
            y: 0.0
        }
    }

    fn unit() -> Self {
        Self {
            x: 1.0,
            y: 0.0
        }
    }

    fn iunit() -> Self {
        Self { 
            x: 0.0, 
            y: 1.0
        }
    }

    fn conj(self) -> Self {
        Self { 
            x: self.x, 
            y: -self.y
        }
    }

    fn inv(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y
        }
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

    fn act(self, act_func: &ComplexActFunction) -> Self {
        act_func.compute_f32(self)
    }

}

// Param implementations for Cf64
impl ComplexParam for Cf64 {
    fn null() -> Self {
        Self {
            x: 0.0,
            y: 0.0
        }
    }

    fn unit() -> Self {
        Self {
            x: 1.0,
            y: 0.0
        }
    }

    fn iunit() -> Self {
        Self { 
            x: 0.0, 
            y: 1.0
        }
    }

    fn conj(self) -> Self {
        Self { 
            x: self.x, 
            y: -self.y
        }
    }

    fn inv(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y
        }
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

    fn act(self, act_func: &ComplexActFunction) -> Self {
        act_func.compute_f64(self)
    }

}