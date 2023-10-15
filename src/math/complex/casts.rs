use super::Cfloat;

pub trait ComplexCast<P> {
    fn to_complex(self) -> Cfloat<P>;
}


impl ComplexCast<f32> for f32 {
    fn to_complex(self) -> Cfloat<f32> {
        Cfloat {
            x: self, 
            y: 0.0 
        }
    }
}

impl ComplexCast<f64> for f64 {
    fn to_complex(self) -> Cfloat<f64> {
        Cfloat {
            x: self, 
            y: 0.0 
        }
    }
}