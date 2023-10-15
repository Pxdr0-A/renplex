use crate::math::complex::Cfloat;

pub trait Number {
    fn null(self) -> Self;

    fn unit(self) -> Self;

    fn inf(self) -> Self;

    // you can also define a vestigial quantity
    // see float::EPSILON
}

macro_rules! numify {
    ( $( $t: ty ), * ) => {
        $(
            impl Number for $t {
                fn null(self) -> $t {
                    0.0
                }

                fn unit(self) -> $t {
                    1.0
                }

                fn inf(self) -> $t {
                    <$t>::INFINITY
                }
            }
        )*
    };
}
numify!{f32, f64}


impl<P> Number for Cfloat<P>
    where 
        P: Number {
        
    fn null(self) -> Self {
        Cfloat { 
            x: self.x.null(),
            y: self.y.null()
        }
    }

    fn unit(self) -> Self {
        Cfloat { 
            x: self.x.unit(),
            y: self.y.unit()
        }
    }

    fn inf(self) -> Self {
        Cfloat {
            x: self.x.inf(), 
            y: self.y.inf() 
        }
    }
}