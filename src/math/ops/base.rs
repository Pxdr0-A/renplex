use crate::math::complex::Cfloat;

pub trait Number {
    fn null() -> Self;

    fn unit() -> Self;

    fn inf() -> Self;

    // you can also define a vestigial quantity
    // see float::EPSILON
}

macro_rules! numify {
    ( $( $t: ty ), * ) => {
        $(
            impl Number for $t {
                fn null() -> $t {
                    0.0
                }

                fn unit() -> $t {
                    1.0
                }

                fn inf() -> $t {
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
        
    fn null() -> Self {
        Cfloat { 
            x: P::null(),
            y: P::null()
        }
    }

    fn unit() -> Self {
        Cfloat { 
            x: P::unit(),
            y: P::null()
        }
    }

    fn inf() -> Self {
        Cfloat {
            x: P::inf(), 
            y: P::inf() 
        }
    }
}