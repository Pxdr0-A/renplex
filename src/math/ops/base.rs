use crate::math::cfloat::Cfloat;

pub trait Number {
    fn null() -> Self;

    fn unit() -> Self;

    fn nan() -> Self;

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

                fn nan() -> $t {
                    <$t>::NAN
                }

                fn inf() -> $t {
                    <$t>::INFINITY
                }
            }
        )*
    };
}
numify!{f32, f64}

impl Number for usize {
    fn null() -> usize {
        0
    }

    fn unit() -> usize {
        1
    }

    fn nan() -> usize {
        usize::MIN
    }

    fn inf() -> usize {
        usize::MAX
    }
}

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

    fn nan() -> Self {
        Cfloat {
            x: P::nan(),
            y: P::nan()
        }
    }

    fn inf() -> Self {
        Cfloat {
            x: P::inf(), 
            y: P::inf() 
        }
    }
}


pub trait Real {
    
    fn new(x: Self) -> Self;

}

impl Real for f32 {
    
    fn new(x: Self) -> Self {
        x
    }

}

impl Real for f64 {
    
    fn new(x: Self) -> Self {
        x
    }

}

pub trait Complex<T> {
    
    fn new(x: T, y: T) -> Self;

    fn re(&self) -> T;

    fn im(&self) -> T;

    fn norm(&self) -> T;

    fn phase(&self) -> T;

    fn conj(&self) -> Self;

    fn inv(&self) -> Self;

}