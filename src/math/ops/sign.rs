use crate::math::cfloat::Cfloat;

pub trait Signable {
    fn is_sign_positive(self) -> bool;
}

macro_rules! signify {
    ( $( $t: ty ), * ) => {
        $(
            impl Signable for $t {

                fn is_sign_positive(self) -> bool {
                    self.is_sign_positive()
                }
            }
        )*
    };
}
signify!{f32, f64}


impl<P> Signable for Cfloat<P> 
    where 
        P: Signable {

    fn is_sign_positive(self) -> bool {
        
        self.x.is_sign_positive()
    }
}