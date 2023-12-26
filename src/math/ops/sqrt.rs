use crate::math::cfloat::Cfloat;

pub trait SquareRootable {
    fn sqrt(self) -> Self;
}

macro_rules! squarerootify {
    ( $( $t: ty ), * ) => {
        $(
            impl SquareRootable for $t {
                fn sqrt(self) -> $t {
                    self.sqrt()
                }
            }
        )*
    };
}
squarerootify!{f32, f64}


impl<P> SquareRootable for Cfloat<P> {
    
    fn sqrt(self) -> Self {
        todo!()    
    }
}