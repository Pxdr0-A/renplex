pub trait Arcable {
    fn asin(self) -> Self;

    fn acos(self) -> Self;

    fn atan(self) -> Self; 

    fn asinh(self) -> Self;

    fn acosh(self) -> Self;

    fn atanh(self) -> Self;
}


macro_rules! arcify {
    ( $( $t: ty ), * ) => {
        $(
            impl Arcable for $t {
                fn asin(self) -> Self {
                    self.asin()
                }
            
                fn acos(self) -> Self {
                    self.acos()
                }
            
                fn atan(self) -> Self {
                    self.atan()
                }

                fn asinh(self) -> Self {
                    self.asinh()
                }

                fn acosh(self) -> Self {
                    self.acosh()
                }
            
                fn atanh(self) -> Self {
                    self.atanh()
                }
            }
        )*
    };
}

arcify!{f32, f64}
