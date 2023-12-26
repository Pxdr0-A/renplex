pub enum CActivationFunction {
    
}

pub trait Param {

    fn new() -> Self;

    fn null() -> Self;

    fn unit() -> Self;

    fn neg(self) -> Self;

    fn add(self, rhs: Self) -> Self;

    fn add_mut(&mut self, rhs: Self);

    fn sub(self, rhs: Self) -> Self;
    
    fn sub_mut(&mut self, rhs: Self);

    fn mul(self, rhs: Self) -> Self;

    fn powi(self, n: i32) -> Self;

    fn act(self, act_func: &CActivationFunction) -> Self;

}