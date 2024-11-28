pub mod io;
pub mod math;

pub trait Module {
    // Option 1
    // two shape type
    // two const types for the lens

    // Option 2
    // ...

    fn new() -> Self;

    fn forward(&self, input: io::Input<'a, DIMINP>) -> io::Output<'a, DIMOUT>;

    fn backward(&self, output: io::Output<'a, DIMINP>) -> io::Input<'a, DIMOUT>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
