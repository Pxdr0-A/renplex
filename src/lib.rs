pub mod math;

pub trait Module {
    // Option 1
    // two shape type
    // two const types for the lens

    // Option 2
    // ...

    fn new() -> Self;

    fn forward(&self);

    fn backward(&self);
}
