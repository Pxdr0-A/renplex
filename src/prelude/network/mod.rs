use super::layer::HiddenLayer;

pub struct Network<W> {
    pub layers: Vec<HiddenLayer<W>>
}

impl<W> Network<W> {
    pub fn new() -> Network<W> {
        Network {
            layers: Vec::<HiddenLayer<W>>::new()
        }
    }
}