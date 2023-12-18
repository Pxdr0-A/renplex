pub mod math;
pub mod prelude;


#[cfg(test)]
mod neuronic {

    #[test]
    fn signal() {
        use crate::math::complex::Cfloat;
        use crate::math::ops::base::Complex;

        use crate::prelude::neuron::UnitLike;
        use crate::prelude::neuron::dense::DenseNeuron;
        use crate::prelude::neuron::ActivationFunction;

        let n1 = DenseNeuron::new(
            vec![Cfloat::new(1.2f32, 1.4), Cfloat::new(5.4, 1.1)], 
            Cfloat::new(1.0, 0.5), 
            ActivationFunction::SIGMOID
        );

        let out = n1.signal(
            &[Cfloat::new(1.5, 1.0), Cfloat::new(2.0, 2.0)]
        );

        println!("{:?}", out);
    }
}

#[cfg(test)]
mod layeronic {
    
    #[test]
    fn signal() {
        use crate::math::complex::Cfloat;
        use crate::math::ops::base::Complex;

        use crate::prelude::neuron::UnitLike;
        use crate::prelude::neuron::dense::DenseNeuron;
        use crate::prelude::neuron::ActivationFunction;

        use crate::prelude::layer::LayerLike;
        use crate::prelude::layer::dense::DenseInputLayer;

        let n1 = DenseNeuron::new(
            vec![Cfloat::new(1.2f32, 1.4), Cfloat::new(5.4, 1.1)], 
            Cfloat::new(1.0, 0.5), 
            ActivationFunction::SIGMOID
        );

        let n2 = DenseNeuron::new(
            vec![Cfloat::new(1.2, 1.4), Cfloat::new(5.4, 1.1)], 
            Cfloat::new(1.0, 0.5), 
            ActivationFunction::SIGMOID
        );

        let mut l1 = DenseInputLayer::new(2);
        l1.add(n1);
        l1.add(n2);

        let out = l1.signal(
            &[
                Cfloat::new(1.5, 1.0), Cfloat::new(2.0, 2.0),
                Cfloat::new(1.5, 1.0), Cfloat::new(2.0, 2.0)
            ]
        );

        println!("{:?}", out);

    }
}

#[cfg(test)]
mod networkonic {

    #[test]
    fn forward() {
        use crate::math::complex::Cfloat;
        use crate::math::ops::base::Complex;

        use crate::prelude::neuron::UnitLike;
        use crate::prelude::neuron::dense::DenseNeuron;
        use crate::prelude::neuron::ActivationFunction;

        use crate::prelude::layer::LayerLike;
        use crate::prelude::layer::{InputLayer, Layer};
        use crate::prelude::layer::dense::{DenseLayer, DenseInputLayer};
        use crate::prelude::network::Network;

        let n1 = DenseNeuron::new(
            vec![Cfloat::new(1.2f32, 1.4), Cfloat::new(5.4, 1.1)], 
            Cfloat::new(1.0, 0.5), 
            ActivationFunction::SIGMOID
        );

        let n2 = DenseNeuron::new(
            vec![Cfloat::new(1.2, 1.4), Cfloat::new(5.4, 1.1)], 
            Cfloat::new(1.0, 0.5), 
            ActivationFunction::SIGMOID
        );

        let n3 = DenseNeuron::new(
            vec![Cfloat::new(1.2f32, 1.4), Cfloat::new(5.4, 1.1)], 
            Cfloat::new(1.0, 0.5), 
            ActivationFunction::SIGMOID
        );

        let n4 = DenseNeuron::new(
            vec![Cfloat::new(1.2, 1.4), Cfloat::new(5.4, 1.1)], 
            Cfloat::new(1.0, 0.5), 
            ActivationFunction::SIGMOID
        );

        let mut l1 = DenseInputLayer::new(2);
        l1.add(n1);
        l1.add(n2);

        let mut l2 = DenseLayer::new(2);
        l2.add(n3);
        l2.add(n4);

        let mut network = Network::new();
        network.add_input(InputLayer::DenseInputLayer(l1));
        network.add_layer(Layer::DenseLayer(l2));

        let out = network.forward(
            &[
                Cfloat::new(1.5, 1.0), Cfloat::new(2.0, 2.0),
                Cfloat::new(1.5, 1.0), Cfloat::new(2.0, 2.0)
            ]
        );

        println!("{:?}", out);
        
    }
}

#[cfg(test)]
mod sandbox {

    #[test]
    fn returning_traits() {
        struct Sheep {}
        struct Cow {}

        trait Animal {
            fn noise(&self) -> &'static str;
        }
        
        impl Animal for Cow {
            fn noise(&self) -> &'static str {
                "muuuuuuu"
            }
        }

        impl Animal for Sheep {
            fn noise(&self) -> &'static str {
                "baaaaaaa"
            }
        }

        fn choose(animal: bool) -> Box<dyn Animal> {
            if animal {
                Box::new(Sheep {})
            } else {
                Box::new(Cow {})
            }
        }

        let mut a = Vec::new();

        a.push(choose(true));
        a.push(choose(false));
        a.push(choose(false));

    }
    
}

#[cfg(test)]
mod datasetronic {

    #[test]
    fn sample() {
        use crate::math::matrix::dataset::Dataset;

        let d = Dataset::<f64, usize>::sample(
            [8, 2], 
            2, 
            [100.0, 5.0], 
            &mut (248356u128 as f64), 
            &mut 11652
        );

        println!("{:?}", d);
    }
    
}