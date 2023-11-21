pub mod math;
pub mod prelude;


#[cfg(test)]
mod ops {
    use crate::math::ops::trig::Trignometricable;

    use super::*;

    #[test]
    fn complex_ops() {
        use math::complex::Cfloat;

        let z = Cfloat::new(2.0, 3.0);
        
        z.re();
        z.im();
        z.norm();
        z.phase();
        z.conj();
        z.inv();
    }

    #[test]
    fn complex_trig() {
        use math::complex::Cfloat;

        let z = Cfloat::new(2.0, 3.0);
        
        z.re();
        z.im();
        z.norm();
        z.phase();
        z.conj();
        z.inv();

        z.sin();
        z.sinh();
        z.cos();
        z.cosh();
        z.tan();
        z.tanh();
    }
}

#[cfg(test)]
mod test_network {

    use super::*;

    #[test]
    fn random_network() {
        use prelude::network::DenseNetwork;
        use prelude::neuron::activation::ActivationFunction;
        use math::matrix::dataset::Dataset;

        let mut net = DenseNetwork::<f64>::init(
            6, 
            2, 
            ActivationFunction::SIGMOID, 
            1.0, 
            1.0,
            &mut 898637u128
        );

        net.add(
            10, 
            ActivationFunction::SIGMOID,  
            1.0, 
            1.0, 
            &mut 287364u128
        );

        net.add(
            2, 
            ActivationFunction::SIGMOID, 
            1.0, 
            1.0, 
            &mut 82157364u128
        );

        let mut seed: u128 = 987234485;
        let data  = Dataset::<f64, u8>::sample(
            //  2 * 3
            [64, 6], 
            2, 
            &mut seed
        );

        net.fit(
            data,
        );

    }

    #[test]
    fn fit_test() {
        use prelude::neuron::Neuron;
        use prelude::neuron::activation::ActivationFunction;
        use prelude::layer::Layer;
        use prelude::network::DenseNetwork;
        use prelude::network::criteria::ComplexCritiria;
        use math::complex::Cfloat;
        use math::matrix::dataset::Dataset;

        let mut net: DenseNetwork<Cfloat<f64>> = DenseNetwork::new(
            3
        );

        // added two hidden layers
        net.add_layer(Layer::new(2));
        net.add_layer(Layer::new(2));

        // 3 input Neurons
        net.add_unit(
            0, 
            Neuron::new(
                vec![Cfloat::new(0.5, 0.1), Cfloat::new(0.4, 0.1)], 
                Cfloat::new(-1.0, -1.0), 
                ActivationFunction::RELU
            )
        );
        net.add_unit(
            0, 
            Neuron::new(
                vec![Cfloat::new(0.7, 0.2), Cfloat::new(-0.4, 1.1)], 
                Cfloat::new(-1.0, -1.0), 
                ActivationFunction::RELU
            )
        );
        net.add_unit(
            0, 
            Neuron::new(
                vec![Cfloat::new(0.2, 0.5), Cfloat::new(0.1, -0.9)], 
                Cfloat::new(-1.0, -1.0), 
                ActivationFunction::RELU
            )
        );

        // add 2 neurons two the first hidden layer
        // input of them must be 3 (3 neurons in the input)
        net.add_unit(
            1, 
            Neuron::new(
                vec![Cfloat::new(0.1, 0.1), Cfloat::new(-0.1, -0.1), Cfloat::new(0.2, 0.1)], 
                Cfloat::new(-1.0, -1.0), 
                ActivationFunction::TANH
            )
        );

        net.add_unit(
            1, 
            Neuron::new(
                vec![Cfloat::new(-0.5, 0.8), Cfloat::new(0.3, 0.1), Cfloat::new(-0.1, 0.5)], 
                Cfloat::new(-1.0, -1.0), 
                ActivationFunction::TANH
            )
        );

        // add 2 neuron with 2 inputs
        net.add_unit(
            2, 
            Neuron::new(
                vec![Cfloat::new(0.3, -0.2), Cfloat::new(-0.3, 0.8)], 
                Cfloat::new(-1.0, -1.0), 
                ActivationFunction::SIGMOID
            )
        );
        net.add_unit(
            2, 
            Neuron::new(
                vec![Cfloat::new(-0.1, -0.2), Cfloat::new(0.3, 0.8)], 
                Cfloat::new(-1.0, -1.0), 
                ActivationFunction::SIGMOID
            )
        );


        let mut seed: u128 = 3485736485;
        let mut data  = Dataset::<f64, u8>::sample(
            //  2 * 3
            [64, 6], 
            2, 
            &mut seed
        );

        net.fit(
            &mut data,
            &ComplexCritiria::PHASE
        );
    }
}

#[cfg(test)]
mod sandbox {

    #[test]
    fn division() {
        let i: f32 = 1.0 / 0.0;
        assert_eq!(f32::INFINITY, i, "It is not");

        println!("{}", i.atan());
    }
}