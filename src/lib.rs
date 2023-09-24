pub mod math;
pub mod prelude;

#[cfg(test)]
mod calcs {
    use super::*;

    #[test]
    fn generic_value_activation() {
        // temporary test
        // afterwards change vars to private
        use prelude::neuron::activation::Activation;
        use prelude::neuron::activation::ActivationFunction;
        use math::complex::Cfloat;

        let a: f32 = 0.34;
        let func1 = ActivationFunction::SIGMOID;

        let _result1 = a.activation(&func1);

        let b: f64 = 0.34;
        let func2 = ActivationFunction::TANH;

        let _result2 = b.activation(&func2);

        let c: f32 = 4.0;
        let func3 = ActivationFunction::RELU;

        let _result3 = c.activation(&func3);

        let d: f32 = -0.23;
        let func4 = ActivationFunction::RELU;
        let _result4 = d.activation(&func4);

        let e = Cfloat::new(2.3, 7.12);
        let func5 = ActivationFunction::TANH;
        let _result5 = e.activation(&func5);
        
    }

    #[test]
    fn cplex_num_ops() {
        use math::complex::Cfloat;

        let a = Cfloat::new(0.34, 0.75);
        let a_rhs = Cfloat::new(0.45, 0.05);
        let b = Cfloat::new(0.81f32, 2.75f32);
        let b_rhs = Cfloat::new(0.12f32, -1.0f32);

        let _res1 = a * a_rhs;
        let _res2 = b * b_rhs;

        let _res3 = a / a_rhs;
        let _res4 = b / b_rhs;

        let _res5 = a + a_rhs;
        let _res6 = b + b_rhs;

        let _res7 = a - a_rhs;
        let mut _res8 = b - b_rhs;
        _res8 += _res7;
        
        let _res9 = &a + &a_rhs;
        let _res10 = &b * &b_rhs;
        let _res11 = &b / &b_rhs;

    }

    #[test]
    fn advanced_ops() {
        use math::complex::Cfloat;

        let c1 = Cfloat::new(1.23f32, 3.33f32);
        let _res1 = c1.phase();
        let _res2 = c1.norm();
        let _res3 = c1.exp();
        let _res4 = c1.tanh();
        let _res5 = c1.is_sign_positive();

        let c2 = Cfloat::new(5.93f64, 2.65f64);
        let _res6 = c2.phase();
        let _res7 = c2.norm();
        let _res8 = c2.exp();
        let _res9 = c2.tanh();
        let _res10 = c2.is_sign_positive();
    }
}

#[cfg(test)]
mod neuron {
    use super::*;

    #[test]
    fn signal() {
        use math::complex::Cfloat;
        use prelude::neuron::Neuron;
        use prelude::neuron::activation::ActivationFunction;
        
        let _n1 = Neuron::new(
            vec![2.34, 5.4, 1.2], 
            1.0, 
            ActivationFunction::SIGMOID
        ).signal(&vec![4.0, 3.0, 2.0]);

        let _n2 = Neuron::new(
            vec![
                Cfloat::new(5.0f32, 1.0f32),
                Cfloat::new(1.0f32, 1.0f32)
            ], 
            Cfloat::new(-1.0f32, -1.0f32), 
            ActivationFunction::RELU
        );
        let _output = _n2.signal(
            &vec![
                Cfloat::new(1.0, 1.0),
                Cfloat::new(1.0, 1.0)
            ]
        );
        
        let _n3 = Neuron::new(
            vec![
                Cfloat::new(3.0f32, 1.5f32),
                Cfloat::new(1.0f32, 2.0f32)
            ], 
            Cfloat::new(-0.5f32, -0.5f32), 
            ActivationFunction::RELU
        );

        let _n4 = Neuron::new(
            vec![
                Cfloat::new(1.0f32, 1.0f32),
                Cfloat::new(3.0f32, 2.0f32)
            ], 
            Cfloat::new(-1.5f32, -2.5f32), 
            ActivationFunction::SIGMOID
        );
    }
}

#[cfg(test)]
mod layer {
    use super::*;

    #[test]
    fn signal() {
        use prelude::neuron::activation::ActivationFunction;
        use prelude::neuron::Neuron;
        use prelude::layer::Layer;
        use prelude::layer::{InputLayer, HiddenLayer};
        
        let mut l_h: HiddenLayer<f64> = Layer::new(3);
        l_h.add(
            Neuron::new(
                vec![1.2, 1.4], 
                0.5, 
                ActivationFunction::SIGMOID
            )
        );

        l_h.add(
            Neuron::new(
                vec![3.2, 1.2], 
                1.0, 
                ActivationFunction::SIGMOID
            )
        );

        l_h.add(
            Neuron::new(
                vec![0.2, 0.5], 
                1.5, 
                ActivationFunction::SIGMOID
            )
        );

        let _out_h = l_h.signal(&vec![1.0, 0.5]);

        let mut l_i: InputLayer<f64> = Layer::new(3);
        l_i.add(
            Neuron::new(
                vec![1.2, 1.4], 
                0.5, 
                ActivationFunction::SIGMOID
            )
        );

        l_i.add(
            Neuron::new(
                vec![3.2, 1.2], 
                1.0, 
                ActivationFunction::SIGMOID
            )
        );

        l_i.add(
            Neuron::new(
                vec![0.2, 0.5], 
                1.5, 
                ActivationFunction::SIGMOID
            )
        );

        let _out_i = l_i.signal(&vec![0.1, 0.5, 1.0, 1.5, 2.5, 0.5]);

        println!("{:?}", _out_i);
    }
}

#[cfg(test)]
mod network {
    use super::*;
    
    #[test]
    fn foward() {
        use prelude::neuron::activation::ActivationFunction;
        use prelude::neuron::Neuron;
        use prelude::layer::{Layer, HiddenLayer};
        use prelude::network::Network;

        let mut network: Network<f64> = Network::new(2);

        let in1 = Neuron::new(
            vec![0.2, 0.5, 0.5], 
            -0.2, 
            ActivationFunction::SIGMOID
        );
        let in2 = Neuron::new(
            vec![0.2, 0.1, 0.5], 
            -0.3, 
            ActivationFunction::RELU
        );

        network.add_unit(0, in1);
        network.add_unit(0, in2);

        network.add(HiddenLayer::new(4));
        network.add(HiddenLayer::new(2));

        let hd11 = Neuron::new(
            vec![1.3, 0.5], 
            -1.0, 
            ActivationFunction::SIGMOID
        );
        let hd12 = Neuron::new(
            vec![0.2, 2.5], 
            -0.1, 
            ActivationFunction::SIGMOID
        );
        let hd13 = Neuron::new(
            vec![1.1, 0.5], 
            -1.0, 
            ActivationFunction::SIGMOID
        );
        let hd14 = Neuron::new(
            vec![0.6, 0.7], 
            -0.2, 
            ActivationFunction::SIGMOID
        );
        
        network.add_unit(1, hd11);
        network.add_unit(1, hd12);
        network.add_unit(1, hd13);
        network.add_unit(1, hd14);

        let hd21 = Neuron::new(
            vec![0.1, 0.5, 0.2, 0.1], 
            -0.1, 
            ActivationFunction::RELU
        );
        let hd22 = Neuron::new(
            vec![0.1, 0.5, 0.2, 0.1], 
            -0.2, 
            ActivationFunction::RELU
        );

        network.add_unit(2, hd21);
        network.add_unit(2, hd22);

        let _out = network.foward(&vec![0.2, 0.3, 1.2, 4.3, 1.0, 0.9]);

        println!("{:?}", _out);
    }
}

#[cfg(test)]
mod sandbox {

    #[test]
    fn slices() {
        let slice1 = &vec![0,1,2,3,4,5,6,7,8,9];

        println!("{:?}", &slice1[9..=9]);
    }
}