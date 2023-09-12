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

    use math::complex::Cfloat;
    use prelude::neuron::Neuron;
    use prelude::layer::Layer;
    use prelude::neuron::activation::ActivationFunction;

    #[test]
    fn signal() {
        let _n1 = Neuron::new(
            1, 
            vec![2.34, 5.4, 1.2], 
            1.0, 
            ActivationFunction::SIGMOID
        ).signal(&vec![4.0, 3.0, 2.0]);

        let _n2 = Neuron::new(
            2, 
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
            2, 
            vec![
                Cfloat::new(3.0f32, 1.5f32),
                Cfloat::new(1.0f32, 2.0f32)
            ], 
            Cfloat::new(-0.5f32, -0.5f32), 
            ActivationFunction::RELU
        );

        let _n4 = Neuron::new(
            2, 
            vec![
                Cfloat::new(1.0f32, 1.0f32),
                Cfloat::new(3.0f32, 2.0f32)
            ], 
            Cfloat::new(-1.5f32, -2.5f32), 
            ActivationFunction::SIGMOID
        );

        let l = Layer::new(
            1, 
            vec![_n2, _n3, _n4]
        );
        let _out_layer = l.signal(&vec![
            Cfloat::new(1.0, 1.0),
            Cfloat::new(1.0, 1.0)
        ]);
    }
}