pub mod math;
pub mod prelude;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_value_activation() {
        // temporary test
        // afterwards change vars to private
        use prelude::neuron::activation::Activation;
        use prelude::neuron::activation::ActivationFunction;

        let a: f32 = 0.34;
        let func1 = ActivationFunction::SIGMOID;

        let _result1 = a.activation(func1);

        let b: f64 = 0.34;
        let func2 = ActivationFunction::TANH;

        let _result2 = b.activation(func2);

        let c: f32 = 4.0;
        let func3 = ActivationFunction::RELU;

        let _result3 = c.activation(func3);

        let d: f32 = -0.23;
        let func4 = ActivationFunction::RELU;

        let _result4 = d.activation(func4);
    }

    #[test]
    fn cplex_num_ops() {
        use math::Cfloat;

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
        let _res8 = b - b_rhs;

        
        let _res9 = &a + &a_rhs;
        let _res10 = &b * &b_rhs;
        let _res11 = &b / &b_rhs;
    }

    #[test]
    fn advanced_ops() {
        use math::Cfloat;

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
