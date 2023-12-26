pub mod math;
pub mod lite;
pub mod prelude;


#[cfg(test)]
mod lite_test {

    use super::*;

    #[test]
    fn dense_neuron_test() {
        use lite::real::unit::dense::DenseNeuron;
        use lite::real::ActivationFunction;

        let n = DenseNeuron::new(
            vec![1.0, 2.0], 
            2.0, 
            ActivationFunction::SIGMOID
        );

        println!("{:?}", n);

        println!("{:?}", n.signal(&[2.0, 2.0]));
    }

    #[test]
    fn dense_layer_test() {
        use lite::real::unit::dense::DenseNeuron;
        use lite::real::layer::dense::DenseInputLayer;
        use lite::real::layer::dense::DenseLayer;
        use lite::real::network::FeedFoward;
        use lite::real::ActivationFunction;

        let n1 = DenseNeuron::new(
            vec![0.1, 0.2], 
            2.0, 
            ActivationFunction::SIGMOID
        );
        let n2 = DenseNeuron::new(
            vec![0.3, 0.2], 
            2.0, 
            ActivationFunction::SIGMOID
        );
        let n3 = DenseNeuron::new(
            vec![1.0, 2.0], 
            2.0, 
            ActivationFunction::SIGMOID
        );

        let n4 = DenseNeuron::new(
            vec![1.0, 2.0, 3.0], 
            2.0, 
            ActivationFunction::SIGMOID
        );
        let n5 = DenseNeuron::new(
            vec![1.0, 4.0, 3.0], 
            2.0, 
            ActivationFunction::SIGMOID
        );

        let mut l1 = DenseInputLayer::new(3, 6);
        l1.add(n1);
        l1.add(n2);
        l1.add(n3);
        
        let mut l2 = DenseLayer::new(2);
        l2.add(n4);
        l2.add(n5);

        println!("{:?}", l1.signal(&[2.0; 6]));
        println!("{:?}", l2.signal(&[1.0, 2.0, 1.0]));

        let input_layer = l1.wrap();
        let layer = l2.wrap();

        let mut network = FeedFoward::new(input_layer);
        network.add(layer);

        println!("{:?}", network.foward(&[2.0; 6]));
    }
}