pub mod math;
pub mod lite;
pub mod dynamic;


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
        network.add_layer(layer);

        println!("{:?}", network.foward(&[2.0; 6]));
    }

    #[test]
    fn random_net_test() {
        use lite::real::layer::dense::DenseInputLayer;
        use lite::real::layer::dense::DenseLayer;
        use lite::real::network::FeedFoward;
        use lite::real::ActivationFunction::{SIGMOID, TANH};

        let ref mut seed = 943456861;
        let scale = 10.0;

        let input_layer = DenseInputLayer::init(
            6, 
            12, 
            TANH, 
            scale, 
            seed
        ).wrap();

        let layer1 = DenseLayer::init(
            16, 
            6, 
            SIGMOID, 
            scale, 
            seed
        ).wrap();

        let layer2 = DenseLayer::init(
            16, 
            16, 
            SIGMOID, 
            scale, 
            seed
        ).wrap();

        let layer3 = DenseLayer::init(
            2, 
            16, 
            SIGMOID, 
            scale, 
            seed
        ).wrap();

        let mut network = FeedFoward::new(input_layer);
        network.add_layer(layer1);
        network.add_layer(layer2);
        network.add_layer(layer3);

        let out1 = network.foward(&[2.3, 0.1, 1.0, 1.2, 0.4, 2.9, 1.8, 0.9, 0.2, 0.1, 3.1, 1.3]);
        let out2 = network.foward(&vec![1.0; 12]);

        println!("{:?}", out1);
        println!("{:?}", out2);
    }

    #[test]
    fn cost() {
        use crate::math::matrix::dataset::Dataset;
        use lite::real::layer::dense::DenseInputLayer;
        use lite::real::layer::dense::DenseLayer;
        use lite::real::network::FeedFoward;
        use lite::real::ActivationFunction::{SIGMOID, TANH};

        let ref mut seed = 92347865;

        let data: Dataset<f32, f32> = Dataset::<f32, f32>::sample(
            [64, 6], 
            4, 
            seed
        );

        let scale = 10.0;

        let input_layer = DenseInputLayer::init(
            3, 
            6, 
            TANH, 
            scale, 
            seed
        ).wrap();

        let layer1 = DenseLayer::init(
            16, 
            3, 
            SIGMOID, 
            scale, 
            seed
        ).wrap();

        let layer2 = DenseLayer::init(
            16, 
            16, 
            SIGMOID, 
            scale, 
            seed
        ).wrap();

        let layer3 = DenseLayer::init(
            4, 
            16, 
            SIGMOID, 
            scale, 
            seed
        ).wrap();

        let mut network = FeedFoward::new(input_layer);
        network.add_layer(layer1);
        network.add_layer(layer2);
        network.add_layer(layer3);

        let cost_func = network.cost(data);
        println!("{:?}", cost_func)
    }
}