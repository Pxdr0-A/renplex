# CVNN library in Rust!

This was a project done in the context of a master thesis. This library is not scalable, has very unefficient linear algebra and computing techniques. For that reason this repository will be archived.

## Overview

A library built with Rust capable of modeling complex-valued neural networks.

It is still in early stages of development but can already be used to build and train architectures based on Multi-Layer Perceptron and Convolutional Neural Networks.

## Usage

Add the library to your `Cargo.toml` on your rust project:
```toml
[dependencies]
renplex = "0.1.0"
```

### Initiate a Layer

Initiate a trainable layer by indicating precision of the calculations, number of input features and output features, initialization method and activation function.

```rust
use renplex::math::Complex;
use renplex::math::cfloat::Cf32;
use renplex::input::IOShape;
use renplex::init::InitMethod;
use renplex::act::ComplexActFunc;
use renplex::cvnn::layer::dense::DenseCLayer;

let ref mut seed: &mut u128 = 63478262957;

// define complex number with 64bits (precision)
// 32 bits for each real and imaginary part
type Precision = Cf32;

// number of scalar input features
let ni = 64;
// number of scalar output features (number of units)
let no = 16;

// input features are scalars (vetor of values)
// in the case of a 2D conv is input features are matrices (vector of matrices)
let input_shape = IOShape::Scalar(ni);
// initialization method
let init_method = InitMethod::XavierGlorotU(ni + no);
// complex activation function
let act_func = ComplexActFunc::RITSigmoid;


let dense_layer: DenseCLayer<Precision> = DenseCLayer::init(
  input_shape, 
  no,
  act_func,
  init_method,
  seed
).unwrap();
```

### Create a Network and Add Layers

Add layers to a feed forward network struct by defining an input layer and subsquent hidden layers.

```rust
use renplex::math::Complex;
use renplex::math::cfloat::Cf32;
use renplex::opt::ComplexLossFunc;	
use renplex::cvnn::layer::CLayer;
use renplex::cvnn::network::CNetwork;

let mut network: CNetwork<Cf32> = CNetwork::new();

// layers need to be wrapped for a common CLayer<T> interface
network.add_input(dense_input_layer.wrap()).unwrap();
network.add(dense_layer.wrap()).unwrap();
```

### Construct Your Batch of Data

Renplex provides a simple dataset interface for building a batch of data with independent and dependent variable.

```rust
use renplex::math::Complex;
use renplex::math::cfloat::Cf32;
use renplex::dataset::Dataset;

// independent variable type
type XFeatures = Cf32;
// dependent variable type
type YFeatures = Cf32; 

// initialize a batch of data
let mut data_batch: Dataset<XFeatures, YFeatures> = Dataset::new();
// extract a unique batch of data points
// can be done in any logic (default order, randomized, ...)
for _ in 0..batch_size {
  // collect data points from a file
  let x = ...;
  let y = ...;
  let data_point = (x, y);
  // add point to the dataset
  data_batch.add_point(data_point);
}
```

### Train with a Gradient-Based Method

Calculate performance metrics train a CVNN with the fully complex back-propagation algorithm.

```rust
use renplex::math::Complex;
use renplex::math::cfloat::Cf32;
use renplex::opt::ComplexLossFunction;

// define loss function
let loss_func = ComplexLossFuntion::MeanSquare;
// history of loss function values
let loss_vals = Vec::new();
// define a learning rate
let learning_rate = Cf32::new(1.0, 0.0);

// calculate the initial loss for the batch of data
let loss = network.loss(
  data_batch,
  &loss_func
).unwrap()
// add loss value to history
// (for optimization algorithms for instance)
loss_vals.push(loss);

// train 1 batch of data
network.gradient_opt(
  data_batch,
  &loss_func,
  learning_rate
).unwrap();

// this pipeline can be repeated to perform an epoch
// and repeated again for as many epochs choosen
```

### Perform Predictions and Intercept Signals

Forward signals in a CVNN and inspect on intermediate features.

```rust
use renplex::input::IOType;

let input_point = IOType::Scalar(vec![0.22, 0.17, 0.13]);

// output of the networks
let prediction = network
  .foward(input_point)
  .unwrap();

// output features of the second layer of the network
let features = network
  .intercept(input_point, 2)
  .unwrap();

```
## Examples in development

In the repository, there is an examples folder with a ``classification.rs`` and ``regression.rs`` files that run each respective pipeline however, ``classification.rs`` requires the MNIST dataset on the root of the project. To run an example code, use the following command after the project is cloned (at the root of the project):
```sh
cargo run --example <example>
```

One can also download each file individually from GitHub and run it inside a project with Renplex as a dependency.
