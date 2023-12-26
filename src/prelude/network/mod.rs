pub mod criteria;

use std::any::TypeId;
use std::fmt::Debug;

use std::ops::{Add, Sub, Neg, AddAssign, Mul, Div};
use std::marker::PhantomData;

use criteria::ComplexCritiria;

use crate::math::ops::base::{Number, Real, Complex};
use crate::math::random::{lcgf32, lcgf64};
use crate::math::cfloat::Cfloat;
use crate::math::cfloat::casts::ComplexCast;
use crate::math::matrix::Matrix;
use crate::math::matrix::dataset::Dataset;
use crate::math::ops::arc::Arcable;
use crate::math::ops::powi::Powerable;
use crate::math::ops::sqrt::SquareRootable;
use crate::math::ops::trig::Trignometricable;

use super::neuron::{ProcessingUnit, UnitLike};
use super::neuron::ActivationFunction;

use super::layer::{Layer, LayerLike, InputLayer};
use super::neuron::param::Param;


pub struct Network<P, U, IL, L>
    where
        P: Param,
        U: UnitLike<P>,
        IL: LayerLike<P, U>,
        L: LayerLike<P, U> {
    
    void_p: PhantomData<P>,
    void_u: PhantomData<U>,
    input: InputLayer<P, U, IL>,
    layers: Vec<Layer<P, U, L>>
}

// general implementations
impl<P, U, IL, L> Network<P, U, IL, L> 
    where
        P: Param + Copy,
        U: UnitLike<P>,
        IL: LayerLike<P, U>,
        L: LayerLike<P, U> {

    /// Returns a `DenseNetwork<W>` with just input. Reallocation will happen everytime a layer is added.
    ///  
    /// # Arguments
    /// 
    /// * `n_units` - Number of input neurons of the network.
    pub fn new() -> Network<P, U, IL, L> {

        Network {
            void_p: PhantomData{},
            void_u: PhantomData{},
            input: InputLayer::VoidP(PhantomData{}),
            layers: Vec::<Layer<P, U, L>>::new()
        }

    }

    pub fn add_input(&mut self, layer: InputLayer<P, U, IL>) {
        // this match repeats a lot. Use a macro or something else!!!
        self.input = match layer {
            InputLayer::DenseInputLayer(l) => { InputLayer::DenseInputLayer(l) },
            InputLayer::VoidP(_) => { panic!("No match for a valid layer.") },
            InputLayer::VoidU(_) => { panic!("No match for a valid layer.") }
        };

    }

    /// Updates a `DenseNetowork<W>` with an `HiddenLayer<W>`
    /// 
    /// # Arguments
    /// 
    /// * `layer` - Hidden layer of neurons to add to the network.
    pub fn add_layer(&mut self, layer: Layer<P, U, L>) {

        self.layers.push(
            match layer {
                Layer::DenseLayer(l) => { Layer::DenseLayer(l) },
                Layer::VoidP(_) => { panic!("No match for a valid layer.") },
                Layer::VoidU(_) => { panic!("No match for a valid layer.") }
            }
        );

    }

    /// Propagates a signal through a `DenseNetwork<W>` given an input.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice with the input data. Must be in agreement with the input length.
    pub fn forward(&self, input: &[P]) -> Vec<P> {
        // update this later to a result!
        let mut out = match &self.input {
            InputLayer::DenseInputLayer(l) => { l.signal(input) },
            InputLayer::VoidP(_) => { panic!("Input Layer was not defined.") },
            InputLayer::VoidU(_) => { panic!("Input Layer was not defined.") }
        };

        for layer in &self.layers {
            out = match layer {
                Layer::DenseLayer(l) => { l.signal(&out) },
                Layer::VoidP(_) => { panic!("Input Layer was not defined.") },
                Layer::VoidU(_) => { panic!("Input Layer was not defined.") }
            };
        }

        out
    }

}

// Real Implementations.
impl<P, U, IL, L> Network<P, U, IL, L> 
    where 
        P: Param + Real + Copy,
        U: UnitLike<P>,
        IL: LayerLike<P, U>,
        L: LayerLike<P, U> {

    pub fn cost(&self, data: Dataset<P, P>) -> Matrix<P> {
    
        let mut predicted_out: Matrix<P> = Matrix::new(
            [data.body.shape[0], data.degree as usize]
        );

        for row in 0..data.body.shape[0] {
            predicted_out.add_row(
                &mut self.forward(data.body.row(row))
            );
        }
    
        // you can find other patterns for calculating the cost
        predicted_out
            .sub(data.target)
            .powi(2)

    }

}

// Complex Implementations.
impl<P, U, IL, L> Network<Cfloat<P>, U, IL, L> 
    where
        P: Param + Real + Copy, 
        Cfloat<P>: Param + Complex<P> + Copy,
        U: UnitLike<Cfloat<P>>,
        IL: LayerLike<Cfloat<P>, U>,
        L: LayerLike<Cfloat<P>, U> {


    pub fn cost(&self, data: Dataset<Cfloat<P>, P>, criteria: ComplexCritiria) -> Matrix<P> {
    
        let mut predicted_out: Matrix<P> = Matrix::new(
            [data.body.shape[0], data.degree as usize]
        );

        for row in 0..data.body.shape[0] {
            predicted_out.add_row(
                &mut self.apply_criteria(data.body.row(row), &criteria)
            );
        }
    
        predicted_out
            .sub(data.target)
            .powi(2)

    }

    fn apply_criteria(
        &self, 
        data_point: &[Cfloat<P>],
        criteria: &ComplexCritiria) -> Vec<P> {

        match criteria  {
            ComplexCritiria::REAL => { 
                self
                    .forward(data_point)
                    .into_iter()
                    .map( |x| x.re() )
                    .collect::<Vec<P>>() 
            },

            ComplexCritiria::IMAGINARY => {
                self
                    .forward(data_point)
                    .into_iter()
                    .map( |x| x.im() )
                    .collect::<Vec<P>>()
            },

            ComplexCritiria::NORM => {
                self
                    .forward(data_point)
                    .into_iter()
                    .map( |x| x.norm() )
                    .collect::<Vec<P>>()
            },

            ComplexCritiria::PHASE => {
                self
                    .forward(data_point)
                    .into_iter()
                    .map( |x| x.phase() )
                    .collect::<Vec<P>>()
            }
        }

    }

}

impl<U, IL, L> Network<f32, U, IL, L> 
    where 
        U: UnitLike<f32>,
        IL: LayerLike<f32, U>,
        L: LayerLike<f32, U> {

}

/*
impl<P> Network<P> 
    where 
        P: Add<Output=P> + Sub<Output=P> + Neg<Output=P> + Mul<Output=P> + Div<Output=P>,
        P: AddAssign,
        P: Activatable + Trignometricable + Arcable + Powerable + SquareRootable,
        P: Number + PartialEq,
        P: Debug,
        P: Copy + Debug  {

    pub fn fit(
        &self, 
        data: Dataset<P, u8>
    ) {

        let dummy_array = data.target.clone();
        let mut degree = dummy_array
            .into_iter()
            .max()
            .unwrap();
        
        degree += 1;

        let mut current_row;
        let mut out;
        
        let mut processed_out: Matrix<P> = Matrix::new(
            [data.body.shape[0], degree as usize]
        );
        
        let mut expected_out: Matrix<P> = Matrix::new(
            [data.body.shape[0], degree as usize]
        );

        let mut expectancy_model;

        // going through all data to collect predictions
        for row in 0..data.body.shape[0] {

            current_row = data.body.row(&row);

            out = self.forward(current_row);

            // write a match for all the critiria
            processed_out.add_row(
                &mut out
            );

            // consider adding gaussian for building this vector
            // or some other stuff
            expectancy_model = vec![
                P::null(); 
                degree as usize
            ];
            // procedure only valid for classification
            expectancy_model[data.target[row] as usize] += P::unit();
            expected_out.add_row(&mut expectancy_model);
        }

        println!("Expected.");
        println!("{:?}", expected_out);
        println!("Obtained.");
        println!("{:?}", processed_out);

        // make a general cost function interface
        let cost_function = (processed_out - expected_out).powi(2i32);

        println!("Cost Function.");
        println!("{:?}", cost_function);

    }

}

impl<P> Network<Cfloat<P>> 
    where 
        P: Add<Output=P> + Sub<Output=P> + Neg<Output=P> + Mul<Output=P> + Div<Output=P>,
        P: AddAssign,
        P: Activatable + Trignometricable + Arcable + Powerable + SquareRootable,
        P: ComplexCast<P> + Number + PartialEq,
        P: Debug,
        Cfloat<P>: Activatable,
        P: Copy + Debug  {

    pub fn fit(
        &self, 
        data: &mut Dataset<P, u8>,
        critiria: &ComplexCritiria
    ) {
        
        let complex_data = data.to_complex();

        let dummy_array = complex_data.target.clone();
        let mut degree = dummy_array
            .into_iter()
            .max()
            .unwrap();
        
        degree += 1;

        let mut current_row;
        let mut out;

        let mut selection_out: Vec<P>;
        
        let mut processed_out: Matrix<P> = Matrix::new(
            [complex_data.body.shape[0], degree as usize]
        );
        
        let mut expected_out: Matrix<P> = Matrix::new(
            [complex_data.body.shape[0], degree as usize]
        );

        let mut expectancy_model;

        // going through all data to collect predictions
        for row in 0..complex_data.body.shape[0] {

            current_row = complex_data.body.row(&row);

            out = self.forward(current_row);

            selection_out = match critiria {
                
                ComplexCritiria::REAL => { 
                    out
                        .into_iter()
                        .map(|z| { z.re() })
                        .collect::<Vec<P>>()
                },

                ComplexCritiria::IMAGINARY => { 
                    out
                        .into_iter()
                        .map(|z| { z.im() })
                        .collect::<Vec<P>>()
                },

                ComplexCritiria::NORM => { 
                    out
                        .into_iter()
                        .map(|z| { z.norm() })
                        .collect::<Vec<P>>()
                },

                ComplexCritiria::PHASE => { 
                    out
                        .into_iter()
                        .map(|z| { z.phase() })
                        .collect::<Vec<P>>()
                }
            };

            // write a match for all the critiria
            processed_out.add_row(
                &mut selection_out
            );

            // consider adding gaussian for building this vector
            // or some other stuff
            expectancy_model = vec![
                P::null(); 
                degree as usize
            ];
            // procedure only valid for classification
            expectancy_model[complex_data.target[row] as usize] += P::unit();
            expected_out.add_row(&mut expectancy_model);
        }

        //println!("Expected.");
        //println!("{:?}", expected_out);
        //println!("Obtained.");
        println!("{:?}", processed_out);

        // make a general cost function interface
        let cost_function = (processed_out - expected_out).powi(2i32);

        println!("Cost Function.");
        println!("{:?}", cost_function);

    }

}

macro_rules! init_real_dense_network {
    ($($generator: ident, $float: ty), *) => {
        $(
            impl Network<$float> {
                pub fn init(
                    input_length: usize,
                    n_units: usize,
                    activation: ActivationFunction,
                    weight_scale: $float,
                    bias_scale: $float,
                    seed: &mut u128
                ) -> Network<$float> {
            
                    let order: usize = 0;
                    let n_weights = match calc_n_weights(input_length, n_units) {
                        
                        Ok(number) => number,
                        
                        Err((input, cells)) => {
                            panic!(
                                "Inconsistent relation between input and number of units. 
                                Input length {input} is not divisable by {cells}."
                            );
                        }
                    };
            
                    let mut network = Network::<$float>::new(n_units);
                    
                    network.add_rnd_param_layer(
                        n_units, 
                        n_weights, 
                        activation, 
                        weight_scale, 
                        bias_scale,
                        order,
                        seed
                    );
            
                    network
                }
            
                pub fn add(
                    &mut self, 
                    n_units: usize,
                    activation: ActivationFunction,
                    weight_scale: $float,
                    bias_scale: $float,
                    seed: &mut u128
                ) {
                
                    // fully connected layer
                    let n_weights = {
            
                        if self.hidden_layers.len() == 0 {
            
                            self.input_layer
                                .units
                                .len()
            
                        } else {
            
                            self.hidden_layers
                                .last()
                                .unwrap()
                                .units
                                .len()
            
                        }
            
                    };
            
                    // allocation of memory related to the hidden layer
                    self.add_layer(HiddenLayer::new(n_units));
                    let order = self.hidden_layers.len();
            
                    self.add_rnd_param_layer(
                        n_units, 
                        n_weights, 
                        activation, 
                        weight_scale,
                        bias_scale,
                        order,
                        seed
                    )
            
                }
            
                fn add_rnd_param_layer(
                    &mut self,
                    n_units: usize,
                    n_weights: usize,
                    activation: ActivationFunction,
                    weight_scale: $float,
                    bias_scale: $float,
                    order: usize,
                    seed: &mut u128
                ) {
            
                    let mut current_weights: Vec<$float> = Vec::with_capacity(n_weights);
            
                    for _ in 0..n_units {
                        for _ in 0..n_weights {
                            current_weights.push(
                                $generator(seed) * weight_scale
                            );
                        }
            
                        self.add_unit(
                            order,
                            Neuron::new(
                                current_weights.clone(), 
                                $generator(seed) * bias_scale,
                                activation.clone()
                            )
                        );
            
                        current_weights.clear();
                    }
            
                }
            
            }
        )*
    };
}

init_real_dense_network!(lcgf32, f32, lcgf64, f64);

macro_rules! init_complex_dense_network {
    ($($generator: ident, $float: ty), *) => {
        $(
            impl Network<Cfloat<$float>> {
                pub fn init(
                    input_length: usize,
                    n_units: usize,
                    activation: ActivationFunction,
                    re_weight_scale: $float,
                    im_weight_scale: $float,
                    re_bias_scale: $float,
                    im_bias_scale: $float,
                    seed: &mut u128
                ) -> Network<Cfloat<$float>> {
            
                    let order: usize = 0;
                    let n_weights = match calc_n_weights(input_length, n_units) {
                        
                        Ok(number) => number,
                        
                        Err((input, cells)) => {
                            panic!(
                                "Inconsistent relation between input and number of units. 
                                Input length {input} is not divisable by {cells}."
                            );
                        }
                    };
            
                    let mut network = Network::<Cfloat<$float>>::new(n_units);
                    
                    network.add_rnd_param_layer(
                        n_units, 
                        n_weights, 
                        activation, 
                        re_weight_scale, 
                        im_weight_scale, 
                        re_bias_scale, 
                        im_bias_scale,
                        order,
                        seed
                    );
            
                    network
                }
            
                pub fn add(
                    &mut self, 
                    n_units: usize,
                    activation: ActivationFunction,
                    re_weight_scale: $float,
                    im_weight_scale: $float,
                    re_bias_scale: $float,
                    im_bias_scale: $float,
                    seed: &mut u128
                ) {
                
                    // fully connected layer
                    let n_weights = {
            
                        if self.hidden_layers.len() == 0 {
            
                            self.input_layer
                                .units
                                .len()
            
                        } else {
            
                            self.hidden_layers
                                .last()
                                .unwrap()
                                .units
                                .len()
            
                        }
            
                    };
            
                    // allocation of memory related to the hidden layer
                    self.add_layer(HiddenLayer::new(n_units));
                    let order = self.hidden_layers.len();
            
                    self.add_rnd_param_layer(
                        n_units, 
                        n_weights, 
                        activation, 
                        re_weight_scale, 
                        im_weight_scale, 
                        re_bias_scale, 
                        im_bias_scale,
                        order,
                        seed
                    )
            
                }
            
                fn add_rnd_param_layer(
                    &mut self,
                    n_units: usize,
                    n_weights: usize,
                    activation: ActivationFunction,
                    re_weight_scale: $float,
                    im_weight_scale: $float,
                    re_bias_scale: $float,
                    im_bias_scale: $float,
                    order: usize,
                    seed: &mut u128
                ) {
            
                    let mut current_weights: Vec<Cfloat<$float>> = Vec::with_capacity(n_weights);
            
                    for _ in 0..n_units {
                        for _ in 0..n_weights {
                            current_weights.push(
                                Cfloat::<$float>::new(
                                    $generator(seed) * re_weight_scale, 
                                    $generator(seed) * im_weight_scale
                                )
                            );
                        }
            
                        self.add_unit(
                            order,
                            Neuron::new(
                                current_weights.clone(), 
                                Cfloat::<$float>::new(
                                    $generator(seed) * re_bias_scale, 
                                    $generator(seed) * im_bias_scale
                                ), 
                                activation.clone()
                            )
                        );
            
                        current_weights.clear();
                    }
            
                }
            
            }
        )*
    };
}

init_complex_dense_network!(lcgf32, f32, lcgf64, f64);

/// Small utility that calculates the number of weights needed for
/// a dense networks's input.
fn calc_n_weights(
    input_length: usize, 
    n_units: usize
) -> Result<usize, (usize, usize)>{

    if input_length % n_units == 0 {
        
        Ok(input_length / n_units)

    } else {

        Err((input_length, n_units))

    }

}

*/