pub mod criteria;

use std::fmt::Debug;
use std::ops::{Add, Sub, Neg, AddAssign, Mul, Div};

use criteria::ComplexCritiria;

use crate::math::random::{lcgf32, lcgf64};
use crate::math::complex::Cfloat;
use crate::math::complex::casts::ComplexCast;
use crate::math::matrix::Matrix;
use crate::math::matrix::dataset::Dataset;
use crate::math::ops::arc::Arcable;
use crate::math::ops::powi::Powerable;
use crate::math::ops::sqrt::SquareRootable;
use crate::math::ops::trig::Trignometricable;

use super::neuron::Neuron;
use super::neuron::activation::{Activatable, ActivationFunction};

use super::layer::{InputLayer, HiddenLayer, Layer};

#[derive(Debug)]
pub struct DenseNetwork<W> {
    pub input_layer: InputLayer<W>,
    pub hidden_layers: Vec<HiddenLayer<W>>
}

impl<W> DenseNetwork<W> {
    /// Returns a `DenseNetwork<W>` with just input. Reallocation will happen everytime a layer is added.
    /// 
    /// # Arguments
    /// 
    /// * `n_units` - Number of input neurons of the network.
    pub fn new(n_units: usize) -> DenseNetwork<W> {
        DenseNetwork {
            input_layer: InputLayer::new(n_units),
            hidden_layers: Vec::<HiddenLayer<W>>::new()
        }
    }

    /// Updates a `DenseNetowork<W>` with an `HiddenLayer<W>`
    /// 
    /// # Arguments
    /// 
    /// * `layer` - Hidden layer of neurons to add to the network.
    pub fn add_layer(&mut self, layer: HiddenLayer<W>) {
        self.hidden_layers.push(layer);
    }

    /// Updates some layer of a `DenseNetowork<W>` with a `Neuron<W>`
    /// 
    /// # Arguments
    /// 
    /// * `order` - Index of the layer to add the neuron. 
    /// * `neuron` - Neuron to add to the respective layer.
    pub fn add_unit(&mut self, order: usize, neuron: Neuron<W>) {
        
        // consider pattern matching
        if order == 0 {
            self.input_layer.add(neuron);
        } else {
            assert!(
                order <= self.hidden_layers.len(), 
                "Attempted to add a unit to an unexistent layer."
            );
            self.hidden_layers[order-1].add(neuron);
        }
    }

    /// Propagates a signal through a `DenseNetwork<W>` given an input.
    /// 
    /// # Arguments
    /// 
    /// * `input` - Slice with the input data. Must be in agreement with the input length.
    pub fn forward(&self, input: &[W]) -> Vec<W>
        where 
            W: AddAssign + Mul<Output = W> + Activatable, 
            W: Copy {
        
        let mut out = self.input_layer.signal(input);
        for layer in &self.hidden_layers {
            out = layer.signal(&out);
        }

        out
    }
}

impl<P> DenseNetwork<Cfloat<P>> {

    pub fn fit(
        &self, 
        data: &mut Dataset<P, u8>,
        critiria: &ComplexCritiria
    ) 
        where 
            P: Add<Output=P> + Sub<Output=P> + Neg<Output=P> + Mul<Output=P> + Div<Output=P>,
            P: AddAssign,
            P: Activatable + Trignometricable + Arcable + Powerable + SquareRootable + ComplexCast<P>,
            Cfloat<P>: Activatable,
            P: Copy + Debug {
        
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

        let val_ref = complex_data.body.elm(&0, &0);
        let null_ref = val_ref.x - val_ref.x;
        let unit_ref = val_ref.x / val_ref.x;
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
            expectancy_model = vec![
                null_ref; 
                degree as usize
            ];
            // procedure only valid for classification
            expectancy_model[complex_data.target[row] as usize] += unit_ref;
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

impl DenseNetwork<Cfloat<f32>> {
    pub fn init(
        input_length: usize,
        n_units: usize,
        activation: ActivationFunction,
        re_weight_scale: f32,
        im_weight_scale: f32,
        re_bias_scale: f32,
        im_bias_scale: f32,
        seed: &mut u128
    ) -> DenseNetwork<Cfloat<f32>> {

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

        let mut network = DenseNetwork::<Cfloat<f32>>::new(n_units);
        
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
        re_weight_scale: f32,
        im_weight_scale: f32,
        re_bias_scale: f32,
        im_bias_scale: f32,
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
        re_weight_scale: f32,
        im_weight_scale: f32,
        re_bias_scale: f32,
        im_bias_scale: f32,
        order: usize,
        seed: &mut u128
    ) {

        let mut current_weights: Vec<Cfloat<f32>> = Vec::with_capacity(n_weights);

        for _ in 0..n_units {
            for _ in 0..n_weights {
                current_weights.push(
                    Cfloat::<f32>::new(
                        lcgf32(seed) * re_weight_scale, 
                        lcgf32(seed) * im_weight_scale
                    )
                );
            }

            self.add_unit(
                order,
                Neuron::new(
                    current_weights.clone(), 
                    Cfloat::<f32>::new(
                        lcgf32(seed) * re_bias_scale, 
                        lcgf32(seed) * im_bias_scale
                    ), 
                    activation.clone()
                )
            );

            current_weights.clear();
        }

    }

}

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