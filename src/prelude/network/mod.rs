pub mod criteria;


use std::fmt::Debug;
use std::ops::{Add, Sub, Neg, AddAssign, Mul, Div};

use criteria::ComplexCritiria;

use crate::math::complex::Cfloat;
use crate::math::complex::casts::ComplexCast;
use crate::math::matrix::Matrix;
use crate::math::matrix::dataset::Dataset;
use crate::math::ops::arc::Arcable;
use crate::math::ops::powi::Powerable;
use crate::math::ops::sqrt::SquareRootable;
use crate::math::ops::trig::Trignometricable;

use super::neuron::Neuron;
use super::neuron::activation::Activatable;

use super::layer::{InputLayer, HiddenLayer, Layer};


pub struct Network<W> {
    pub input_layer: InputLayer<W>,
    pub hidden_layers: Vec<HiddenLayer<W>>
}

impl<W> Network<W> {
    /// Returns a `Network<W>` with just input. Reallocation will happen everytime a layer is added.
    /// 
    /// # Arguments
    /// 
    /// * `n_units` - Number of input neurons of the network.
    pub fn new(n_units: usize) -> Network<W> {
        Network {
            input_layer: InputLayer::new(n_units),
            hidden_layers: Vec::<HiddenLayer<W>>::new()
        }
    }

    /// Updates a `Netowork<W>` with an `HiddenLayer<W>`
    /// 
    /// # Arguments
    /// 
    /// * `layer` - Hidden layer of neurons to add to the network.
    pub fn add(&mut self, layer: HiddenLayer<W>) {
        self.hidden_layers.push(layer);
    }

    /// Updates some layer of a `Netowork<W>` with a `Neuron<W>`
    /// 
    /// # Arguments
    /// 
    /// * `order` - Index of the layer to add the neuron. 
    /// * `neuron` - Neuron to add to the respective layer.
    pub fn add_unit(&mut self, order: usize, neuron: Neuron<W>) {
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

    /// Propagates a signal through a `Network<W>` given an input.
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

impl<P> Network<Cfloat<P>> {
    pub fn fit(&self, 
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

        let cost_function = (processed_out - expected_out).powi(2i32);

        println!("Cost Function.");
        println!("{:?}", cost_function);
    }
}