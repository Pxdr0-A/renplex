//! Module for public error handling related to the network.

#[derive(Debug)]
pub enum LayerForwardError {
  InvalidInput
}

#[derive(Debug)]
pub enum LayerInitError {
  InvalidInputShape,
  AlreadyInitialized
}

#[derive(Debug)]
pub enum ForwardError {
  MissingLayers
}

pub enum BackwardError {
  MissingLayers
}

#[derive(Debug)]
pub enum LayerAdditionError {
  MissingInput,
  ExistentInput,
  EarlyInitialization,
  IncompatibleIO
}

#[derive(Debug)]
pub enum LossCalcError {
  IncompatibleDataset,
  InconsistentIO
}

#[derive(Debug)]
pub enum PredicionError {
  CriticalIndexOverflow
}

#[derive(Debug)]
pub enum GradientError {
  InconsistentShape,
  InvalidBiasShape,
  InvalidWeightShape,
  Unimplemented
}