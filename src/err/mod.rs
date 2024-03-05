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

#[derive(Debug)]
pub enum LayerAdditionError {
  MissingInput,
  ExistentInput,
  EarlyInitialization,
  IncompatibleIO
}

pub enum CostError {
  IncompatibleDataset,
  InconsistentIO
}

#[derive(Debug)]
pub enum PredicionError {
  CriticalIndexOverflow
}