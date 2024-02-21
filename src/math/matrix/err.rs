#[derive(Debug)]
pub enum AccessError {
  OutOfBounds
}

#[derive(Debug)]
pub enum UpdateError {
  InconsistentLength,
  Overflow
}

#[derive(Debug)]
pub enum DeletionError {
  OutOfBounds,
  Empty
}

#[derive(Debug)]
pub enum OperationError {
  OutOfBounds,
  InconsistentShape,
  InvalidRHS
}