/// Returns a random number between 0 and 1 with 2^31 different equal spaced values.
/// 
/// # Arguments
/// 
/// * `seed` - a mutable `u128` with a certain initial value (seed), that will be changing throughout calls.
///
/// # Example
/// ```
/// use renplex::math::random::lcgf32;
/// 
/// let mut seed = 12345u128;
/// let mut val: f32;
/// for _ in 0..10 {
///     val = lcgf32(&mut seed);
///     println!("Value Update: {}", val);
///     println!("Seed Update: {}", seed);
/// }
/// ```
pub fn lcgf32(seed: &mut u128) -> f32 {
    // IBM C/C++ convention params
    let a: u128 = 1103515245;
    let b: u128 = 12345;
    let m: u128 = 2u128.pow(31);

    *seed = (a * *seed + b) % (m - 1);
    
    (*seed as f32) / (m as f32)

}

pub fn lcgf64(seed: &mut u128) -> f64 {
    // IBM C/C++ convention params
    let a: u128 = 1103515245;
    let b: u128 = 12345;
    let m: u128 = 2u128.pow(31);

    *seed = (a * *seed + b) % (m - 1);
    
    (*seed as f64) / (m as f64)

}

pub fn lcgi(seed: &mut u128, m: u128) -> usize {
    // IBM C/C++ convention params
    let a: u128 = 1103515245;
    let b: u128 = 12345;

    *seed = (a * *seed + b) % (m - 1);
    
    (*seed as usize) / (m as usize)

}