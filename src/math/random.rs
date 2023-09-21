/// Returns a random number between 0 and 1 with 2^31 different equal spaced values.
/// 
/// # Arguments
/// 
/// * `seed` - a mutable `u128` with a certain initial value (seed), that will be changing throughout calls.
///
/// # Example
/// ```
/// use renplex::math::random::lcg;
/// 
/// let mut seed = 12345u128;
/// let mut val: f64;
/// for _ in 0..10 {
///     val = lcg(&mut seed);
///     println!("Value Update: {}", val);
///     println!("Seed Update: {}", seed);
/// }
/// ```
pub fn lcg(seed: &mut u128) -> f64 {
    // IBM C/C++ convention params
    let a: u128 = 1103515245;
    let b: u128 = 12345;
    let m: u128 = 2u128.pow(31);

    *seed = (a * *seed + b) % (m - 1);
    let rand = (*seed as f64) / (m as f64);

    rand
}

