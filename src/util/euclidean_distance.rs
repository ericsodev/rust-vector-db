use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub fn calculate_euclidean_distance(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(a, b)| (*b - *a).powf(2.0))
        .sum::<f32>()
        .powf(0.5)
}
