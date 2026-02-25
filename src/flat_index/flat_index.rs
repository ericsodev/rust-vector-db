use crate::{index::Searchable, vector::vector::VectorNode};

pub struct FlatIndex {
    dimension: usize,
    vectors: Vec<VectorNode>,
    strategy: FlatIndexStrategy,
}

pub enum FlatIndexStrategy {
    COSINE,
    EUCLIDEAN,
}

impl FlatIndex {
    pub fn new(strategy: FlatIndexStrategy, dimension: usize) -> FlatIndex {
        return FlatIndex {
            dimension,
            strategy,
            vectors: vec![],
        };
    }
}

impl crate::index::Index for FlatIndex {
    fn train(&mut self) -> Result<(), String> {
        // No training for Flat index
        return Ok(());
    }

    fn add(&mut self, vector: VectorNode) -> Result<(), String> {
        if vector.get_vector().len() != self.dimension {
            return Err("Incorrect dimension".to_string());
        }

        self.vectors.push(vector);
        return Ok(());
    }

    fn add_batch(&mut self, vectors: Vec<VectorNode>) -> Result<(), String> {
        for vec in vectors.iter() {
            if vec.get_vector().len() != self.dimension {
                return Err("Incorrect dimension".to_string());
            }
        }

        for vec in vectors {
            self.vectors.push(vec);
        }

        return Ok(());
    }

    fn remove(&mut self, id: u64) -> Result<(), String> {
        for (i, vec) in self.vectors.iter().enumerate() {
            if vec.get_id() == id {
                self.vectors.remove(i);
                return Ok(());
            }
        }

        return Err("No vector with ID".to_string());
    }
}

struct VectorDistance<'a> {
    vector: &'a VectorNode,
    distance: f32,
}

impl Searchable for FlatIndex {
    fn search(&self, vector: &Vec<f32>, k: u32) -> Result<Vec<&VectorNode>, String> {
        if vector.len() != self.dimension {
            return Err("Incorrect dimension".to_string());
        }

        let distance: Vec<VectorDistance> = match self.strategy {
            FlatIndexStrategy::COSINE => sort_vectors_cosine(vector, &self.vectors),
            FlatIndexStrategy::EUCLIDEAN => sort_vectors_euclidean(vector, &self.vectors),
        };

        Ok(distance.iter().take(k as usize).map(|v| v.vector).collect())
    }
}

fn sort_vectors_euclidean<'a>(
    query: &Vec<f32>,
    vectors: &'a Vec<VectorNode>,
) -> Vec<VectorDistance<'a>> {
    let mut with_distance: Vec<VectorDistance> = vectors
        .iter()
        .map(|a| VectorDistance {
            vector: a,
            distance: euclidean_distance(a.get_vector(), query),
        })
        .collect();
    with_distance.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

    with_distance
}

struct UnitVector<'a> {
    vector: &'a VectorNode,
    unit: Vec<f32>,
}

fn calculate_unit_vector(vector: &Vec<f32>) -> Vec<f32> {
    let dist: f32 = vector.iter().map(|v| v.powf(2.0)).sum();

    vector.iter().map(|v| v / dist).collect()
}

fn sort_vectors_cosine<'a>(
    query: &Vec<f32>,
    vectors: &'a Vec<VectorNode>,
) -> Vec<VectorDistance<'a>> {
    let unit_vectors: Vec<UnitVector> = vectors
        .iter()
        .map(|v| UnitVector {
            vector: v,
            unit: calculate_unit_vector(v.get_vector()),
        })
        .collect();

    let query_unit = calculate_unit_vector(query);

    let with_similarity: Vec<VectorDistance> = unit_vectors
        .iter()
        .map(|a| VectorDistance {
            vector: a.vector,
            distance: cosine_similarity(&a.unit, &query_unit).abs(),
        })
        .collect();
    with_similarity
}

fn euclidean_distance(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    return a.iter().zip(b.iter()).map(|(a, b)| (b - a).powf(2.0)).sum();
}

fn cosine_similarity(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    return a.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
}
