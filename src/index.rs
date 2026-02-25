use crate::vector::vector::VectorNode;

pub trait Index {
    fn train(&mut self) -> Result<(), String>;
    fn add(&mut self, vector: VectorNode) -> Result<(), String>;
    fn add_batch(&mut self, vectors: Vec<VectorNode>) -> Result<(), String>;
    fn remove(&mut self, id: u64) -> Result<(), String>;
}

pub trait Searchable {
    fn search(&self, vector: &Vec<f32>, k: u32) -> Result<Vec<&VectorNode>, String>;
}
