use std::sync::atomic::AtomicU64;

pub struct VectorNode {
    id: u64,
    vector: Vec<f32>,
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

impl VectorNode {
    pub fn new_vector_node(vector: Vec<f32>) -> VectorNode {
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return VectorNode { id, vector };
    }

    pub fn new_vector_node_with_id(vector: Vec<f32>, id: u64) -> VectorNode {
        return VectorNode { id, vector };
    }

    pub fn get_vector(&self) -> &Vec<f32> {
        return &self.vector;
    }

    pub fn get_id(&self) -> u64 {
        return self.id;
    }
}
