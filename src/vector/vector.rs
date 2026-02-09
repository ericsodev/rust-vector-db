use atomic_id::{x64, AtomicId};

struct VectorNode {
    id: u64,
    vector: Vec<f32>,
}

pub fn new_vector_node(vector: Vec<f32>) -> VectorNode {
    let id = AtomicId::<u64>::new();
    return VectorNode { id, vector };
}

pub fn new_vector_node_with_id(vector: Vec<f32>, id: u64) -> VectorNode {
    return VectorNode { id, vector };
}
