use crate::{
    flat_index::flat_index::{FlatIndex, FlatIndexStrategy},
    index::{Index, Searchable},
    vector::vector::VectorNode,
};

// =============================================================================
// Index trait tests
// =============================================================================

#[test]
fn test_train_returns_ok() {
    let mut index = FlatIndex::new(FlatIndexStrategy::COSINE, 4);
    let result = index.train();
    assert!(result.is_ok());
}

#[test]
fn test_add_single_vector() {
    let mut index = FlatIndex::new(FlatIndexStrategy::COSINE, 4);
    let vector = VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0, 4.0], 0);

    let result = index.add(vector);
    assert!(result.is_ok());
}

#[test]
fn test_add_rejects_wrong_dimension() {
    let mut index = FlatIndex::new(FlatIndexStrategy::COSINE, 4);
    let vector = VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0], 0); // 3 dims instead of 4

    let result = index.add(vector);
    assert!(result.is_err());
}

#[test]
fn test_add_batch_success() {
    let mut index = FlatIndex::new(FlatIndexStrategy::COSINE, 4);

    let result = index.add_batch(vec![
        VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0, 4.0], 0),
        VectorNode::new_vector_node_with_id(vec![2.0, 2.0, 3.0, 2.0], 1),
        VectorNode::new_vector_node_with_id(vec![0.0, 3.0, 3.0, 1.0], 2),
        VectorNode::new_vector_node_with_id(vec![0.0, 0.0, 0.0, 1.0], 3),
    ]);

    assert!(result.is_ok());
}

#[test]
fn test_add_batch_rejects_wrong_dimension() {
    let mut index = FlatIndex::new(FlatIndexStrategy::COSINE, 4);

    let result = index.add_batch(vec![
        VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0, 4.0], 0),
        VectorNode::new_vector_node_with_id(vec![2.0, 2.0, 3.0], 1), // Wrong dimension
    ]);

    assert!(result.is_err());
}

#[test]
fn test_remove_existing_vector() {
    let mut index = FlatIndex::new(FlatIndexStrategy::COSINE, 4);

    index
        .add(VectorNode::new_vector_node_with_id(
            vec![1.0, 2.0, 3.0, 4.0],
            42,
        ))
        .unwrap();

    let result = index.remove(42);
    assert!(result.is_ok());
}

#[test]
fn test_remove_nonexistent_vector() {
    let mut index = FlatIndex::new(FlatIndexStrategy::COSINE, 4);

    index
        .add(VectorNode::new_vector_node_with_id(
            vec![1.0, 2.0, 3.0, 4.0],
            42,
        ))
        .unwrap();

    let result = index.remove(999); // ID doesn't exist
    assert!(result.is_err());
}

#[test]
fn test_remove_from_empty_index() {
    let mut index = FlatIndex::new(FlatIndexStrategy::COSINE, 4);

    let result = index.remove(42);
    assert!(result.is_err());
}

// =============================================================================
// Searchable trait tests - Euclidean distance
// =============================================================================

#[test]
fn test_search_euclidean_returns_k_nearest() {
    let mut index = FlatIndex::new(FlatIndexStrategy::EUCLIDEAN, 4);

    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 0.0, 0.0, 0.0], 0),
            VectorNode::new_vector_node_with_id(vec![2.0, 0.0, 0.0, 0.0], 1),
            VectorNode::new_vector_node_with_id(vec![10.0, 0.0, 0.0, 0.0], 2),
            VectorNode::new_vector_node_with_id(vec![100.0, 0.0, 0.0, 0.0], 3),
        ])
        .unwrap();

    let query = vec![0.0, 0.0, 0.0, 0.0];
    let results = index.search(&query, 2).unwrap();

    assert_eq!(results.len(), 2);
    // Closest should be [1.0, 0, 0, 0] (id=0) and [2.0, 0, 0, 0] (id=1)
    assert_eq!(results[0].get_id(), 0);
    assert_eq!(results[1].get_id(), 1);
}

#[test]
fn test_search_euclidean_exact_match() {
    let mut index = FlatIndex::new(FlatIndexStrategy::EUCLIDEAN, 3);

    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0], 0),
            VectorNode::new_vector_node_with_id(vec![4.0, 5.0, 6.0], 1),
            VectorNode::new_vector_node_with_id(vec![7.0, 8.0, 9.0], 2),
        ])
        .unwrap();

    let query = vec![4.0, 5.0, 6.0]; // Exact match with id=1
    let results = index.search(&query, 1).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get_id(), 1);
}

#[test]
fn test_search_euclidean_rejects_wrong_dimension() {
    let mut index = FlatIndex::new(FlatIndexStrategy::EUCLIDEAN, 4);

    index
        .add(VectorNode::new_vector_node_with_id(
            vec![1.0, 2.0, 3.0, 4.0],
            0,
        ))
        .unwrap();

    let query = vec![1.0, 2.0, 3.0]; // 3 dims instead of 4
    let result = index.search(&query, 1);

    assert!(result.is_err());
}

// =============================================================================
// Searchable trait tests - Cosine similarity
// =============================================================================

#[test]
fn test_search_cosine_returns_k_nearest() {
    let mut index = FlatIndex::new(FlatIndexStrategy::COSINE, 3);

    // Vectors pointing in similar directions should be closer
    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 0.0, 0.0], 0), // Points along x-axis
            VectorNode::new_vector_node_with_id(vec![1.0, 0.1, 0.0], 1), // Slightly off x-axis
            VectorNode::new_vector_node_with_id(vec![0.0, 1.0, 0.0], 2), // Points along y-axis (perpendicular)
            VectorNode::new_vector_node_with_id(vec![-1.0, 0.0, 0.0], 3), // Opposite direction
        ])
        .unwrap();

    let query = vec![1.0, 0.0, 0.0]; // Query along x-axis
    let results = index.search(&query, 2).unwrap();

    assert_eq!(results.len(), 2);
    // Most similar should be [1, 0, 0] (id=0) and [1, 0.1, 0] (id=1)
    assert_eq!(results[0].get_id(), 0);
    assert_eq!(results[1].get_id(), 1);
}

#[test]
fn test_search_cosine_identical_vectors() {
    let mut index = FlatIndex::new(FlatIndexStrategy::COSINE, 3);

    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0], 0),
            VectorNode::new_vector_node_with_id(vec![2.0, 4.0, 6.0], 1), // Same direction, different magnitude
            VectorNode::new_vector_node_with_id(vec![0.0, 0.0, 1.0], 2),
        ])
        .unwrap();

    let query = vec![1.0, 2.0, 3.0];
    let results = index.search(&query, 2).unwrap();

    assert_eq!(results.len(), 2);
    // Both id=0 and id=1 should have cosine similarity of 1.0 (same direction)
    let ids: Vec<u64> = results.iter().map(|v| v.get_id()).collect();
    assert!(ids.contains(&0));
    assert!(ids.contains(&1));
}

#[test]
fn test_search_cosine_rejects_wrong_dimension() {
    let mut index = FlatIndex::new(FlatIndexStrategy::COSINE, 4);

    index
        .add(VectorNode::new_vector_node_with_id(
            vec![1.0, 2.0, 3.0, 4.0],
            0,
        ))
        .unwrap();

    let query = vec![1.0, 2.0]; // 2 dims instead of 4
    let result = index.search(&query, 1);

    assert!(result.is_err());
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn test_search_k_larger_than_index_size() {
    let mut index = FlatIndex::new(FlatIndexStrategy::EUCLIDEAN, 3);

    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0], 0),
            VectorNode::new_vector_node_with_id(vec![4.0, 5.0, 6.0], 1),
        ])
        .unwrap();

    let query = vec![0.0, 0.0, 0.0];
    let results = index.search(&query, 10).unwrap(); // k=10, but only 2 vectors

    // Should return at most 2 results
    assert!(results.len() <= 2);
}

#[test]
fn test_add_multiple_vectors_sequentially() {
    let mut index = FlatIndex::new(FlatIndexStrategy::EUCLIDEAN, 3);

    index
        .add(VectorNode::new_vector_node_with_id(vec![1.0, 0.0, 0.0], 0))
        .unwrap();
    index
        .add(VectorNode::new_vector_node_with_id(vec![0.0, 1.0, 0.0], 1))
        .unwrap();
    index
        .add(VectorNode::new_vector_node_with_id(vec![0.0, 0.0, 1.0], 2))
        .unwrap();

    let query = vec![1.0, 0.0, 0.0];
    let results = index.search(&query, 3).unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].get_id(), 0); // Exact match should be first
}

#[test]
fn test_remove_then_search() {
    let mut index = FlatIndex::new(FlatIndexStrategy::EUCLIDEAN, 3);

    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 0.0, 0.0], 0),
            VectorNode::new_vector_node_with_id(vec![2.0, 0.0, 0.0], 1),
            VectorNode::new_vector_node_with_id(vec![3.0, 0.0, 0.0], 2),
        ])
        .unwrap();

    // Remove the closest vector
    index.remove(0).unwrap();

    let query = vec![0.0, 0.0, 0.0];
    let results = index.search(&query, 2).unwrap();

    assert_eq!(results.len(), 2);
    // After removing id=0, closest should now be id=1
    assert_eq!(results[0].get_id(), 1);
}
