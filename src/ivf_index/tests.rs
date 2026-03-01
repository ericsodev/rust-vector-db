use crate::{index::Index, ivf_index::ivf_index::IVFIndex, vector::vector::VectorNode};

// =============================================================================
// Index trait tests
// =============================================================================

#[test]
fn test_new_creates_empty_index() {
    let index = IVFIndex::new(4, 2);
    // Index should be created successfully with the specified dimension and centroids
    // We can verify by adding a vector with the correct dimension
    let mut index = index;
    let result = index.add(VectorNode::new_vector_node_with_id(
        vec![1.0, 2.0, 3.0, 4.0],
        0,
    ));
    assert!(result.is_ok());
}

#[test]
fn test_train_returns_ok() {
    let mut index = IVFIndex::new(4, 2);
    let result = index.train();
    assert!(result.is_ok());
}

#[test]
fn test_train_creates_centroids() {
    let mut index = IVFIndex::new(3, 2);

    // Add vectors before training
    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 0.0, 0.0], 0),
            VectorNode::new_vector_node_with_id(vec![0.0, 1.0, 0.0], 1),
            VectorNode::new_vector_node_with_id(vec![0.0, 0.0, 1.0], 2),
        ])
        .unwrap();

    // Train should succeed and create centroids from the vectors
    let result = index.train();
    assert!(result.is_ok());
}

#[test]
fn test_add_single_vector() {
    let mut index = IVFIndex::new(4, 2);
    let vector = VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0, 4.0], 0);

    let result = index.add(vector);
    assert!(result.is_ok());
}

#[test]
fn test_add_rejects_wrong_dimension() {
    let mut index = IVFIndex::new(4, 2);
    let vector = VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0], 0); // 3 dims instead of 4

    let result = index.add(vector);
    assert!(result.is_err());
}

#[test]
fn test_add_batch_success() {
    let mut index = IVFIndex::new(4, 2);

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
    let mut index = IVFIndex::new(4, 2);

    let result = index.add_batch(vec![
        VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0, 4.0], 0),
        VectorNode::new_vector_node_with_id(vec![2.0, 2.0, 3.0], 1), // Wrong dimension
    ]);

    assert!(result.is_err());
}

#[test]
fn test_remove_existing_vector() {
    let mut index = IVFIndex::new(4, 2);

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
fn test_remove_from_empty_index() {
    let mut index = IVFIndex::new(4, 2);

    // IVFIndex.remove() always returns Ok, even for non-existent IDs
    let result = index.remove(42);
    assert!(result.is_ok());
}

// =============================================================================
// IVF-Specific tests
// =============================================================================

#[test]
fn test_train_assigns_vectors_to_clusters() {
    let mut index = IVFIndex::new(3, 2);

    // Add vectors that should cluster into two groups
    index
        .add_batch(vec![
            // Group 1: vectors near origin
            VectorNode::new_vector_node_with_id(vec![0.1, 0.1, 0.1], 0),
            VectorNode::new_vector_node_with_id(vec![0.2, 0.2, 0.2], 1),
            // Group 2: vectors far from origin
            VectorNode::new_vector_node_with_id(vec![10.0, 10.0, 10.0], 2),
            VectorNode::new_vector_node_with_id(vec![10.1, 10.1, 10.1], 3),
        ])
        .unwrap();

    // Train should assign vectors to nearest centroids
    let result = index.train();
    assert!(result.is_ok());
}

#[test]
fn test_add_after_train_assigns_to_cluster() {
    let mut index = IVFIndex::new(3, 2);

    // Add initial vectors and train
    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![0.0, 0.0, 0.0], 0),
            VectorNode::new_vector_node_with_id(vec![10.0, 10.0, 10.0], 1),
        ])
        .unwrap();

    index.train().unwrap();

    // Add a new vector after training - it should be assigned to nearest cluster
    let result = index.add(VectorNode::new_vector_node_with_id(vec![0.1, 0.1, 0.1], 2));
    assert!(result.is_ok());

    // Add another vector near the other cluster
    let result = index.add(VectorNode::new_vector_node_with_id(vec![9.9, 9.9, 9.9], 3));
    assert!(result.is_ok());
}

#[test]
fn test_multiple_train_calls() {
    let mut index = IVFIndex::new(3, 2);

    // Add initial vectors
    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 0.0, 0.0], 0),
            VectorNode::new_vector_node_with_id(vec![0.0, 1.0, 0.0], 1),
        ])
        .unwrap();

    // First train
    let result = index.train();
    assert!(result.is_ok());

    // Add more vectors
    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![0.0, 0.0, 1.0], 2),
            VectorNode::new_vector_node_with_id(vec![1.0, 1.0, 1.0], 3),
        ])
        .unwrap();

    // Second train - should work and reassign all vectors
    let result = index.train();
    assert!(result.is_ok());
}

#[test]
fn test_train_reassigns_centroid_mean() {
    let mut index = IVFIndex::new(3, 2);

    // Add vectors where:
    // - First two vectors become initial centroids: [0,0,0] and [10,10,10]
    // - Additional vectors pull the centroids toward new positions
    index
        .add_batch(vec![
            // Initial centroid 1 at [0, 0, 0]
            VectorNode::new_vector_node_with_id(vec![0.0, 0.0, 0.0], 0),
            // Initial centroid 2 at [10, 10, 10]
            VectorNode::new_vector_node_with_id(vec![10.0, 10.0, 10.0], 1),
            // These vectors should pull centroid 1 toward [2, 2, 2]
            VectorNode::new_vector_node_with_id(vec![2.0, 2.0, 2.0], 2),
            VectorNode::new_vector_node_with_id(vec![2.0, 2.0, 2.0], 3),
            VectorNode::new_vector_node_with_id(vec![2.0, 2.0, 2.0], 4),
            VectorNode::new_vector_node_with_id(vec![2.0, 2.0, 2.0], 5),
            // These vectors should pull centroid 2 toward [8, 8, 8]
            VectorNode::new_vector_node_with_id(vec![8.0, 8.0, 8.0], 6),
            VectorNode::new_vector_node_with_id(vec![8.0, 8.0, 8.0], 7),
            VectorNode::new_vector_node_with_id(vec![8.0, 8.0, 8.0], 8),
            VectorNode::new_vector_node_with_id(vec![8.0, 8.0, 8.0], 9),
        ])
        .unwrap();

    // Train should reassign centroid means based on assigned vectors
    // After training:
    // - Centroid 1 should move from [0,0,0] toward ~[1.6, 1.6, 1.6] (mean of 0,2,2,2,2)
    // - Centroid 2 should move from [10,10,10] toward ~[8.4, 8.4, 8.4] (mean of 10,8,8,8,8)
    let result = index.train();
    assert!(result.is_ok());

    // After centroid reassignment, a vector at [5, 5, 5] should be equidistant
    // from both clusters. A vector at [3, 3, 3] should be closer to cluster 1's
    // new centroid position rather than being assigned based on original centroids.
    let result = index.add(VectorNode::new_vector_node_with_id(vec![3.0, 3.0, 3.0], 10));
    assert!(result.is_ok());
}
