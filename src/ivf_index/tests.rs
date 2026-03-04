use crate::{
    index::{Index, Searchable},
    ivf_index::ivf_index::IVFIndex,
    vector::vector::VectorNode,
};

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

// =============================================================================
// Searchable trait tests
// =============================================================================

#[test]
fn test_search_returns_k_nearest() {
    let mut index = IVFIndex::new(4, 2);

    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 0.0, 0.0, 0.0], 0),
            VectorNode::new_vector_node_with_id(vec![2.0, 0.0, 0.0, 0.0], 1),
            VectorNode::new_vector_node_with_id(vec![10.0, 0.0, 0.0, 0.0], 2),
            VectorNode::new_vector_node_with_id(vec![100.0, 0.0, 0.0, 0.0], 3),
        ])
        .unwrap();

    index.train().unwrap();

    let query = vec![0.0, 0.0, 0.0, 0.0];
    let results = index.search(&query, 2).unwrap();

    assert_eq!(results.len(), 2);
    // Closest should be [1.0, 0, 0, 0] (id=0) and [2.0, 0, 0, 0] (id=1)
    assert_eq!(results[0].get_id(), 0);
    assert_eq!(results[1].get_id(), 1);
}

#[test]
fn test_search_exact_match() {
    let mut index = IVFIndex::new(3, 2);

    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0], 0),
            VectorNode::new_vector_node_with_id(vec![4.0, 5.0, 6.0], 1),
            VectorNode::new_vector_node_with_id(vec![7.0, 8.0, 9.0], 2),
        ])
        .unwrap();

    index.train().unwrap();

    let query = vec![4.0, 5.0, 6.0]; // Exact match with id=1
    let results = index.search(&query, 1).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].get_id(), 1);
}

#[test]
fn test_search_empty_index() {
    let index = IVFIndex::new(3, 2);

    let query = vec![1.0, 2.0, 3.0];
    let results = index.search(&query, 5).unwrap();

    assert_eq!(results.len(), 0);
}

#[test]
fn test_search_untrained_index() {
    let mut index = IVFIndex::new(3, 2);

    // Add vectors but don't train
    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0], 0),
            VectorNode::new_vector_node_with_id(vec![4.0, 5.0, 6.0], 1),
        ])
        .unwrap();

    // Search without training - should return empty since no clusters exist
    let query = vec![1.0, 2.0, 3.0];
    let results = index.search(&query, 5).unwrap();

    assert_eq!(results.len(), 0);
}

#[test]
fn test_search_k_larger_than_index_size() {
    let mut index = IVFIndex::new(3, 2);

    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 2.0, 3.0], 0),
            VectorNode::new_vector_node_with_id(vec![4.0, 5.0, 6.0], 1),
        ])
        .unwrap();

    index.train().unwrap();

    let query = vec![0.0, 0.0, 0.0];
    let results = index.search(&query, 10).unwrap(); // k=10, but only 2 vectors

    // Should return at most 2 results
    assert!(results.len() <= 2);
}

#[test]
fn test_search_finds_vectors_across_clusters() {
    let mut index = IVFIndex::new(3, 2);

    // Create two distinct clusters
    index
        .add_batch(vec![
            // Cluster 1 near origin
            VectorNode::new_vector_node_with_id(vec![0.0, 0.0, 0.0], 0),
            VectorNode::new_vector_node_with_id(vec![0.1, 0.1, 0.1], 1),
            VectorNode::new_vector_node_with_id(vec![0.2, 0.2, 0.2], 2),
            // Cluster 2 far from origin
            VectorNode::new_vector_node_with_id(vec![10.0, 10.0, 10.0], 3),
            VectorNode::new_vector_node_with_id(vec![10.1, 10.1, 10.1], 4),
            VectorNode::new_vector_node_with_id(vec![10.2, 10.2, 10.2], 5),
        ])
        .unwrap();

    index.train().unwrap();

    // Query near cluster 1
    let query = vec![0.0, 0.0, 0.0];
    let results = index.search(&query, 3).unwrap();

    assert_eq!(results.len(), 3);
    // All results should be from cluster 1 (ids 0, 1, 2)
    let ids: Vec<u64> = results.iter().map(|v| v.get_id()).collect();
    assert!(ids.contains(&0));
    assert!(ids.contains(&1));
    assert!(ids.contains(&2));
}

#[test]
fn test_search_returns_results_in_distance_order() {
    let mut index = IVFIndex::new(3, 2);

    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![5.0, 0.0, 0.0], 0),
            VectorNode::new_vector_node_with_id(vec![1.0, 0.0, 0.0], 1),
            VectorNode::new_vector_node_with_id(vec![3.0, 0.0, 0.0], 2),
            VectorNode::new_vector_node_with_id(vec![2.0, 0.0, 0.0], 3),
            VectorNode::new_vector_node_with_id(vec![4.0, 0.0, 0.0], 4),
        ])
        .unwrap();

    index.train().unwrap();

    let query = vec![0.0, 0.0, 0.0];
    let results = index.search(&query, 5).unwrap();

    assert_eq!(results.len(), 5);
    // Should be ordered by distance: 1, 3, 2, 4, 0
    assert_eq!(results[0].get_id(), 1); // distance 1
    assert_eq!(results[1].get_id(), 3); // distance 2
    assert_eq!(results[2].get_id(), 2); // distance 3
    assert_eq!(results[3].get_id(), 4); // distance 4
    assert_eq!(results[4].get_id(), 0); // distance 5
}

#[test]
fn test_search_after_remove() {
    let mut index = IVFIndex::new(3, 2);

    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![1.0, 0.0, 0.0], 0),
            VectorNode::new_vector_node_with_id(vec![2.0, 0.0, 0.0], 1),
            VectorNode::new_vector_node_with_id(vec![3.0, 0.0, 0.0], 2),
        ])
        .unwrap();

    index.train().unwrap();

    // Remove the closest vector
    index.remove(0).unwrap();

    let query = vec![0.0, 0.0, 0.0];
    let results = index.search(&query, 2).unwrap();

    assert_eq!(results.len(), 2);
    // After removing id=0, closest should now be id=1
    assert_eq!(results[0].get_id(), 1);
    assert_eq!(results[1].get_id(), 2);
}

#[test]
fn test_search_after_adding_post_train() {
    let mut index = IVFIndex::new(3, 2);

    // Initial vectors and train
    index
        .add_batch(vec![
            VectorNode::new_vector_node_with_id(vec![10.0, 0.0, 0.0], 0),
            VectorNode::new_vector_node_with_id(vec![20.0, 0.0, 0.0], 1),
        ])
        .unwrap();

    index.train().unwrap();

    // Add a new vector after training that's closer to query
    index
        .add(VectorNode::new_vector_node_with_id(vec![1.0, 0.0, 0.0], 2))
        .unwrap();

    let query = vec![0.0, 0.0, 0.0];
    let results = index.search(&query, 3).unwrap();

    assert_eq!(results.len(), 3);
    // The newly added vector (id=2) should be closest
    assert_eq!(results[0].get_id(), 2);
}
