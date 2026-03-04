use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    time::Instant,
    vec,
};

use crate::{
    index::{Index, Searchable},
    util::euclidean_distance::{self, calculate_euclidean_distance},
    vector::vector::VectorNode,
};

static NUM_PROBES: usize = 5;
static NUM_ROUNDS: usize = 10;

pub struct IVFIndex {
    dimension: usize,
    num_centroids: u32,
    vectors: HashMap<u64, VectorNode>,
    centroids: Vec<Cluster>,
}

struct Cluster {
    centroid: Vec<f32>,
    /// Set of IDs of vectors stored in the index vectors map
    vectors: HashSet<u64>,
}

impl Cluster {
    fn remove_vector(&mut self, id: u64) -> bool {
        self.vectors.remove(&id)
    }
}

impl IVFIndex {
    pub fn new(dimension: usize, centroids: u32) -> IVFIndex {
        return IVFIndex {
            dimension,
            num_centroids: centroids,
            centroids: vec![],
            vectors: HashMap::<u64, VectorNode>::new(),
        };
    }
}

impl Index for IVFIndex {
    fn remove(&mut self, id: u64) -> Result<(), String> {
        self.vectors.remove(&id);
        for c in self.centroids.iter_mut() {
            if c.remove_vector(id) {
                break;
            }
        }
        Ok(())
    }

    fn add_batch(&mut self, vectors: Vec<VectorNode>) -> Result<(), String> {
        for vec in vectors.iter() {
            if vec.get_vector().len() != self.dimension {
                return Err("Incorrect dimension".to_string());
            }
        }
        // Assign to nearest centroids if any
        for vec in vectors {
            let _ = self.add(vec);
        }
        Ok(())
    }

    fn add(&mut self, vector: VectorNode) -> Result<(), String> {
        if self.dimension != vector.get_vector().len() {
            return Err("Dimension mismatch".to_string());
        }
        // Assign to nearest centroid if any
        assign_vector_to_cluster(self, &vector);
        self.vectors.insert(vector.get_id(), vector);
        Ok(())
    }

    fn train(&mut self) -> Result<(), String> {
        if self.vectors.len() == 0 {
            return Ok(());
        }

        self.centroids = get_initial_clusters(&self.vectors, self.num_centroids);
        for i in 0..NUM_ROUNDS {
            let start = Instant::now();
            assign_vectors_to_cluster(&self.vectors, &mut self.centroids);
            reassign_cluster_centroid(self);
            println!(
                "Cluster IVF Round ({}/{}). Took {:?}",
                i,
                NUM_ROUNDS,
                start.elapsed()
            )
        }

        Ok(())
    }
}

fn get_initial_clusters<'a>(vectors: &HashMap<u64, VectorNode>, num_clusters: u32) -> Vec<Cluster> {
    let centers: Vec<Cluster> = vectors
        .iter()
        .take(num_clusters as usize)
        .map(|v| Cluster {
            centroid: v.1.get_vector().to_owned(),
            vectors: HashSet::new(),
        })
        .collect();

    centers
}

fn reassign_cluster_centroid(index: &mut IVFIndex) {
    for cluster in index.centroids.iter_mut() {
        let mut mean = vec![0.0; index.dimension];
        let mut total_vectors = 0;
        for id in cluster.vectors.iter() {
            let vec_result = index.vectors.get(&id);
            if let Some(vec) = vec_result {
                mean = mean
                    .par_iter()
                    .zip(vec.get_vector().par_iter())
                    .map(|(a, b)| *a + *b)
                    .collect();
                total_vectors += 1;
            } else {
                // Handle gracefully, remove from this centroids
            }
        }

        if total_vectors == 0 {
            cluster.centroid = mean;
            return;
        }

        cluster.centroid = mean.iter().map(|a| *a / total_vectors as f32).collect()
    }
}

fn assign_vector_to_cluster(index: &mut IVFIndex, vector: &VectorNode) -> bool {
    if index.centroids.len() == 0 {
        return false;
    }

    let mut cluster_distances: Vec<ClusterDistanceMut> = index
        .centroids
        .iter_mut()
        .map(|v| ClusterDistanceMut {
            cluster: v,
            distance: 0.0,
        })
        .collect();

    for cluster in cluster_distances.iter_mut() {
        cluster.distance = cluster
            .cluster
            .centroid
            .iter()
            .zip(vector.get_vector().iter())
            .map(|(a, b)| (*b - *a).powf(2.0))
            .sum::<f32>();
    }

    cluster_distances.par_sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

    cluster_distances
        .first_mut()
        .unwrap()
        .cluster
        .vectors
        .insert(vector.get_id());

    true
}

struct ClusterDistanceMut<'a> {
    cluster: &'a mut Cluster,
    distance: f32,
}

struct ClusterDistance<'a> {
    cluster: &'a Cluster,
    distance: f32,
}

fn assign_vectors_to_cluster<'a>(
    vectors: &'a HashMap<u64, VectorNode>,
    clusters: &'a mut Vec<Cluster>,
) {
    if clusters.is_empty() {
        return;
    }

    // Clear out cluster assigned vectors
    for cluster in clusters.iter_mut() {
        cluster.vectors.clear();
    }

    let assigned_pairs: Vec<(u64, usize)> = vectors
        .par_iter()
        .map(|vector| {
            let mut cluster_distances: Vec<(usize, f32, &Cluster)> = clusters
                .iter()
                .enumerate()
                .map(|(i, v)| (i, 0.0, v))
                .collect();

            for cluster in cluster_distances.iter_mut() {
                cluster.1 =
                    calculate_euclidean_distance(&cluster.2.centroid, vector.1.get_vector());
            }

            cluster_distances.par_sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            return (*vector.0, cluster_distances.first().unwrap().0);
        })
        .collect::<Vec<(u64, usize)>>();

    // Add to cluster
    for (vector_id, cluster_idx) in assigned_pairs.iter() {
        clusters
            .get_mut(*cluster_idx)
            .unwrap()
            .vectors
            .insert(*vector_id);
    }
}

impl Searchable for IVFIndex {
    fn search(&self, vector: &Vec<f32>, k: u32) -> Result<Vec<&VectorNode>, String> {
        let mut result = Vec::new();

        // Sort clusters by distance from centroid to vector
        let mut centroid_distances: Vec<ClusterDistance> = self
            .centroids
            .iter()
            .map(|c| ClusterDistance {
                cluster: c,
                distance: c
                    .centroid
                    .iter()
                    .zip(vector.iter())
                    .map(|(a, b)| (a - b).powf(2.0))
                    .sum(),
            })
            .collect();

        centroid_distances.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        // Collect all vectors in closest NUM_PROBES clusters
        let mut candidate_vectors: Vec<&VectorNode> = Vec::new();
        for centroid in centroid_distances.iter().take(NUM_PROBES) {
            candidate_vectors.extend(
                centroid
                    .cluster
                    .vectors
                    .iter()
                    .map(|id| self.vectors.get(id).unwrap())
                    .collect::<Vec<&VectorNode>>(),
            );
        }

        // Sort all candidate vectors by distance to query
        let mut vector_distances = candidate_vectors
            .par_iter()
            .map(|v| {
                let dist = v
                    .get_vector()
                    .iter()
                    .zip(vector.iter())
                    .map(|(a, b)| (a - b).powf(2.0))
                    .sum();
                (*v, dist)
            })
            .collect::<Vec<(&VectorNode, f32)>>();

        vector_distances.par_sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Take top k smallest distances
        result.extend(vector_distances.into_iter().take(k as usize).map(|v| v.0));

        Ok(result)
    }
}
