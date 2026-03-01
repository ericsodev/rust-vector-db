use std::{
    collections::{HashMap, HashSet},
    vec,
};

use crate::{index::Index, vector::vector::VectorNode};

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
        self.centroids = get_initial_clusters(&self.vectors, self.num_centroids);
        assign_vectors_to_cluster(&self.vectors, &mut self.centroids);

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

fn assign_vector_to_cluster(index: &mut IVFIndex, vector: &VectorNode) -> bool {
    if index.centroids.len() == 0 {
        return false;
    }

    let mut cluster_distances: Vec<ClusterDistance> = index
        .centroids
        .iter_mut()
        .map(|v| ClusterDistance {
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

    cluster_distances.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

    cluster_distances
        .first_mut()
        .unwrap()
        .cluster
        .vectors
        .insert(vector.get_id());

    true
}

struct ClusterDistance<'a> {
    cluster: &'a mut Cluster,
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

    let mut cluster_distances: Vec<ClusterDistance<'a>> = clusters
        .into_iter()
        .map(|v| ClusterDistance {
            cluster: v,
            distance: 0.0,
        })
        .collect();

    for vector in vectors.iter() {
        for cluster in cluster_distances.iter_mut() {
            cluster.distance = cluster
                .cluster
                .centroid
                .iter()
                .zip(vector.1.get_vector().iter())
                .map(|(a, b)| (*b - *a).powf(2.0))
                .sum::<f32>();
        }

        cluster_distances.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        cluster_distances
            .first_mut()
            .unwrap()
            .cluster
            .vectors
            .insert(*vector.0);
    }
}
