# rust-vector-db

A vector similarity search library written in Rust. Supports multiple indexing strategies for efficient nearest neighbor search.

## Features

- **Flat Index** - Exact brute-force search with cosine similarity and Euclidean distance
- **IVF (Inverted File Index)** - Approximate search using clustering for faster queries on large datasets
- **PQ (Product Quantization)** - *Planned* - Compressed vector representation for memory-efficient search


## Usage

### Flat Index

Brute-force search that compares the query against all vectors. Best for small to medium datasets where exact results are required.

```rust
use rust_vector_db::flat_index::flat_index::{FlatIndex, FlatIndexStrategy};
use rust_vector_db::index::{Index, Searchable};
use rust_vector_db::vector::vector::VectorNode;

// Create an index with cosine similarity and 128 dimensions
let mut index = FlatIndex::new(FlatIndexStrategy::COSINE, 128);

// Add vectors
index.add(VectorNode::new_vector_node_with_id(vec![0.1; 128], 1))?;
index.add(VectorNode::new_vector_node_with_id(vec![0.2; 128], 2))?;

// Or add in batch
index.add_batch(vec![
    VectorNode::new_vector_node_with_id(vec![0.3; 128], 3),
    VectorNode::new_vector_node_with_id(vec![0.4; 128], 4),
])?;

// Search for k nearest neighbors
let query = vec![0.15; 128];
let results = index.search(&query, 5)?;

for result in results {
    println!("Found vector with id: {}", result.get_id());
}
```

### Distance Metrics

- `FlatIndexStrategy::COSINE` - Cosine similarity (normalized dot product)
- `FlatIndexStrategy::EUCLIDEAN` - Euclidean (L2) distance

## API

### Index Trait

```rust
trait Index {
    fn train(&mut self) -> Result<()>;
    fn add(&mut self, vector: VectorNode) -> Result<()>;
    fn add_batch(&mut self, vectors: Vec<VectorNode>) -> Result<()>;
    fn remove(&mut self, id: u64) -> Result<()>;
}
```

### Searchable Trait

```rust
trait Searchable {
    fn search(&self, vector: &Vec<f32>, k: u32) -> Result<Vec<&VectorNode>>;
}
```

## Binaries

### GloVe Lookup

An interactive tool for finding similar words using GloVe embeddings and the IVF index.

```bash
cargo run --bin glove_lookup <path-to-glove-file>
```

The tool loads [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings (100-dimensional), builds an IVF index with 25 clusters, and then accepts word queries from stdin. For each query, it returns the 5 most similar words.

**Example:**

```bash
$ cargo run --bin glove_lookup ~/data/glove.6B.100d.txt
Loaded Glove embeddings
Training IVF Index
king
Query time taken: 1.23ms
Top 0: king
Top 1: prince
Top 2: queen
Top 3: monarch
Top 4: kingdom
```

## Roadmap

- [x] Flat index with brute-force search
- [x] Cosine similarity
- [x] Euclidean distance
- [x] IVF (Inverted File Index)
- [ ] PQ (Product Quantization)
- [ ] HNSW (Hierarchical Navigable Small World)

## License

MIT
