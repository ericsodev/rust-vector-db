use std::{
    collections::{HashMap, HashSet},
    env,
    fs::{canonicalize, File},
    io::{BufRead, BufReader},
    path::Path,
    time,
};

use rust_vector_db::{
    flat_index::flat_index::{self, FlatIndex},
    index::Searchable,
    ivf_index::ivf_index::IVFIndex,
    vector::vector::VectorNode,
};

struct QueryResult {
    training_time: u128,
    total_query_time: u128,
    avg_query_time: u128,
    max_query_time: u128,
    min_query_time: u128,
    response: HashMap<String, Vec<String>>,
}

static GLOVE_DIM: usize = 100;

fn get_vectors_from_words(words: &Vec<&str>, filepath: &str) -> Vec<(String, Vec<f32>)> {
    let path = Path::new(&filepath);
    let canon_path = canonicalize(path).unwrap();
    let file = File::open(canon_path).unwrap();
    let reader = BufReader::new(file);

    let word_set: HashSet<&str> = HashSet::from_iter(words.iter().map(|v| v.to_owned()));
    let mut result: Vec<(String, Vec<f32>)> = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if let Ok(s) = line {
            let mut tokens = s.split_terminator(' ');
            let word = tokens.next().unwrap();
            if !word_set.contains(word) {
                continue;
            }

            let embedding = tokens
                .map(|v| match v.parse::<f32>() {
                    Ok(f) => f,
                    Err(_) => panic!("Bad float token {} (line {}) in \n {}", v, i, s),
                })
                .collect::<Vec<f32>>();
            if embedding.len() != GLOVE_DIM {
                panic!(
                    "Embedding for {} is {} long. Expected {}.",
                    word,
                    embedding.len(),
                    GLOVE_DIM
                )
            }
            result.push((word.to_string(), embedding));
        }
    }

    result
}

fn load_glove_embeddings(
    index: &mut Box<dyn Searchable>,
    filepath: &str,
) -> Result<(), std::io::Error> {
    let path = Path::new(&filepath);
    let canon_path = canonicalize(path).unwrap();
    let file = File::open(canon_path)?;
    let reader = BufReader::new(file);

    for (i, line) in reader.lines().enumerate() {
        if let Ok(s) = line {
            let tokens = s.split_terminator(' ');
            let embedding = tokens
                .skip(1)
                .map(|v| match v.parse::<f32>() {
                    Ok(f) => f,
                    Err(_) => panic!("Bad float token {} (line {}) in \n {}", v, i, s),
                })
                .collect::<Vec<f32>>();
            let vector = VectorNode::new_vector_node_with_id(embedding, i.try_into().unwrap());

            let _ = index.add(vector).unwrap();
        }
    }

    println!("Loaded Glove embeddings");

    Ok(())
}

fn lookup_glove_word_from_indices(
    indices: impl Iterator<Item = u64>,
    path: &str,
) -> HashMap<u64, String> {
    let path = Path::new(path);
    let canon_path = canonicalize(path).unwrap();
    let file = File::open(canon_path).unwrap();
    let reader = BufReader::new(file);
    let mut index_set: HashSet<u64> = HashSet::new();
    index_set.extend(indices);

    let mut result: HashMap<u64, String> = HashMap::new();

    for (i, line) in reader.lines().enumerate() {
        if let Ok(s) = line {
            let mut tokens = s.split_whitespace();
            match tokens.next() {
                Some(s) => {
                    if !index_set.contains(&(i as u64)) {
                        continue;
                    }

                    result.insert(i as u64, s.to_string());
                }
                None => continue,
            }
        }
    }

    result
}

fn benchmark_index(
    index: &mut Box<dyn Searchable>,
    k: u32,
    words: &Vec<(String, Vec<f32>)>,
    glove_path: &str,
) -> Result<QueryResult, String> {
    let mut now = time::Instant::now();
    let _ = load_glove_embeddings(index, glove_path).unwrap();
    let insert_time = now.elapsed().as_millis();
    println!("Loaded GloVe embeddings in {}ms", insert_time);

    now = time::Instant::now();
    let _ = index.train().unwrap();

    let training_time = now.elapsed().as_millis();
    println!("Total training time: {}ms", training_time);

    let mut query_times: Vec<u128> = Vec::new();

    let mut search_results: Vec<Vec<&VectorNode>> = Vec::new();

    for word in words {
        let now = time::Instant::now();
        let result = index.search(&word.1, k).unwrap();
        search_results.push(result);
        query_times.push(now.elapsed().as_millis());
    }

    let id_to_word_map = lookup_glove_word_from_indices(
        search_results
            .iter()
            .map(|v| v.iter().map(|v| v.get_id()).collect::<Vec<u64>>())
            .flatten(),
        glove_path,
    );

    let mut query_result_map: HashMap<String, Vec<String>> = HashMap::new();
    for (query, result) in words.iter().zip(search_results.iter()) {
        let result_words: Vec<String> = result
            .iter()
            .map(|v| id_to_word_map.get(&v.get_id()).unwrap().to_owned())
            .collect();

        query_result_map.insert(query.0.to_string(), result_words);
    }

    let avg_query = query_times.iter().sum::<u128>() / (query_times.len() as u128);
    let max_query = *query_times.iter().max().unwrap();
    let min_query = *query_times.iter().min().unwrap();
    let total_query = query_times.iter().sum::<u128>();

    return Ok({
        QueryResult {
            training_time,
            total_query_time: total_query,
            avg_query_time: avg_query,
            max_query_time: max_query,

            min_query_time: min_query,
            response: query_result_map,
        }
    });
}

fn main() {
    let test_words = vec![
        "hello",
        "fox",
        "bird",
        "tree",
        "code",
        "rust",
        "vector",
        "database",
        "a",
        "single",
        "query",
        "result",
        "benchmark",
        "test",
        "is",
        "hard",
        "to",
        "write",
        "random",
        "foo",
        "cosine",
    ];

    let glove_path: String;

    if let Some(arg) = env::args().nth(1) {
        glove_path = shellexpand::full(&arg).unwrap().as_ref().to_owned();
    } else {
        panic!("Missing user supplied path to GloVe embeddings.")
    }

    let word_to_vector: Vec<(String, Vec<f32>)> = get_vectors_from_words(&test_words, &glove_path);

    let mut indexes: Vec<Box<dyn Searchable>> = vec![
        Box::new(FlatIndex::new(
            flat_index::FlatIndexStrategy::EUCLIDEAN,
            GLOVE_DIM,
        )),
        Box::new(IVFIndex::new(GLOVE_DIM, 16, 3, 5)),
        Box::new(IVFIndex::new(GLOVE_DIM, 32, 5, 5)),
        Box::new(IVFIndex::new(GLOVE_DIM, 48, 3, 5)),
    ];

    let top_k = 50;

    let mut results: Vec<QueryResult> = Vec::new();
    for index in indexes.iter_mut() {
        results.push(benchmark_index(index, top_k, &word_to_vector, &glove_path).unwrap());
    }

    // we can use flat euclidean as the benchmark for accuracy. as we know it always returns the
    // closest vectors

    // NOTE: assuming flat index is first in results
    let flat_index_results = &results[0];

    for (result, index) in results.iter().zip(indexes.iter()) {
        let accuracy = calculate_accuracy(flat_index_results, result);
        println!("=============================================");
        index.print_configuration();
        println!("=============================================\n");
        println!("Training time: {}ms", result.training_time);
        println!("Average query time: {}ms", result.avg_query_time);
        println!("Min query time: {}ms", result.min_query_time);
        println!("Max query time: {}ms", result.max_query_time);
        println!("Total query time: {}ms", result.total_query_time);
        println!("Accuracy: {:.2}%", accuracy * 100.0);
        println!();
        println!();
    }
}

/// Compares the response result against flat_result (ground truth).
/// Returns recall@k: the fraction of true nearest neighbors that were found.
fn calculate_accuracy(flat_result: &QueryResult, result: &QueryResult) -> f32 {
    let mut total_recall = 0.0;
    let mut query_count = 0;

    for (query_word, ground_truth) in flat_result.response.iter() {
        if let Some(approximate) = result.response.get(query_word) {
            let ground_truth_set: HashSet<&String> = ground_truth.iter().collect();
            let approximate_set: HashSet<&String> = approximate.iter().collect();

            // Count how many of the ground truth results were found
            let intersection = ground_truth_set.intersection(&approximate_set).count();

            if !ground_truth_set.is_empty() {
                total_recall += intersection as f32 / ground_truth_set.len() as f32;
            }
            query_count += 1;
        }
    }

    if query_count == 0 {
        return 0.0;
    }

    total_recall / query_count as f32
}
