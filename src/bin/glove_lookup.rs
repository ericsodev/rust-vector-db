use std::{
    collections::{HashMap, HashSet},
    env,
    fs::{canonicalize, File},
    io::{stdin, BufRead, BufReader},
    path::Path,
    time::Instant,
};

use rust_vector_db::{
    index::{Index, Searchable},
    ivf_index::ivf_index::IVFIndex,
    vector::vector::VectorNode,
};

static GLOVE_DIMENSIONS: usize = 100;

fn load_glove_embeddings(index: &mut IVFIndex, filepath: &str) -> Result<(), std::io::Error> {
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

    let _ = index.train().unwrap();

    println!("Training IVF Index");

    Ok(())
}

fn main() {
    let mut index = IVFIndex::new(GLOVE_DIMENSIONS, 25);

    let mut path: String;

    if let Some(arg) = env::args().nth(1) {
        path = shellexpand::full(&arg).unwrap().as_ref().to_owned();
    } else {
        panic!("Missing user supplied path to GloVe embeddings.")
    }

    load_glove_embeddings(&mut index, &path).unwrap();

    fn lookup_vec_from_word(word: &str, path: &str) -> Option<Vec<f32>> {
        let path = Path::new(path);
        let canon_path = canonicalize(path).unwrap();
        let file = File::open(canon_path).unwrap();
        let reader = BufReader::new(file);
        for line in reader.lines() {
            if let Ok(s) = line {
                let mut tokens = s.split_terminator(' ');
                match tokens.next() {
                    Some(s) => {
                        if s != word {
                            continue;
                        }

                        return Some(
                            tokens
                                .map(|v| v.parse::<f32>().unwrap())
                                .collect::<Vec<f32>>(),
                        );
                    }
                    None => continue,
                }
            }
        }

        return None;
    }

    fn lookup_word_from_indices(
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

    loop {
        let mut input = String::new();
        let bytes_read = stdin().read_line(&mut input).unwrap();
        if bytes_read == 0 {
            break;
        }

        let len_before = input.len();
        let trimmed_len = input.trim_end_matches(&['\r', '\n'][..]).len();
        if trimmed_len != len_before {
            input.truncate(trimmed_len);
        }

        // Lookup glove embedding in file
        let query_vector = lookup_vec_from_word(&input, &path);
        if let Some(vec) = query_vector {
            let start = Instant::now();
            let result = index.search(&vec, 5).unwrap();
            let duration = start.elapsed();

            println!("Query time taken: {:?}", duration);

            let result_ids = result.iter().map(|v| v.get_id());
            let words_from_result = lookup_word_from_indices(result_ids, &path);

            for (i, vec) in result.iter().enumerate() {
                let default = "UNKNOWN WORD".to_string();
                let word = words_from_result.get(&vec.get_id()).unwrap_or(&default);
                println!("Top {}: {}", i, word)
            }
        } else {
            println!("Word '{input}' is not in dictionary {path}")
        }
    }
}
