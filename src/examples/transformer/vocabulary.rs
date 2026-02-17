use super::tokenizer;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct Vocabulary {
    pub stoi: HashMap<String, usize>,
    pub itos: HashMap<usize, String>,
    pub vocab_size: u32,
}

pub fn build_vocabulary(data_path: &str) -> Result<(Vocabulary, Vec<String>), String> {
    let file = File::open(data_path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().map(|l| l.expect("Read error")).collect();

    let mut words = HashSet::new();
    words.insert("<PAD>".to_string());
    words.insert("<END>".to_string());

    for line in &lines {
        for token in tokenizer::tokenize_graphemes(line) {
            words.insert(token);
        }
    }

    let mut words_vec: Vec<String> = words.into_iter().collect();
    words_vec.sort();
    let vocab_size = words_vec.len() as u32;

    let stoi: HashMap<String, usize> = words_vec
        .iter()
        .enumerate()
        .map(|(i, w)| (w.clone(), i))
        .collect();

    let itos: HashMap<usize, String> = words_vec
        .iter()
        .enumerate()
        .map(|(i, w)| (i, w.clone()))
        .collect();

    Ok((
        Vocabulary {
            stoi,
            itos,
            vocab_size,
        },
        lines,
    ))
}
