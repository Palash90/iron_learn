use std::fs;

use super::types::Data;
use crate::{nn::ModelData, numeric::FloatingPoint};

/// Read and parse a dataset JSON file into a `Data` structure.
///
/// Returns a `Result` with parsed `Data` on success or a JSON parsing
/// error on failure. The function will print a diagnostic message if
/// the file cannot be read.
pub fn deserialize_data<D>(data_path: &str) -> Result<Data<D>, serde_json::Error>
where
    D: FloatingPoint,
{
    let contents =
        fs::read_to_string(&data_path).expect(&format!("Failed to read data from {}", data_path));
    let data: Data<D> = match serde_json::from_str(&contents) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("Failed to parse JSON data: {}", err);
            return Result::Err(err);
        }
    };
    Ok(data.clone())
}

/// Attempt to read and parse a saved `ModelData` JSON file.
///
/// Returns `Some(ModelData)` on success or `None` if reading/parsing fails.
pub fn deserialize_model<D>(data_path: &str) -> Option<ModelData<D>>
where
    D: FloatingPoint,
{
    let contents = match fs::read_to_string(&data_path) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("{}", &format!("Failed to read data from {}", data_path));
            String::new()
        }
    };

    let data: ModelData<D> = match serde_json::from_str(&contents) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("Failed to parse Model data: {}", err);
            return None;
        }
    };
    Some(data)
}
