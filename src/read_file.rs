use std::fs;

use crate::{neural_network::ModelData, runners::DataDoublePrecision, Data};

pub fn deserialize_data(data_path: &str) -> Result<Data, serde_json::Error> {
    let contents =
        fs::read_to_string(&data_path).expect(&format!("Failed to read data from {}", data_path));
    let data: Data = match serde_json::from_str(&contents) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("Failed to parse JSON data: {}", err);
            return Result::Err(err);
        }
    };
    Ok(data.clone())
}

pub fn deserialize_model(data_path: &str) -> Option<ModelData> {
    let contents =
        fs::read_to_string(&data_path).expect(&format!("Failed to read data from {}", data_path));
    let data: ModelData = match serde_json::from_str(&contents) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("Failed to parse Model data: {}", err);
            return None;
        }
    };
    Some(data)
}

pub fn deserialize_data_double_precision(
    data_path: &str,
) -> Result<DataDoublePrecision, serde_json::Error> {
    let contents =
        fs::read_to_string(&data_path).expect(&format!("Failed to read data from {}", data_path));
    let data: DataDoublePrecision = match serde_json::from_str(&contents) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("Failed to parse JSON data: {}", err);
            return Result::Err(err);
        }
    };
    Ok(data.clone())
}
