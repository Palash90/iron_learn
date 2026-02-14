use regex::Regex;

pub fn tokenize(text: &str) -> Vec<String> {
    // We initialize the Regex inside, or use 'once_cell' for performance later
    let re = Regex::new(r"[\w']+|[.,!?;()]").unwrap();
    
    let sanitized = text
        .to_lowercase()
        .replace(['“', '”'], "\"")
        .replace(['‘', '’'], "'")
        .replace('—', " ");

    re.captures_iter(&sanitized)
        .map(|cap| cap[0].to_string())
        .collect()
}