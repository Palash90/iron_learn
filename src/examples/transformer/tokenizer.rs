use unicode_segmentation::UnicodeSegmentation;

pub fn tokenize_graphemes(text: &str) -> Vec<String> {
    // true = use extended grapheme clusters
    text.graphemes(true).map(|s| s.to_string()).collect()
}
