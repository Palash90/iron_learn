import time

import cupy as np
import numpy as onp
from transformer import Transformer
from simple_layers import Softmax, CrossEntropyLoss

try:
    np.cuda.Device(0).use()
    print("GPU found and will be used for training.")
except Exception as e:
    print(f"GPU not found or error: {e}")

class CorpusTokenizer:
    def __init__(self, corpus_text):
        # 1. Clean and Wrap lines with special tokens
        lines = corpus_text.splitlines()
        lines = [line.strip() for line in lines if line.strip()]
        
        # 2. Define special tokens
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>"]
        
        # 3. Build Word-Level Vocab
        # We split the text by spaces to get unique words
        words = set()
        for line in lines:
            words.update(line.split())
        
        self.vocab = self.special_tokens + sorted(list(words))
        
        self.char_to_id = {w: i for i, w in enumerate(self.vocab)}
        self.id_to_char = {i: w for i, w in enumerate(self.vocab)}
        
        self.pad_id = self.char_to_id["<PAD>"]
        self.sos_id = self.char_to_id["<SOS>"]
        self.eos_id = self.char_to_id["<EOS>"]
        self.vocab_size = len(self.vocab)

    def encode_sentence(self, text, seq_len):
        # Split text into words to match the vocab
        words = text.split()
        
        # Format: <SOS> + [words] + <EOS>
        tokens = [self.sos_id] + [self.char_to_id[w] for w in words if w in self.char_to_id] + [self.eos_id]
        
        # Padding or Truncating to seq_len
        if len(tokens) < seq_len:
            tokens += [self.pad_id] * (seq_len - len(tokens))
        else:
            tokens = tokens[:seq_len]
            
        return np.array(tokens, dtype=np.int32)

    def decode(self, ids):
        if hasattr(ids, 'get'): ids = ids.get()
        # Join with spaces since this is word-level, filtering out specials (indices 0, 1, 2)
        return " ".join([self.id_to_char[int(i)] for i in ids if int(i) > 2])
    
class CharTokenizer:
    def __init__(self, text):
        # 1. Define special tokens first to reserve IDs 0, 1, 2
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>"]
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        
        # 2. Get unique characters from text
        unique_chars = sorted(list(set(text)))
        
        # 3. Build maps including specials
        self.char_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        for i, ch in enumerate(unique_chars):
            self.char_to_id[ch] = i + len(self.special_tokens)
            
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)

    def encode_sentence(self, text, seq_len):
        # Use the actual pad_id instead of hardcoded 0
        indices = [self.char_to_id[ch] for ch in text if ch in self.char_to_id]
        
        if len(indices) < seq_len:
            indices += [self.pad_id] * (seq_len - len(indices))
        else:
            indices = indices[:seq_len]
        return np.array(indices, dtype=np.int32)

    def decode(self, indices):
        if hasattr(indices, 'get'): 
            indices = indices.get()
        
        # Filter out special tokens during decoding for cleaner text
        return "".join([self.id_to_char[int(i)] for i in indices if int(i) > 2])