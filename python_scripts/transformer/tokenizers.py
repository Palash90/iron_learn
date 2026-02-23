import time

import cupy as np
import numpy as onp
from transformer import Transformer
from simple_layers import Softmax, CrossEntropyLoss

try:
    np.cuda.Device(0).use()
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
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return np.array([self.char_to_int[ch] for ch in text], dtype=np.int32)

    def decode(self, indices):
        if hasattr(indices, 'get'): indices = indices.get()
        return "".join([self.int_to_char[int(i)] for i in indices])
