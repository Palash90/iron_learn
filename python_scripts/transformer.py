import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Settings ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 64   
batch_size = 64   
n_embd = 128
n_head = 4
n_layer = 4
learning_rate = 3e-4
max_iters = 100000
eval_interval = 500

# --- 1. Data Handling ---
with open('/home/palash/Downloads/hasan-etal-2020-low/2.75M/original_corpus.bn', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    ds = train_data if split == 'train' else val_data
    ix = torch.randint(len(ds) - block_size, (batch_size,))
    x = torch.stack([ds[i:i+block_size] for i in ix])
    y = torch.stack([ds[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- 2. Model Architecture ---
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        T = x.shape[1]
        # FIX: The missing causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        
        attn_out, _ = self.sa(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x

class BengaliGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

model = BengaliGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- 3. Training & Real-time Generation ---
print(f"Training on {device} with vocab size {vocab_size}...")

for iter in range(max_iters):
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0:
        print(f"\nStep {iter} | Loss: {loss.item():.4f}")
        # TRIGGER GENERATION
        seed_text = "বইটি পড়ার সময়"
        context = torch.tensor([encode(seed_text)], dtype=torch.long, device=device)
        generated = model.generate(context, max_tokens=60)
        print(f"Output: {decode(generated[0].tolist())}")
        print("-" * 30)