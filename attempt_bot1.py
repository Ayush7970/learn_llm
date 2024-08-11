import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='This is a demonstration program')
parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch size')
args = parser.parse_args()

# Set device based on availability
device = 'mps' if torch.cuda.is_available() else 'cpu'
batch_size = int(args.batch_size)
block_size = 128
max_steps = 200
learning_rate = 3e-4
eval_interval = 100
embedding_dim = 384
num_heads = 1
num_layers = 1
dropout_rate = 0.2

print(device)

# Load vocabulary
with open("openwebtext/vocab.txt", 'r', encoding='utf-8') as f:
    text_data = f.read()
    unique_chars = sorted(list(set(text_data)))

vocab_size = len(unique_chars)
char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
idx_to_char = {i: ch for i, ch in enumerate(unique_chars)}
encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l])

class SelfAttentionHead(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        attention_weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        attention_weights = attention_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        v = self.value(x)
        out = attention_weights @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Simple linear layers followed by a non-linearity."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attention_out = self.attention(x)
        x = self.norm1(x + attention_out)
        feed_forward_out = self.feed_forward(x)
        x = self.norm2(x + feed_forward_out)
        return x

class GPTLanguageModel(nn.Module):
    """Language model based on GPT architecture."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)])
        self.norm_f = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_indices, targets=None):
        B, T = input_indices.shape
        token_embeddings = self.token_embeddings(input_indices)
        position_embeddings = self.position_embeddings(torch.arange(T, device=device))
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.norm_f(x)
        logits = self.output_layer(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, input_indices, max_new_tokens):
        for _ in range(max_new_tokens):
            input_cond = input_indices[:, -block_size:]
            logits, _ = self.forward(input_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probs, num_samples=1)
            input_indices = torch.cat((input_indices, next_index), dim=1)
        return input_indices

# Load the model
model = GPTLanguageModel(vocab_size)
print('Loading model parameters...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('Model loaded successfully!')
model = model.to(device)

# Interactive prompt
while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_text}')
