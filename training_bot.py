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
context_size = 128
max_iterations = 200
learning_rate = 3e-4
evaluation_steps = 100
embedding_size = 384
num_heads = 1
num_layers = 1
dropout_rate = 0.2

print(device)

# Load vocabulary
with open("openwebtext/vocab.txt", 'r', encoding='utf-8') as f:
    text_data = f.read()
    unique_chars = sorted(list(set(text_data)))

vocab_size = len(unique_chars)
char_to_index = {ch: i for i, ch in enumerate(unique_chars)}
index_to_char = {i: ch for i, ch in enumerate(unique_chars)}
encode = lambda s: [char_to_index[c] for c in s]
decode = lambda l: ''.join([index_to_char[i] for i in l])

# Load a random chunk of data
def load_random_chunk(data_split):
    filename = "openwebtext/train_split.txt" if data_split == 'train' else "openwebtext/val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, file_size - context_size * batch_size)
            mm.seek(start_pos)
            chunk = mm.read(context_size * batch_size - 1)
            decoded_chunk = chunk.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_chunk), dtype=torch.long)
    return data

# Get a batch of data
def get_batch(data_split):
    data = load_random_chunk(data_split)
    indices = torch.randint(len(data) - context_size, (batch_size,))
    x_batch = torch.stack([data[i:i + context_size] for i in indices])
    y_batch = torch.stack([data[i + 1:i + context_size + 1] for i in indices])
    return x_batch.to(device), y_batch.to(device)

@torch.no_grad()
def evaluate_loss():
    results = {}
    model.eval()
    for data_split in ['train', 'val']:
        batch_losses = torch.zeros(evaluation_steps)
        for i in range(evaluation_steps):
            x_batch, y_batch = get_batch(data_split)
            logits, loss = model(x_batch, y_batch)
            batch_losses[i] = loss.item()
        results[data_split] = batch_losses.mean()
    model.train()
    return results

class AttentionHead(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_dim, bias=False)
        self.query = nn.Linear(embedding_size, head_dim, bias=False)
        self.value = nn.Linear(embedding_size, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(head_dim * num_heads, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
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
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        head_dim = embedding_dim // num_heads
        self.self_attention = MultiHeadSelfAttention(num_heads, head_dim)
        self.feed_forward = FeedForward(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attention_out = self.self_attention(x)
        x = self.norm1(x + attention_out)
        feed_forward_out = self.feed_forward(x)
        x = self.norm2(x + feed_forward_out)
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(context_size, embedding_size)
        self.blocks = nn.Sequential(*[TransformerBlock(embedding_size, num_heads=num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embedding_size)
        self.output_layer = nn.Linear(embedding_size, vocab_size)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_indices, target_indices=None):
        B, T = input_indices.shape
        token_emb = self.token_embedding(input_indices)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.output_layer(x)

        if target_indices is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target_indices = target_indices.view(B * T)
            loss = F.cross_entropy(logits, target_indices)

        return logits, loss

    def generate(self, input_indices, num_new_tokens):
        for _ in range(num_new_tokens):
            logits, _ = self.forward(input_indices)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probs, num_samples=1)
            input_indices = torch.cat((input_indices, next_index), dim=1)
        return input_indices

model = GPTModel(vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iteration in range(max_iterations):
    print(iteration)
    if iteration % evaluation_steps == 0:
        losses = evaluate_loss()
        print(f"Step: {iteration}, Train Loss: {losses['train']:.3f}, Val Loss: {losses['val']:.3f}")

    x_batch, y_batch = get_batch('train')
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Model saved')
