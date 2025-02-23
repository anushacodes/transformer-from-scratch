import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Parameters
batch_size = 1
vocab_size = 8  # Vocabulary size
max_len = 6     # Sequence length
d_model = 8     # Embedding size
num_heads = 2
d_ff = 32       # FFN expansion

torch.manual_seed(132)

# PosEmbedding (unchanged)
class PosEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = self.pos_encod(max_len, d_model)
        self.max_len = max_len

    def pos_encod(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model, device=device)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                den = 10000 ** (i / d_model)
                pe[pos, i] = torch.sin(torch.tensor(pos, dtype=torch.float, device=device) / den)
                if i + 1 < d_model:
                    pe[pos, i + 1] = torch.cos(torch.tensor(pos, dtype=torch.float, device=device) / den)
        return pe.unsqueeze(0)

    def forward(self, x):
        token_embeddings = self.token_embed(x)
        return token_embeddings + self.pos_embed[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1))
        dk = q.size(-1)
        scores = scores / torch.sqrt(torch.tensor(float(dk), device=device))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        attn_output = self.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.wo(attn_output)

# TransformerEncoderLayer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# TransformerEncoder (Pipeline)
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super().__init__()
        self.embedding = PosEmbedding(vocab_size, d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        return x

# Test Pipeline
model = TransformerEncoder(
    vocab_size=8,
    d_model=8,
    num_heads=2,
    d_ff=32,
    num_layers=2,
    max_len=6
).to(device)

x = torch.randint(0, 8, (1, 6)).to(device)  # Random token indices [1, 6]
output = model(x)
print("Encoder output size:", output.size())  # Expected: [1, 6, 8]