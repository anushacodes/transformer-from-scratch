import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

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

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1))
        dk = q.size(-1)
        scores = scores / torch.sqrt(torch.tensor(float(dk), device=device))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.wo(attn_output)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, tgt_mask=None, src_tgt_mask=None):
        # Self-attention with masking for target sequence
        masked_attn_out = self.masked_attn(x, x, x, tgt_mask)
        x = self.norm1(x + masked_attn_out)
        
        # Cross-attention between decoder and encoder
        cross_attn_out = self.cross_attn(x, encoder_output, encoder_output, src_tgt_mask)
        x = self.norm2(x + cross_attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super().__init__()
        self.embedding = PosEmbedding(vocab_size, d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, tgt_mask=None, src_tgt_mask=None):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_tgt_mask)
            
        x = self.fc_out(x)
        return x

# Reusing PosEmbedding from encoder.py
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

# Test the decoder
if __name__ == "__main__":
    batch_size = 1
    vocab_size = 8
    max_len = 6
    d_model = 8
    num_heads = 2
    d_ff = 32
    num_layers = 2
    
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len
    ).to(device)
    
    tgt = torch.randint(0, vocab_size, (batch_size, max_len)).to(device)
    encoder_output = torch.rand(batch_size, max_len, d_model).to(device)
    
    # Create look-ahead mask for decoder self-attention
    size = tgt.size(1)
    tgt_mask = torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1).unsqueeze(0)
    
    output = decoder(tgt, encoder_output, tgt_mask)
    print("Decoder output size:", output.size())  # Expected: [1, 6, 8]
