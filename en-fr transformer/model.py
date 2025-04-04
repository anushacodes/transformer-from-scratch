import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in "Attention is All You Need"
    """
    def __init__(self, d_model, num_heads):
        """
        Initialize Multi-Head Attention layer
        
        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Key parameters
        self.d_model = d_model  # Model dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head
        
        # Linear projections for queries, keys, values, and output
        self.W_q = nn.Linear(d_model, d_model)  # Query projection
        self.W_k = nn.Linear(d_model, d_model)  # Key projection
        self.W_v = nn.Linear(d_model, d_model)  # Value projection
        self.W_o = nn.Linear(d_model, d_model)  # Output projection
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention
        
        Args:
            Q (torch.Tensor): Queries [batch_size, num_heads, seq_len, d_k]
            K (torch.Tensor): Keys [batch_size, num_heads, seq_len, d_k]
            V (torch.Tensor): Values [batch_size, num_heads, seq_len, d_k]
            mask (torch.Tensor): Attention mask [batch_size, 1, seq_len, seq_len]
            
        Returns:
            torch.Tensor: Attention output [batch_size, num_heads, seq_len, d_k]
        """
        # Calculate attention scores
        # matmul(Q, K.transpose) / sqrt(d_k)
        batch_size = Q.size(0)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for causal/padding attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)
        
        return output
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k)
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model]
            batch_size (int): Batch size
            
        Returns:
            torch.Tensor: Reshaped tensor [batch_size, num_heads, seq_len, d_k]
        """
        # Reshape from [batch_size, seq_len, d_model] to [batch_size, seq_len, num_heads, d_k]
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        
        # Transpose to [batch_size, num_heads, seq_len, d_k]
        return x.transpose(1, 2)
    
    def combine_heads(self, x, batch_size):
        """
        Combine the heads back into d_model dimension
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, num_heads, seq_len, d_k]
            batch_size (int): Batch size
            
        Returns:
            torch.Tensor: Combined tensor [batch_size, seq_len, d_model]
        """
        # Transpose from [batch_size, num_heads, seq_len, d_k] to [batch_size, seq_len, num_heads, d_k]
        x = x.transpose(1, 2)
        
        # Combine heads: [batch_size, seq_len, d_model]
        return x.contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention
        
        Args:
            query (torch.Tensor): Query tensor [batch_size, seq_len, d_model]
            key (torch.Tensor): Key tensor [batch_size, seq_len, d_model]
            value (torch.Tensor): Value tensor [batch_size, seq_len, d_model]
            mask (torch.Tensor): Attention mask [batch_size, 1, seq_len, seq_len]
            
        Returns:
            torch.Tensor: Multi-head attention output [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        # [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        Q = self.split_heads(self.W_q(query), batch_size)
        K = self.split_heads(self.W_k(key), batch_size)
        V = self.split_heads(self.W_v(value), batch_size)
        
        # Scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output projection
        output = self.W_o(self.combine_heads(attention_output, batch_size))
        
        return output


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff):
        """
        Initialize Position-wise Feed Forward Network
        
        Args:
            d_model (int): Model dimension
            d_ff (int): Feed forward dimension
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.fc2 = nn.Linear(d_ff, d_model)  # Second linear layer
        self.relu = nn.ReLU()  # ReLU activation
        
    def forward(self, x):
        """
        Forward pass for position-wise feed forward network
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, d_model]
        """
        # Apply first linear layer and ReLU
        x = self.relu(self.fc1(x))
        
        # Apply second linear layer
        x = self.fc2(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding as described in "Attention is All You Need"
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    """
    def __init__(self, d_model, max_seq_len):
        """
        Initialize Positional Encoding
        
        Args:
            d_model (int): Model dimension
            max_seq_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        
        # Create position vector [max_seq_len, 1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create division term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Fill positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cos
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x (torch.Tensor): Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Embeddings with positional information
        """
        # Add positional encoding to input and apply dropout
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        
        return x


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Initialize Encoder Layer
        
        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            d_ff (int): Feed forward dimension
            dropout (float): Dropout rate
        """
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Position-wise feed forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass for encoder layer
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model]
            mask (torch.Tensor): Attention mask [batch_size, 1, seq_len, seq_len]
            
        Returns:
            torch.Tensor: Encoder layer output [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm
        
        # Feed forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # Add & Norm
        
        return x


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        """
        Initialize Decoder Layer
        
        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            d_ff (int): Feed forward dimension
            dropout (float): Dropout rate
        """
        super(DecoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Multi-head cross-attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        
        # Position-wise feed forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Forward pass for decoder layer
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model]
            enc_output (torch.Tensor): Encoder output [batch_size, src_seq_len, d_model]
            src_mask (torch.Tensor): Source mask [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask (torch.Tensor): Target mask [batch_size, 1, tgt_seq_len, tgt_seq_len]
            
        Returns:
            torch.Tensor: Decoder layer output [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer normalization
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))  # Add & Norm
        
        # Cross-attention with residual connection and layer normalization
        cross_attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))  # Add & Norm
        
        # Feed forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # Add & Norm
        
        return x


class Transformer(nn.Module):
    """
    Full Transformer Model for sequence-to-sequence tasks
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                           d_model, num_heads, num_encoder_layers,
                           num_decoder_layers, d_ff, max_seq_len, dropout):
        """
        Initialize Transformer
        
        Args:
            src_vocab_size (int): Source vocabulary size
            tgt_vocab_size (int): Target vocabulary size
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            num_encoder_layers (int): Number of encoder layers
            num_decoder_layers (int): Number of decoder layers
            d_ff (int): Feed forward dimension
            max_seq_len (int): Maximum sequence length
            dropout (float): Dropout rate
        """
        super(Transformer, self).__init__()
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        
        # Final linear layer for output
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        
        # Model parameters
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src, tgt):
        """
        Generate source and target masks
        
        Args:
            src (torch.Tensor): Source tensor [batch_size, src_len]
            tgt (torch.Tensor): Target tensor [batch_size, tgt_len]
            
        Returns:
            tuple: (src_mask, tgt_mask) for attention mechanisms
        """
        # Source mask for padding tokens
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
        
        # Target mask for padding tokens
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_len]
        
        # Causal mask for autoregressive decoding
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        causal_mask = causal_mask.to(tgt.device)
        
        # Combine padding mask and causal mask
        tgt_mask = tgt_mask & ~causal_mask.unsqueeze(0).unsqueeze(0) 
        
        return src_mask, tgt_mask
    
    def encode(self, src, src_mask):
        """
        Encode source sequence
        
        Args:
            src (torch.Tensor): Source tensor [batch_size, src_len]
            src_mask (torch.Tensor): Source mask [batch_size, 1, 1, src_len]
            
        Returns:
            torch.Tensor: Encoder output [batch_size, src_len, d_model]
        """
        # Get source embeddings and add positional encoding
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        
        # Pass through encoder layers
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
        return enc_output
     

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        """ 
        Decode target sequence
        
        Args:
            tgt (torch.Tensor): Target tensor [batch_size, tgt_len]
            enc_output (torch.Tensor): Encoder output [batch_size, src_len, d_model]
            src_mask (torch.Tensor): Source mask [batch_size, 1, 1, src_len]
            tgt_mask (torch.Tensor): Target mask [batch_size, 1, tgt_len, tgt_len]
            
        Returns:
            torch.Tensor: Decoder output [batch_size, tgt_len, d_model]
        """
        # Get target embeddings and add positional encoding
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # Pass through decoder layers
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
        return dec_output

    
    def forward(self, src, tgt):
        
        """
        
        Forward pass for transformer
        
        Args:
            src (torch.Tensor): Source tensor [batch_size, src_len]
            tgt (torch.Tensor): Target tensor [batch_size, tgt_len]
            
        Returns:
            torch.Tensor: Output logits [batch_size, tgt_len, tgt_vocab_size]
            
        """
        
        # Generate masks
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Encode source sequence
        enc_output = self.encode(src, src_mask)
        
        # Prepare decoder input (remove last token, typically <eos>)
        decoder_input = tgt[:, :-1]
        # decoder_input shape: [batch_size, tgt_len - 1]

        # Prepare the target mask for the decoder input sequence length
        # We need the top-left (tgt_len - 1) x (tgt_len - 1) part of the full tgt_mask
        decoder_tgt_mask = tgt_mask[:, :, :decoder_input.size(1), :decoder_input.size(1)]
        # decoder_tgt_mask shape: [batch_size, 1, tgt_len - 1, tgt_len - 1]

        # Decode target sequence
        # Note: src_mask remains the same for cross-attention
        dec_output = self.decode(decoder_input, enc_output, src_mask, decoder_tgt_mask)
        # dec_output shape: [batch_size, tgt_len - 1, d_model]

        # Project to vocabulary size
        output = self.final_linear(dec_output)
        
        return output
    

    def greedy_decode(self, src, max_len, start_symbol):
        """
        Greedy decoding for inference
        
        Args:
            src (torch.Tensor): Source tensor [batch_size, src_len]
            max_len (int): Maximum decoding length
            start_symbol (int): Start symbol index
            
        Returns:
            torch.Tensor: Generated sequence [batch_size, max_len]
        """
        batch_size = src.size(0)
        device = src.device
        
        # Create source mask
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Encode source sequence
        enc_output = self.encode(src, src_mask)
        
        # Initialize decoder input with start symbol
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src).to(device)
        
        # Greedy decoding
        for i in range(max_len - 1):
            # Get target mask
            tgt_mask = self.generate_subsequent_mask(ys.size(1)).to(device)
            
            # Decode current sequence
            out = self.decode(ys, enc_output, src_mask, tgt_mask)
            
            # Get probabilities for next token
            prob = self.final_linear(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            
            # Append to the sequence
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
            
        return ys
    
    def generate_subsequent_mask(self, size):
        """
        Generate causal mask for decoding
        
        Args:
            size (int): Sequence length
            
        Returns:
            torch.Tensor: Causal mask [1, size, size]
        """
        # Create upper triangular mask
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        
        # Invert and reshape
        return ~mask.unsqueeze(0).unsqueeze(0)


def create_transformer_model(src_vocab_size, tgt_vocab_size, 
                             d_model, num_heads, num_encoder_layers, 
                             num_decoder_layers, d_ff, max_seq_len, dropout):
    """
    Create a transformer model with the specified parameters
    
    Args:
        src_vocab_size (int): Source vocabulary size
        tgt_vocab_size (int): Target vocabulary size
        device (torch.device): Device to place model on
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        num_encoder_layers (int): Number of encoder layers
        num_decoder_layers (int): Number of decoder layers
        d_ff (int): Feed forward dimension
        dropout (float): Dropout rate
        max_seq_len
        
    Returns:
        Transformer: Initialized transformer model
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len, 
        dropout=dropout
    )
    
    # Initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        
    return model