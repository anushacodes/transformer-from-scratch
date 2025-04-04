import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import TransformerEncoder
from decoder import TransformerDecoder

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_encoder_layers, 
                 num_decoder_layers, max_src_len, max_tgt_len):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            max_len=max_src_len
        )
        
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            max_len=max_tgt_len
        )
        
    def create_masks(self, src, tgt):
        # Source mask is None in this simple implementation
        src_mask = None
        
        # Decoder masks (look-ahead mask for self-attention)
        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device) * float('-inf'), diagonal=1).unsqueeze(0)
        
        # Source-target mask is None in this simple implementation
        src_tgt_mask = None
        
        return src_mask, tgt_mask, src_tgt_mask
        
    def forward(self, src, tgt):
        src_mask, tgt_mask, src_tgt_mask = self.create_masks(src, tgt)
        
        # Pass through encoder
        encoder_output = self.encoder(src)
        
        # Modify encoder output shape for decoder input
        # We need the features before the final projection to vocab_size
        encoder_features = encoder_output
        
        # Pass through decoder
        output = self.decoder(tgt, encoder_features, tgt_mask, src_tgt_mask)
        
        return output

# Test the full transformer
if __name__ == "__main__":
    batch_size = 1
    src_vocab_size = 8
    tgt_vocab_size = 8
    max_src_len = 6
    max_tgt_len = 6
    d_model = 8
    num_heads = 2
    d_ff = 32
    num_encoder_layers = 2
    num_decoder_layers = 2
    
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    ).to(device)
    
    src = torch.randint(0, src_vocab_size, (batch_size, max_src_len)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, max_tgt_len)).to(device)
    
    output = transformer(src, tgt)
    print("Transformer output size:", output.size())  # Expected: [1, 6, 8]
