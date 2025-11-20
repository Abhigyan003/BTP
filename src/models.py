# models.py (Updated)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TranAD(nn.Module):
    def __init__(self, feat_dim, d_model=64, n_heads=4, n_layers=2, window_size=60):
        super(TranAD, self).__init__()
        self.window_size = window_size
        self.feat_dim = feat_dim
        self.d_model = d_model
        
        # Input Embeddings
        self.pos_encoder = PositionalEncoding(d_model)
        self.token_embedding = nn.Linear(feat_dim, d_model)
        
        # Encoder (Captures local trend)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        # Decoder (Captures global dependency)
        decoder_layers = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=d_model*2, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layers, n_layers)
        
        # Final Projection
        self.projection = nn.Linear(d_model, feat_dim)
        
        # Attention mask for the window (Standard Triangle mask for autoregression)
        # In TranAD anomaly detection, we often allow full attention, 
        # but strict TranAD uses masking. We keep it None for basic reconstruction.
        self.src_mask = None

    def forward(self, src):
        # src shape: [Batch, Window, Feats]
        
        # 1. Embedding
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # 2. Encoder Phase
        memory = self.encoder(x)
        
        # 3. Decoder Phase 
        # (In TranAD Phase 1, we try to reconstruct the input using the memory)
        x_out = self.decoder(x, memory)
        
        # 4. Projection
        reconstruction = self.projection(x_out)
        
        return reconstruction