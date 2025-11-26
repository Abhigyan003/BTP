# models.py - TranAD Implementation (VLDB 2022)
# Deep Transformer Networks for Anomaly Detection in Multivariate Time Series
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math

class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)


class TranAD(nn.Module):
    """
    TranAD: Transformer-based Anomaly Detection
    
    Key Features:
    - Self-conditioning mechanism using anomaly scores
    - Dual decoders for adversarial training
    - Two-phase training: Phase 1 (baseline) + Phase 2 (anomaly-aware)
    
    Args:
        feats: Number of features in the multivariate time series
        lr: Learning rate (default: 1e-3)
        batch: Batch size (default: 128)
        n_window: Window size for time series (default: 10)
    """
    def __init__(self, feats=None, feat_dim=None, lr=1e-3, batch=128, n_window=10):
        super(TranAD, self).__init__()
        
        # Accept both 'feats' and 'feat_dim' for backward compatibility
        if feats is None and feat_dim is None:
            raise ValueError("Either 'feats' or 'feat_dim' must be provided")
        if feats is None:
            feats = feat_dim
        
        self.name = 'TranAD'
        self.lr = lr
        self.batch = batch
        self.n_feats = feats
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        
        # Positional Encoding - input dimension is 2*feats for concatenation
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        
        # Transformer Encoder - single shared encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * feats, 
            nhead=feats, 
            dim_feedforward=16, 
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        
        # Transformer Decoder 1 - for Phase 1 (without anomaly scores)
        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * feats, 
            nhead=feats, 
            dim_feedforward=16, 
            dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        
        # Transformer Decoder 2 - for Phase 2 (with anomaly scores)
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * feats, 
            nhead=feats, 
            dim_feedforward=16, 
            dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        
        # Final projection layer
        self.fcn = nn.Sequential(
            nn.Linear(2 * feats, feats), 
            nn.Sigmoid()
        )
    
    @property
    def encoder(self):
        """Backward compatibility alias for transformer_encoder"""
        return self.transformer_encoder
    
    @property
    def decoder(self):
        """Backward compatibility alias for transformer_decoder2 (best decoder)"""
        return self.transformer_decoder2

    def encode(self, src, c, tgt):
        """
        Encoding with context concatenation
        
        Args:
            src: Source sequence [window_size, batch_size, feats]
            c: Context (anomaly scores or zeros) [window_size, batch_size, feats]
            tgt: Target element [1, batch_size, feats]
            
        Returns:
            tgt: Repeated target [1, batch_size, 2*feats]
            memory: Encoded memory from transformer encoder
        """
        # Concatenate source with context (normal features + anomaly awareness)
        src = torch.cat((src, c), dim=2)
        
        # Scale by sqrt(n_feats) as in standard transformer
        src = src * math.sqrt(self.n_feats)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Encode through transformer encoder
        memory = self.transformer_encoder(src)
        
        # Repeat target to match doubled dimension
        tgt = tgt.repeat(1, 1, 2)
        
        return tgt, memory

    def forward_tranad(self, src, tgt):
        """
        Original TranAD forward pass with two-phase reconstruction
        
        Args:
            src: Source window [window_size, batch_size, feats]
            tgt: Target element (last element of window) [1, batch_size, feats]
            
        Returns:
            x1: Phase 1 reconstruction (without anomaly awareness)
            x2: Phase 2 reconstruction (with anomaly awareness)
        """
        # Phase 1 - Without anomaly scores (baseline reconstruction)
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        
        # Phase 2 - With anomaly scores (anomaly-aware reconstruction)
        # Use squared reconstruction error from Phase 1 as context
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        
        return x1, x2

    def forward(self, src, tgt=None):
        """
        Backward-compatible forward method that auto-detects calling convention
        
        Usage:
            # Old style (backward compatible):
            output = model(batch)  # batch: [batch_size, window_size, feats]
            
            # New TranAD style:
            x1, x2 = model(src, tgt)  # src: [window_size, batch_size, feats]
                                       # tgt: [1, batch_size, feats]
        
        Args:
            src: Either batch [B, W, F] (old) or window [W, B, F] (new)
            tgt: Optional target [1, B, F] for new TranAD style
            
        Returns:
            - If tgt is None: Single reconstruction tensor (backward compatible)
            - If tgt provided: Tuple (x1, x2) for TranAD training
        """
        if tgt is None:
            # Old calling convention: model(batch)
            # Assume batch_first format: [batch_size, window_size, feats]
            batch = src
            batch_size, input_window_size, feats = batch.shape
            
            # TranAD works with windows of size n_window
            # If input is larger, take only the last n_window timesteps
            if input_window_size > self.n_window:
                batch = batch[:, -self.n_window:, :]
                actual_window_size = self.n_window
            else:
                actual_window_size = input_window_size
            
            # Convert to TranAD format: [window_size, batch_size, feats]
            src_converted = batch.permute(1, 0, 2)
            
            # Extract target (last timestep): [1, batch_size, feats]
            tgt_converted = src_converted[-1:, :, :]
            
            # Run TranAD forward pass
            x1, x2 = self.forward_tranad(src_converted, tgt_converted)
            
            # Return only x2 (best reconstruction) in batch_first format
            # x2 is [1, batch_size, feats], expand to match input window size
            # Repeat the reconstruction across the original window for compatibility
            reconstruction = x2.permute(1, 0, 2).repeat(1, input_window_size, 1)
            
            return reconstruction
        else:
            # New TranAD calling convention: model(src, tgt)
            # Assume correct format already provided
            return self.forward_tranad(src, tgt)


# Legacy/Alternative version without self-conditioning (for comparison)
class TranAD_Basic(nn.Module):
    """Basic TranAD without self-conditioning - single decoder only"""
    def __init__(self, feats, lr=1e-3, batch=128, n_window=10):
        super(TranAD_Basic, self).__init__()
        self.name = 'TranAD_Basic'
        self.lr = lr
        self.batch = batch
        self.n_feats = feats
        self.n_window = n_window
        self.n = self.n_feats * self.n_window
        
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(
            d_model=feats, 
            nhead=feats, 
            dim_feedforward=16, 
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers = TransformerDecoderLayer(
            d_model=feats, 
            nhead=feats, 
            dim_feedforward=16, 
            dropout=0.1
        )
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.fcn = nn.Sigmoid()

    def forward(self, src, tgt):
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        x = self.transformer_decoder(tgt, memory)
        x = self.fcn(x)
        return x