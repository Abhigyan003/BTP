"""
TranAD Utility Functions

Helper functions for using TranAD-specific features:
- Data preparation (windowing, format conversion)
- Time-dependent loss function
- Training utilities
"""

import torch
import torch.nn as nn
import numpy as np


def convert_to_windows(data, window_size=10, stride=1):
    """
    Convert time series data to sliding windows
    
    Args:
        data: numpy array or tensor [timesteps, features]
        window_size: size of sliding window
        stride: step size for sliding
        
    Returns:
        windows: tensor [num_windows, window_size, features]
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    
    return torch.stack(windows) if windows else torch.empty(0, window_size, data.shape[1])


def prepare_tranad_batch(batch):
    """
    Convert batch_first format to TranAD's expected format
    
    Args:
        batch: [batch_size, window_size, features]
        
    Returns:
        src: [window_size, batch_size, features]
        tgt: [1, batch_size, features] - last element
    """
    # Permute to [window_size, batch_size, features]
    src = batch.permute(1, 0, 2)
    # Extract last timestep as target
    tgt = src[-1:, :, :]
    return src, tgt


def tranad_loss(model, batch, epoch, criterion=None):
    """
    Compute TranAD's time-dependent loss
    
    Loss weights shift from Phase 1 to Phase 2 over epochs:
    - Early training: Focus on Phase 1 (baseline reconstruction)
    - Later training: Focus on Phase 2 (anomaly-aware reconstruction)
    
    Args:
        model: TranAD model
        batch: Input batch [batch_size, window_size, features]
        epoch: Current epoch number (1-indexed)
        criterion: Loss function (default: MSELoss)
        
    Returns:
        loss: Combined weighted loss
    """
    if criterion is None:
        criterion = nn.MSELoss()
    
    # Prepare batch in TranAD format
    src, tgt = prepare_tranad_batch(batch)
    
    # Forward pass through TranAD
    x1, x2 = model.forward_tranad(src, tgt)
    
    # Time-dependent loss weighting
    n = epoch + 1
    l1 = criterion(x1, tgt)
    l2 = criterion(x2, tgt)
    
    # Early epochs: weight = 1/n (larger), later: weight = 1-1/n (grows)
    loss = (1.0 / n) * l1 + (1.0 - 1.0 / n) * l2
    
    return loss


def train_tranad(model, data_loader, epochs=10, device='cpu', lr=1e-3, verbose=True):
    """
    Train TranAD model with proper time-dependent loss
    
    Args:
        model: TranAD model instance
        data_loader: DataLoader with batches [B, W, F]
        epochs: Number of training epochs
        device: 'cpu' or 'cuda'
        lr: Learning rate
        verbose: Print training progress
        
    Returns:
        model: Trained model
        losses: List of epoch losses
    """
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch, _ in data_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            loss = tranad_loss(model, batch, epoch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if verbose:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
    
    return model, losses


def detect_anomalies_tranad(model, test_data, window_size=10, device='cpu'):
    """
    Detect anomalies using TranAD
    
    Args:
        model: Trained TranAD model
        test_data: Test data [timesteps, features]
        window_size: Window size used during training
        device: 'cpu' or 'cuda'
        
    Returns:
        anomaly_scores: Reconstruction error for each timestep
    """
    model = model.to(device)
    model.eval()
    
    # Create windows
    windows = convert_to_windows(test_data, window_size, stride=1)
    
    scores = []
    with torch.no_grad():
        for window in windows:
            window = window.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get TranAD format
            src, tgt = prepare_tranad_batch(window)
            
            # Forward pass - use Phase 2 output (anomaly-aware)
            _, x2 = model.forward_tranad(src, tgt)
            
            # Compute reconstruction error
            error = torch.mean((tgt - x2) ** 2).item()
            scores.append(error)
    
    return np.array(scores)
