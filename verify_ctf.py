import sys
import os
# Add current directory to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from src.ctf import CTF_Trainer

def test_ctf():
    print("Testing CTF Implementation...")
    
    # Generate synthetic data
    # [1000, 10]
    print("Generating synthetic data...")
    data = np.random.rand(1000, 10)
    
    # Trainer
    print("Initializing CTF_Trainer...")
    trainer = CTF_Trainer(feat_dim=10, device='cpu')
    
    # Train
    print("Starting Offline Training...")
    periodic_weights = np.ones(10)
    trainer.train_offline(data, periodic_weights, n_clusters=2, epochs=2)
    
    # Detect
    print("Starting Detection...")
    test_data = np.random.rand(200, 10)
    scores = trainer.detect(test_data)
    
    print(f"Scores shape: {scores.shape}")
    if scores.shape[0] == 200:
        print("Success! Scores shape matches input.")
    else:
        print(f"Mismatch! Expected 200, got {scores.shape[0]}")

if __name__ == "__main__":
    test_ctf()
