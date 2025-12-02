import argparse
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import ProcessedDataLoader
from src.clustering import WHAC_Clustering
from src.models import RNN_VAE

def diagnose(dataset_name='CTF', limit=5):
    print(f"DIAGNOSING DATASET: {dataset_name}")
    loader = ProcessedDataLoader(processed_dir='processed')
    entities = loader.get_available_entities(dataset_name)
    
    if not entities:
        print("No entities found.")
        return

    print(f"Found {len(entities)} entities. Inspecting first {limit}...")
    
    for i, entity in enumerate(entities[:limit]):
        print(f"\n--- Entity: {entity} ---")
        train = loader.load(dataset_name, entity, 'train')
        test = loader.load(dataset_name, entity, 'test')
        
        print(f"Train Shape: {train.shape}")
        print(f"Test Shape: {test.shape}")
        
        # Check normalization
        print(f"Train Range: [{train.min():.4f}, {train.max():.4f}]")
        print(f"Test Range: [{test.min():.4f}, {test.max():.4f}]")
        
        # Check feature variance
        train_std = train.std(axis=0)
        low_var_indices = np.where(train_std < 0.01)[0]
        print(f"Features with low variance (<0.01): {len(low_var_indices)}/{train.shape[1]}")
        if len(low_var_indices) > 0:
            print(f"Indices: {low_var_indices}")
            print(f"Values (first 5 rows): \n{train[:5, low_var_indices]}")
            
        # Check Clustering
        print("Running Clustering Check...")
        whac = WHAC_Clustering(np.ones(train.shape[1]), window_size=60)
        segments = whac.segment_data(train)
        
        # Check segment statistics
        seg_std = segments.std(axis=1).mean(axis=0)
        print(f"Avg Segment Std Dev: {seg_std.mean():.4f}")
        
        # Try finding k
        # Use a small subset for speed
        subset = segments[:1000]
        stat_features = whac._extract_statistical_features(subset)
        
        # Check feature scaling in clustering
        print(f"Stat Features Shape: {stat_features.shape}")
        print(f"Stat Features Std: {stat_features.std(axis=0).mean():.4f}")
        
        # Quick Model Test
        print("Running Model Test (RNN_VAE)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = RNN_VAE(feat_dim=train.shape[1], hidden_dim=64, latent_dim=10).to(device)
        
        # Forward pass on a batch
        batch = torch.DoubleTensor(segments[:32]).to(device) # [32, 60, feats]
        model.double()
        
        recon, mu, logvar = model(batch)
        
        mse = torch.mean((batch - recon)**2).item()
        print(f"Initial MSE Loss (Untrained): {mse:.6f}")
        
        # Check if reconstruction is just copying input (Teacher Forcing leak?)
        # If untrained model has very low MSE, it might be just copying x
        # But untrained weights are random, so it shouldn't copy perfectly unless skip connection exists.
        # RNN_VAE doesn't have skip connection.
        
        print("-" * 30)

if __name__ == "__main__":
    diagnose()
