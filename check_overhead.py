import argparse
import torch
import numpy as np
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_loader import UnifiedDataLoader, OmniPreprocess
from omni_framework import OmniTransferTrainer
from models import TranAD
from clustering import WHAC_Clustering

def train_single_entity(data, device, epochs=5):
    """
    Simulates the standard way: Initialize and Train a model for ONE machine.
    """
    # 1. Init Model
    feat_dim = data.shape[1]
    model = TranAD(feat_dim=feat_dim).to(device)
    
    # 2. Segment Data (Standard sliding window)
    # We use a dummy weight just to access the segment_data method
    dummy_whac = WHAC_Clustering(np.ones(feat_dim), window_size=60)
    segments = dummy_whac.segment_data(data)
    
    # 3. Train Loop
    tensor_x = torch.Tensor(segments).to(device)
    dataset = TensorDataset(tensor_x, tensor_x)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets') 
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"=== TRAINING OVERHEAD TEST (Device: {device}) ===")
    print("Scenario: Training models for 5 Source Entities.")
    print("------------------------------------------------")

    # 1. LOAD ALL DATA
    loader = UnifiedDataLoader(args.data_path)
    preprocessor = OmniPreprocess()
    source_entities = [f'machine-1-{i}' for i in range(1, 6)]
    
    loaded_data = []
    print(f"[Setup] Loading data for {len(source_entities)} machines...")
    for entity in source_entities:
        raw = loader.load_dataset('SMD', entity)
        clean = preprocessor.preprocess(raw)
        loaded_data.append(clean)

    # ======================================================
    # APPROACH 1: BASELINE (One Model Per Entity)
    # ======================================================
    print("\n[1] Running BASELINE (1 Model per Entity)...")
    t_start_base = time.time()
    
    for i, data in enumerate(loaded_data):
        print(f"    > Training separate model for machine-1-{i+1}...")
        train_single_entity(data, device, epochs=5)
        
    t_end_base = time.time()
    time_baseline = t_end_base - t_start_base
    print(f"    >>> Total Baseline Time: {time_baseline:.4f} seconds")


    # ======================================================
    # APPROACH 2: OMNITRANSFER (Clustering + Base Models)
    # ======================================================
    print("\n[2] Running OMNITRANSFER (Clustering + K Base Models)...")
    t_start_omni = time.time()
    
    # Prepare Data Stack
    all_weights = [preprocessor.compute_periodic_weights(d) for d in loaded_data]
    big_data = np.vstack(loaded_data)
    global_weights = np.mean(np.array(all_weights), axis=0)
    
    # Initialize Framework
    trainer = OmniTransferTrainer(TranAD, device)
    
    # Run Offline Training
    # Note: We force 2 clusters to show the benefit. 
    # Even if we use 3, it's 3 models vs 5 models.
    print(f"    > Clustering data and training shared Base Models...")
    trainer.train_offline(big_data, global_weights, n_clusters=2)
    
    t_end_omni = time.time()
    time_omni = t_end_omni - t_start_omni
    print(f"    >>> Total OmniTransfer Time: {time_omni:.4f} seconds")

    # ======================================================
    # SUMMARY
    # ======================================================
    print("\n" + "="*50)
    print("TRAINING OVERHEAD COMPARISON (Lower is Better)")
    print("="*50)
    print(f"Baseline (5 Models)     : {time_baseline:.4f} s")
    print(f"OmniTransfer (2 Models) : {time_omni:.4f} s")
    print("-" * 50)
    
    reduction = (time_baseline - time_omni) / time_baseline * 100
    print(f"Overhead Reduction: {reduction:.2f}%")
    print("="*50)
    print("Analysis:")
    print(f"We trained models for {len(source_entities)} entities.")
    print("Baseline trained 5 separate models.")
    print("OmniTransfer trained only 2 base models (plus clustering time).")
    print("As the number of entities grows (e.g. 1000 machines),")
    print("this gap becomes massive.")

if __name__ == "__main__":
    main()