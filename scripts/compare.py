import argparse
import torch
import numpy as np
import os
import time
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.data_loader import ProcessedDataLoader, PeriodicityCalculator
from src.omni_framework import OmniTransferTrainer
from src.models import TranAD
from src.clustering import WHAC_Clustering
from src.causality import CausalWeightCalculator

# ==========================================
# HELPERS
# ==========================================
def calc_point_adjusted_f1(scores, labels):
    """
    Calculates F1 using the Point Adjustment protocol.
    """
    if len(scores) < len(labels): labels = labels[-len(scores):]
    else: scores = scores[-len(labels):]
    
    best_f1 = 0.0
    # Optimized search
    thresholds = np.percentile(scores, np.linspace(0, 99.9, 50))
    
    actual = (labels == 1)
    
    # Pre-calculate anomaly segments
    anomaly_segments = []
    i = 0
    while i < len(labels):
        if actual[i]:
            j = i
            while j < len(labels) and actual[j]: j += 1
            anomaly_segments.append((i, j))
            i = j
        else:
            i += 1

    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        adjusted_preds = np.array(preds)
        
        # Vectorized PA application
        for (start, end) in anomaly_segments:
            if np.sum(preds[start:end]) > 0:
                adjusted_preds[start:end] = 1
                
        tp = np.sum((adjusted_preds == 1) & (labels == 1))
        fp = np.sum((adjusted_preds == 1) & (labels == 0))
        fn = np.sum((adjusted_preds == 0) & (labels == 1))
        
        if tp > 0:
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1 = 2 * (p * r) / (p + r + 1e-8)
            if f1 > best_f1: best_f1 = f1
            
    return best_f1

def train_scratch(data, device, epochs=10):
    """
    Standard training loop (Baseline)
    """
    feat_dim = data.shape[1]
    model = TranAD(feat_dim=feat_dim).to(device)
    
    # Use WHAC purely for segmentation logic
    dummy_whac = WHAC_Clustering(np.ones(feat_dim), window_size=60)
    segments = dummy_whac.segment_data(data)
    
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
    parser.add_argument('--dataset', type=str, default='SMD', help='Dataset name (folder in processed/)')
    parser.add_argument('--use_causal', action='store_true', help='Enable Tigramite Causal Weighting')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== HEAD-TO-HEAD: {args.dataset} (Omni vs Scratch) ===")
    if args.use_causal:
        print("!!! CAUSAL WEIGHTING ENABLED !!!")
    
    # 1. Initialize Tools
    loader = ProcessedDataLoader(processed_dir='processed')
    p_calc = PeriodicityCalculator()
    c_calc = CausalWeightCalculator(max_lag=2)
    
    # 2. Get Entities
    all_entities = loader.get_available_entities(args.dataset)
    if not all_entities:
        print(f"No data found for {args.dataset}. Check 'processed' folder.")
        return

    # 3. Split 50/50
    split_idx = len(all_entities) // 2
    source_entities = all_entities[:split_idx]
    target_entities = all_entities[split_idx:]
    
    print(f"Total Entities: {len(all_entities)}")
    print(f"   > Source Domain: {len(source_entities)} (Used for Shape Library)")
    print(f"   > Target Domain: {len(target_entities)} (Used for Comparison)")
    
    # ==========================================
    # PHASE 1: BUILD OFFLINE SHAPE LIBRARY
    # ==========================================
    print(f"\n[Phase 1] Building Shape Library...")
    
    source_list = []
    final_weights_list = []
    
    for i, e in enumerate(source_entities):
        # Load Train
        d = loader.load(args.dataset, e, 'train')
        source_list.append(d)
        
        # 1. Periodic Weights
        w_p = p_calc.compute_weights(d)
        
        # 2. Causal Weights (Optional)
        if args.use_causal:
            print(f"   > Causal Analysis: {e}...", end='\r')
            w_c = c_calc.compute_causal_weights(d)
            # Hybrid Weighting
            w_final = w_p * (1.0 + w_c)
        else:
            w_final = w_p
            
        final_weights_list.append(w_final)
    
    if args.use_causal: print("   > Causal Analysis Complete.          ")

    # Train Base Models
    big_data = np.vstack(source_list)
    global_weights = np.mean(np.array(final_weights_list), axis=0)
    
    omni_trainer = OmniTransferTrainer(TranAD, device=device)
    # Train offline (Using 10 epochs for robustness)
    omni_trainer.train_offline(big_data, global_weights, n_clusters=5)
    print("      Shape Library Ready.")
    
    # ==========================================
    # PHASE 2: COMPARISON ON TARGETS
    # ==========================================
    print(f"\n[Phase 2] Running Head-to-Head Comparison...")
    results = []
    
    for idx, e in enumerate(target_entities):
        print(f"--- Comparing {e} ({idx+1}/{len(target_entities)}) ---")
        
        # Load Data
        train = loader.load(args.dataset, e, 'train')
        test = loader.load(args.dataset, e, 'test')
        labels = loader.load(args.dataset, e, 'labels')
        
        # Fix label shape if needed
        if len(labels.shape) > 1: labels = labels[:, 0]
        
        # -----------------------------------
        # METHOD A: OMNITRANSFER
        # -----------------------------------
        t0 = time.time()
        # No threshold passed -> Uses Auto-Calculated Beta
        model_omni = omni_trainer.online_transfer(train) 
        t_omni = time.time() - t0
        
        s_omni = omni_trainer.detect(model_omni, test)
        f1_omni = calc_point_adjusted_f1(s_omni, labels)
        
        # -----------------------------------
        # METHOD B: SCRATCH (BASELINE)
        # -----------------------------------
        t0 = time.time()
        # Train Scratch (10 epochs for fairness)
        model_scratch = train_scratch(train, device, epochs=10)
        t_scratch = time.time() - t0
        
        s_scratch = omni_trainer.detect(model_scratch, test)
        f1_scratch = calc_point_adjusted_f1(s_scratch, labels)
        
        print(f"   > Omni    | Time: {t_omni:.2f}s | F1: {f1_omni:.4f}")
        print(f"   > Scratch | Time: {t_scratch:.2f}s | F1: {f1_scratch:.4f}")
        
        results.append({
            "Machine": e,
            "Omni_F1": f1_omni, "Scratch_F1": f1_scratch,
            "Omni_Time": t_omni, "Scratch_Time": t_scratch
        })
        
    # Save Results
    if results:
        df = pd.DataFrame(results)
        
        suffix = "_causal" if args.use_causal else ""
        out_file = f"results/csv/{args.dataset}_comparison{suffix}.csv"
        
        df.to_csv(out_file, index=False)
        print(f"\nSaved to {out_file}")
        
        avg_omni = df['Omni_F1'].mean()
        avg_scratch = df['Scratch_F1'].mean()
        print(f"AVG OMNI F1: {avg_omni:.4f}")
        print(f"AVG SCRATCH F1: {avg_scratch:.4f}")

if __name__ == "__main__":
    main()