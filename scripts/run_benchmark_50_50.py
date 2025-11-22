import argparse
import torch
import numpy as np
import os
import time
import pandas as pd
from src.data_loader import ProcessedDataLoader, PeriodicityCalculator
from src.omni_framework import OmniTransferTrainer
from src.models import TranAD

# --- Helpers ---
def get_best_f1(scores, labels):
    if len(scores) < len(labels): labels = labels[-len(scores):]
    else: scores = scores[-len(labels):]
    
    best_f1 = 0
    best_thresh = 0
    thresholds = np.percentile(scores, np.linspace(0, 100, 100))
    
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * (p * r) / (p + r + 1e-8)
        if f1 > best_f1: 
            best_f1 = f1
            best_thresh = thresh
    return best_f1, best_thresh

def calc_point_adjusted_f1(scores, labels):
    if len(scores) < len(labels): labels = labels[-len(scores):]
    else: scores = scores[-len(labels):]

    best_f1 = 0.0
    thresholds = np.percentile(scores, np.linspace(0, 99.9, 100))
    
    actual = (labels == 1)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SMD', help='Dataset name inside processed/ folder')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== OMNITRANSFER: {args.dataset} BENCHMARK (50/50 Split) ===")
    
    # 1. Initialize Loader
    loader = ProcessedDataLoader(processed_dir='processed')
    p_calc = PeriodicityCalculator()
    
    # 2. Get Entities Dynamically
    all_entities = loader.get_available_entities(args.dataset)
    
    if len(all_entities) == 0:
        print(f"Error: No processed data found for '{args.dataset}'.")
        print(f"Did you run: python preprocess.py {args.dataset}")
        return

    # 3. Split 50/50
    split_idx = len(all_entities) // 2
    source_entities = all_entities[:split_idx]
    target_entities = all_entities[split_idx:]
    
    print(f"Total Entities: {len(all_entities)}")
    print(f"   > Source Domain: {len(source_entities)} (Offline)")
    print(f"   > Target Domain: {len(target_entities)} (Online)")
    
    # ==========================================
    # PHASE 1: OFFLINE TRAINING
    # ==========================================
    print(f"\n[Phase 1] Loading Source Data...")
    
    source_data_list = []
    all_weights = []
    
    for entity in source_entities:
        # Load Train only
        data = loader.load(args.dataset, entity, 'train')
        source_data_list.append(data)
        # Compute Weights
        w = p_calc.compute_weights(data)
        all_weights.append(w)
        
    big_data = np.vstack(source_data_list)
    global_weights = np.mean(np.array(all_weights), axis=0)
    
    print(f"      Dataset Shape: {big_data.shape}")
    
    print("\n[Phase 1] Training Base Models...")
    trainer = OmniTransferTrainer(TranAD, device=device)
    trainer.train_offline(big_data, global_weights, n_clusters=5)
    print("      Shape Library Built.")

    # ==========================================
    # PHASE 2: ONLINE EVALUATION
    # ==========================================
    print(f"\n[Phase 2] Evaluating Targets...")
    
    results = []
    
    for idx, entity in enumerate(target_entities):
        print(f"\n--- Target {idx+1}/{len(target_entities)}: {entity} ---")
        
        # Load Train (for Transfer)
        train_data = loader.load(args.dataset, entity, 'train')
        
        # Load Test & Labels (for Evaluation)
        test_data = loader.load(args.dataset, entity, 'test')
        labels = loader.load(args.dataset, entity, 'labels')
        
        # Flatten labels if needed
        if len(labels.shape) > 1: labels = labels[:, 0]
        
        # Transfer
        t0 = time.time()
        final_model = trainer.online_transfer(train_data, beta_threshold=0.5)
        init_time = time.time() - t0
        
        # Detect
        scores = trainer.detect(final_model, test_data)
        
        # Score
        raw_f1, _ = get_best_f1(scores, labels)
        pa_f1 = calc_point_adjusted_f1(scores, labels)
        
        print(f"   > Init Time: {init_time:.2f}s | Raw F1: {raw_f1:.4f} | PA F1: {pa_f1:.4f}")
        
        results.append({
            "Machine": entity,
            "Raw_F1": raw_f1,
            "PA_F1": pa_f1,
            "InitTime": init_time
        })

    # Save Results
    if results:
        df = pd.DataFrame(results)
        out_file = f"results/csv/{args.dataset}_benchmark_50_50.csv"
        df.to_csv(out_file, index=False)
        print(f"\nResults saved to {out_file}")
        print(f"AVG PA F1: {df['PA_F1'].mean():.4f}")

if __name__ == "__main__":
    main()