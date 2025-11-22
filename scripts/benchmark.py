import argparse
import torch
import numpy as np
import os
import time
import pandas as pd
from src.data_loader import ProcessedDataLoader, PeriodicityCalculator
from src.omni_framework import OmniTransferTrainer
from src.models import TranAD
from src.causality import CausalWeightCalculator

# ==========================================
# HELPER 1: RAW F1 (Strict Point-wise)
# ==========================================
def get_best_f1(scores, labels):
    """
    Calculates standard point-wise F1 score.
    Returns: (best_f1, best_threshold)
    """
    # Align lengths
    if len(scores) < len(labels): labels = labels[-len(scores):]
    else: scores = scores[-len(labels):]
    
    best_f1 = 0
    best_thresh = 0
    
    # Percentile search for speed and robustness
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

# ==========================================
# HELPER 2: POINT ADJUSTED F1 (Paper Standard)
# ==========================================
def calc_point_adjusted_f1(scores, labels):
    """
    Calculates F1 using the Point Adjustment protocol.
    If any point in a ground truth anomaly segment is detected, 
    the whole segment is marked correctly detected.
    """
    if len(scores) < len(labels): labels = labels[-len(scores):]
    else: scores = scores[-len(labels):]

    best_f1 = 0.0
    # Optimized search range for anomalies (usually high scores)
    thresholds = np.percentile(scores, np.linspace(0, 99.9, 100))
    
    # Pre-calculate anomaly segments for speed
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
            
            if f1 > best_f1:
                best_f1 = f1
            
    return best_f1

# ==========================================
# MAIN BENCHMARK LOGIC
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SMD', help='Dataset name (folder in processed/)')
    parser.add_argument('--use_causal', action='store_true', help='Enable Tigramite Causal Weighting')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== OMNITRANSFER: {args.dataset} BENCHMARK (50/50 Split) ===")
    if args.use_causal:
        print("!!! CAUSAL WEIGHTING ENABLED (Physics-Aware Clustering) !!!")
    
    # 1. Initialize Components
    loader = ProcessedDataLoader(processed_dir='processed')
    p_calc = PeriodicityCalculator()
    c_calc = CausalWeightCalculator(max_lag=2)
    
    # 2. Load Entities
    all_entities = loader.get_available_entities(args.dataset)
    if not all_entities:
        print(f"Error: No data found for '{args.dataset}'. Did you run preprocess.py?")
        return

    # 3. Split 50/50 (Source vs Target)
    split_idx = len(all_entities) // 2
    source_entities = all_entities[:split_idx]
    target_entities = all_entities[split_idx:]
    
    print(f"Total Entities: {len(all_entities)}")
    print(f"   > Source Domain (Offline Training): {len(source_entities)}")
    print(f"   > Target Domain (Online Evaluation): {len(target_entities)}")
    
    # ==========================================
    # PHASE 1: OFFLINE TRAINING
    # ==========================================
    print(f"\n[Phase 1] Loading Source Data & Computing Weights...")
    
    source_data_list = []
    final_weights_list = []
    
    for i, entity in enumerate(source_entities):
        # Load Train Data
        data = loader.load(args.dataset, entity, 'train')
        source_data_list.append(data)
        
        # 1. Calculate Periodic Weights (Standard)
        w_p = p_calc.compute_weights(data)
        
        # 2. Calculate Causal Weights (Optional)
        if args.use_causal:
            print(f"   > Analyzing Causality: {entity}...", end='\r')
            w_c = c_calc.compute_causal_weights(data)
            # Hybrid Formula: Boost periodic metrics if they are Drivers
            w_final = w_p * (1.0 + w_c)
        else:
            w_final = w_p
            
        final_weights_list.append(w_final)
        
    if args.use_causal: print(f"   > Causal Analysis Complete.          ")

    # Stack Data
    big_data = np.vstack(source_data_list)
    global_weights = np.mean(np.array(final_weights_list), axis=0)
    
    print(f"      Dataset Shape: {big_data.shape}")
    
    print("\n[Phase 1] Training Base Models (W-HAC + TranAD)...")
    trainer = OmniTransferTrainer(TranAD, device=device)
    # Train offline (Auto-calculates thresholds internally)
    trainer.train_offline(big_data, global_weights, n_clusters=5)
    print("      Shape Library Built.")

    # ==========================================
    # PHASE 2: ONLINE EVALUATION LOOP
    # ==========================================
    print(f"\n[Phase 2] Evaluating Target Domain...")
    
    results = []
    
    for idx, entity in enumerate(target_entities):
        print(f"--- Target {idx+1}/{len(target_entities)}: {entity} ---")
        
        # Load Data
        train_data = loader.load(args.dataset, entity, 'train')
        test_data = loader.load(args.dataset, entity, 'test')
        labels = loader.load(args.dataset, entity, 'labels')
        
        # Handle label shape (Time, 1) -> (Time,)
        if len(labels.shape) > 1: labels = labels[:, 0]
        
        # 1. Transfer Learning
        t0 = time.time()
        # Pass NO beta_threshold to use the Auto-Calculated Beta from Phase 1
        final_model = trainer.online_transfer(train_data)
        init_time = time.time() - t0
        
        # 2. Detection
        scores = trainer.detect(final_model, test_data)
        
        # 3. Scoring
        raw_f1, _ = get_best_f1(scores, labels)
        pa_f1 = calc_point_adjusted_f1(scores, labels)
        
        print(f"   > Init Time: {init_time:.2f}s | Raw F1: {raw_f1:.4f} | PA F1: {pa_f1:.4f}")
        
        results.append({
            "Machine": entity,
            "Raw_F1": raw_f1,
            "PA_F1": pa_f1,
            "InitTime": init_time
        })

    # ==========================================
    # FINAL REPORT
    # ==========================================
    if results:
        df_res = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print(f"FINAL RESULTS: {args.dataset}")
        print("="*60)
        print(f"AVERAGE PA F1 SCORE    : {df_res['PA_F1'].mean():.4f}")
        print(f"AVERAGE INIT TIME      : {df_res['InitTime'].mean():.4f} s")
        print("-" * 60)
        
        # Dynamic filename based on settings
        suffix = "_causal" if args.use_causal else ""
        out_file = f"results/csv/{args.dataset}_benchmark{suffix}.csv"
        
        df_res.to_csv(out_file, index=False)
        print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()