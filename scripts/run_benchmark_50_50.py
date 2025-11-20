import argparse
import torch
import numpy as np
import os
import time
import pandas as pd
from src.data_loader import UnifiedDataLoader, OmniPreprocess
from src.omni_framework import OmniTransferTrainer
from src.models import TranAD

# ==========================================
# HELPER 1: RAW F1 (Returns Tuple)
# ==========================================
def get_best_f1(scores, labels):
    """
    Calculates standard point-wise F1 score without Point Adjustment.
    Returns: (best_f1, best_threshold)
    """
    # Align lengths
    if len(scores) < len(labels):
        labels = labels[-len(scores):]
    else:
        scores = scores[-len(labels):]
    
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
# HELPER 2: POINT ADJUSTED F1 (Returns Scalar)
# ==========================================
def calc_point_adjusted_f1(scores, labels):
    """
    Calculates F1 using the Point Adjustment protocol.
    If any point in a ground truth anomaly segment is detected, 
    the whole segment is marked correctly detected.
    Returns: best_f1 (float)
    """
    # Align lengths
    if len(scores) < len(labels):
        labels = labels[-len(scores):]
    else:
        scores = scores[-len(labels):]

    best_f1 = 0.0
    
    # Search thresholds (Optimized range)
    thresholds = np.percentile(scores, np.linspace(0, 99.9, 100))
    
    # Optimization: Pre-calculate anomaly segments
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
        
        # Apply Point Adjustment
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets') 
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== OMNITRANSFER: FULL SMD BENCHMARK (50/50 Split) ===")
    print(f"Device: {device}")

    # 1. Define ALL machines
    all_machines = []
    all_machines.extend([f"machine-1-{i}" for i in range(1, 9)])
    all_machines.extend([f"machine-2-{i}" for i in range(1, 10)])
    all_machines.extend([f"machine-3-{i}" for i in range(1, 12)])
    
    # 2. Split 50/50 (Section 5.1 of Paper)
    split_idx = len(all_machines) // 2
    source_machines = all_machines[:split_idx]
    target_machines = all_machines[split_idx:]
    
    print(f"Total Machines: {len(all_machines)}")
    print(f"   > Source Domain (Offline): {len(source_machines)} machines")
    print(f"   > Target Domain (Online):  {len(target_machines)} machines")
    
    loader = UnifiedDataLoader(args.data_path)
    
    # ==========================================
    # PHASE 1: GLOBAL OFFLINE TRAINING
    # ==========================================
    print(f"\n[Phase 1] Processing Source Data (Local Scaling)...")
    
    processed_train_data = []
    all_weights = []
    
    for m in source_machines:
        # Load raw
        raw = loader.load_dataset('SMD', m)
        
        # Per-Entity Scaling: Fit scaler LOCALLY
        temp_prep = OmniPreprocess()
        clean = temp_prep.preprocess(raw) 
        
        processed_train_data.append(clean)
        w = temp_prep.compute_periodic_weights(clean)
        all_weights.append(w)
        
    # Stack for Clustering
    big_data_matrix = np.vstack(processed_train_data)
    global_weights = np.mean(np.array(all_weights), axis=0)
    
    print(f"      Offline Dataset Shape: {big_data_matrix.shape}")
    print(f"      Global Weights Shape: {global_weights.shape}")
    
    print("\n[Phase 1] Training Base Models (W-HAC + TranAD)...")
    trainer = OmniTransferTrainer(TranAD, device=device)
    
    # Train Offline (Using 10 epochs for better stability)
    trainer.train_offline(big_data_matrix, global_weights, n_clusters=5)
    print("      Shape Library Built.")


    # ==========================================
    # PHASE 2: ONLINE EVALUATION LOOP
    # ==========================================
    print(f"\n[Phase 2] Starting Evaluation on Target Domain ({len(target_machines)} machines)...")
    print(f"          Metric: Point Adjusted F1 (PA-F1)")
    
    results = []
    
    for idx, target_entity in enumerate(target_machines):
        print(f"\n--- Target {idx+1}/{len(target_machines)}: {target_entity} ---")
        
        # 1. Load Target Train Data
        raw_train = loader.load_dataset('SMD', target_entity)
        
        # 2. Scale Target (Fit locally)
        # We mimic the scenario where we only see THIS machine's data
        local_prep = OmniPreprocess()
        clean_train = local_prep.preprocess(raw_train)
        
        # 3. Transfer Learning
        t0 = time.time()
        final_model = trainer.online_transfer(clean_train, beta_threshold=0.5)
        init_time = time.time() - t0
        
        # 4. Detection on Test Data
        test_path = os.path.join(args.data_path, 'SMD', 'test', f'{target_entity}.txt')
        label_path = os.path.join(args.data_path, 'SMD', 'test_label', f'{target_entity}.txt')
        
        print(test_path)
        
        if not os.path.exists(test_path):
            print(f"   x Warning: Test file not found. Skipping.")
            continue
            
        raw_test = np.genfromtxt(test_path, delimiter=',')
        labels = np.genfromtxt(label_path, delimiter=',')
        
        # Apply Same Scaler to Test
        clean_test = local_prep.scaler.transform(raw_test)
        
        # Run Model
        scores = trainer.detect(final_model, clean_test)
        
        # 5. Scoring
        raw_f1, _ = get_best_f1(scores, labels)
        pa_f1 = calc_point_adjusted_f1(scores, labels)
        
        print(f"   > Init Time: {init_time:.2f}s | Raw F1: {raw_f1:.4f} | PA F1: {pa_f1:.4f}")
        
        results.append({
            "Machine": target_entity,
            "Raw_F1": raw_f1,
            "PA_F1": pa_f1,
            "InitTime": init_time
        })

    # ==========================================
    # FINAL REPORT
    # ==========================================
    if len(results) > 0:
        df_res = pd.DataFrame(results)
        avg_raw = df_res['Raw_F1'].mean()
        avg_pa = df_res['PA_F1'].mean()
        avg_time = df_res['InitTime'].mean()
        
        print("\n" + "="*60)
        print("OMNITRANSFER FINAL BENCHMARK (Unseen Targets)")
        print("="*60)
        print(f"{'Machine':<15} | {'Raw F1':<10} | {'PA F1 (Paper)':<15} | {'Time (s)':<10}")
        print("-" * 60)
        for index, row in df_res.iterrows():
            print(f"{row['Machine']:<15} | {row['Raw_F1']:<10.4f} | {row['PA_F1']:<15.4f} | {row['InitTime']:<10.4f}")
        print("-" * 60)
        print(f"AVERAGE PA F1 SCORE    : {avg_pa:.4f}")
        print(f"AVERAGE INIT TIME      : {avg_time:.4f} s")
        print("="*60)
        
        df_res.to_csv("results/csv/smd_benchmark_50_50.csv", index=False)
        print("Results saved to smd_benchmark_50_50.csv")
    else:
        print("No results generated (Check data paths).")

if __name__ == "__main__":
    main()