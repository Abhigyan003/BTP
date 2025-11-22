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
from src.entropy import EntropicWeightCalculator # <--- NEW IMPORT

# ... (Keep helper functions get_best_f1 and calc_point_adjusted_f1 same as before) ...
# (Copy them from previous version if needed, omitted here to save space)
def get_best_f1(scores, labels):
    if len(scores) < len(labels): labels = labels[-len(scores):]
    else: scores = scores[-len(labels):]
    best_f1 = 0; best_thresh = 0
    thresholds = np.percentile(scores, np.linspace(0, 100, 100))
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8); f1 = 2*p*r/(p+r+1e-8)
        if f1 > best_f1: best_f1 = f1; best_thresh = thresh
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
        else: i += 1
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        adjusted_preds = np.array(preds)
        for (start, end) in anomaly_segments:
            if np.sum(preds[start:end]) > 0: adjusted_preds[start:end] = 1
        tp = np.sum((adjusted_preds == 1) & (labels == 1))
        fp = np.sum((adjusted_preds == 1) & (labels == 0))
        fn = np.sum((adjusted_preds == 0) & (labels == 1))
        if tp > 0:
            p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8); f1 = 2*p*r/(p+r+1e-8)
            if f1 > best_f1: best_f1 = f1
    return best_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SMD', help='Dataset name')
    parser.add_argument('--use_causal', action='store_true', help='Enable Tigramite Causal Weighting')
    parser.add_argument('--use_entropy', action='store_true', help='Use Entropic Weighting (replaces Periodicity)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== OMNITRANSFER: {args.dataset} BENCHMARK ===")
    
    # Logic Configuration
    if args.use_entropy:
        print("!!! USING ENTROPIC WEIGHTING (Structure-Based) !!!")
    else:
        print("... Using Periodic Weighting (Cycle-Based)")
        
    if args.use_causal:
        print("!!! CAUSAL WEIGHTING ENABLED !!!")
    
    loader = ProcessedDataLoader(processed_dir='processed')
    
    # Initialize Calculators
    p_calc = PeriodicityCalculator()
    e_calc = EntropicWeightCalculator(alpha=2.0) # alpha=2 punishes noise harder
    c_calc = CausalWeightCalculator(max_lag=2)
    
    all_entities = loader.get_available_entities(args.dataset)
    if not all_entities: return

    split_idx = len(all_entities) // 2
    source_entities = all_entities[:split_idx]
    target_entities = all_entities[split_idx:]
    
    # ==========================================
    # PHASE 1: OFFLINE
    # ==========================================
    print(f"\n[Phase 1] Computing Weights...")
    source_list = []
    weights_list = []
    
    for entity in source_entities:
        data = loader.load(args.dataset, entity, 'train')
        source_list.append(data)
        
        # 1. Base Weight (Entropy OR Periodicity)
        if args.use_entropy:
            # Use Entropy (Good for Steps/Ramps)
            w_base = e_calc.compute_entropic_weights(data)
        else:
            # Use Periodicity (Good for Sine Waves)
            w_base = p_calc.compute_weights(data)
            
        # 2. Causal Boost (Optional)
        if args.use_causal:
            print(f"   > Analyzing Causality: {entity}...", end='\r')
            w_c = c_calc.compute_causal_weights(data)
            w_final = w_base * (1.0 + w_c)
        else:
            w_final = w_base
            
        weights_list.append(w_final)
        
    if args.use_causal: print("   > Causal Analysis Complete.          ")

    big_data = np.vstack(source_list)
    global_weights = np.mean(np.array(weights_list), axis=0)
    
    print(f"      Weight Vector Sample: {global_weights[:5]}")
    
    trainer = OmniTransferTrainer(TranAD, device=device)
    # Auto-cluster (n_clusters=None) is best with Entropy as it creates sharper clusters
    trainer.train_offline(big_data, global_weights, n_clusters=None) 
    
    # ==========================================
    # PHASE 2: ONLINE
    # ==========================================
    print(f"\n[Phase 2] Evaluation...")
    results = []
    
    for idx, entity in enumerate(target_entities):
        print(f"--- Target {idx+1}/{len(target_entities)}: {entity} ---")
        train = loader.load(args.dataset, entity, 'train')
        test = loader.load(args.dataset, entity, 'test')
        labels = loader.load(args.dataset, entity, 'labels')
        if len(labels.shape) > 1: labels = labels[:, 0]
        
        t0 = time.time()
        final_model = trainer.online_transfer(train)
        init_time = time.time() - t0
        
        scores = trainer.detect(final_model, test)
        raw_f1, _ = get_best_f1(scores, labels)
        pa_f1 = calc_point_adjusted_f1(scores, labels)
        
        print(f"   > Init Time: {init_time:.2f}s | Raw F1: {raw_f1:.4f} | PA F1: {pa_f1:.4f}")
        results.append({
            "Machine": entity, "Raw_F1": raw_f1, "PA_F1": pa_f1, "InitTime": init_time
        })

    # Save
    if results:
        df = pd.DataFrame(results)
        # Dynamic Filename
        tag = "_entropy" if args.use_entropy else "_periodic"
        if args.use_causal: tag += "_causal"
        
        out_file = f"results/csv/{args.dataset}_benchmark{tag}.csv"
        df.to_csv(out_file, index=False)
        print(f"\nResults saved to {out_file}")
        print(f"AVG PA F1: {df['PA_F1'].mean():.4f}")

if __name__ == "__main__":
    main()