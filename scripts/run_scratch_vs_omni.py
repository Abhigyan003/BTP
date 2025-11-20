import argparse
import torch
import numpy as np
import os
import time
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.data_loader import UnifiedDataLoader, OmniPreprocess
from src.omni_framework import OmniTransferTrainer
from src.models import TranAD
from src.clustering import WHAC_Clustering

# ==========================================
# HELPER: POINT ADJUSTED F1
# ==========================================
def calc_point_adjusted_f1(scores, labels):
    if len(scores) < len(labels): labels = labels[-len(scores):]
    else: scores = scores[-len(labels):]
    
    best_f1 = 0.0
    thresholds = np.percentile(scores, np.linspace(50, 99.9, 50))
    
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        adjusted_preds = np.array(preds)
        actual = (labels == 1)
        
        i = 0
        while i < len(labels):
            if actual[i]:
                j = i
                while j < len(labels) and actual[j]: j += 1
                if np.sum(preds[i:j]) > 0:
                    adjusted_preds[i:j] = 1
                i = j
            else:
                i += 1
                
        tp = np.sum((adjusted_preds == 1) & (labels == 1))
        fp = np.sum((adjusted_preds == 1) & (labels == 0))
        fn = np.sum((adjusted_preds == 0) & (labels == 1))
        
        if tp > 0:
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1 = 2 * (p * r) / (p + r + 1e-8)
            if f1 > best_f1: best_f1 = f1
    return best_f1

# ==========================================
# HELPER: TRAIN SCRATCH
# ==========================================
def train_scratch(data, device, epochs=10):
    feat_dim = data.shape[1]
    model = TranAD(feat_dim=feat_dim).to(device)
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
    parser.add_argument('--data_path', type=str, default='datasets') 
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"=== TARGET DOMAIN COMPARISON: OmniTransfer vs Scratch ===")
    
    # 1. Define Machines & Split
    all_machines = []
    all_machines.extend([f"machine-1-{i}" for i in range(1, 9)])
    all_machines.extend([f"machine-2-{i}" for i in range(1, 10)])
    all_machines.extend([f"machine-3-{i}" for i in range(1, 12)])
    
    split_idx = len(all_machines) // 2
    source_machines = all_machines[:split_idx]
    target_machines = all_machines[split_idx:] # The 14 Unseen Machines
    
    loader = UnifiedDataLoader(args.data_path)
    
    # ==========================================
    # PHASE 1: BUILD OFFLINE SHAPE LIBRARY
    # ==========================================
    print(f"\n[Phase 1] Training Shape Library on Source Domain ({len(source_machines)} machines)...")
    processed_source = []
    all_weights = []
    
    for m in source_machines:
        raw = loader.load_dataset('SMD', m)
        prep = OmniPreprocess()
        clean = prep.preprocess(raw)
        processed_source.append(clean)
        all_weights.append(prep.compute_periodic_weights(clean))
        
    big_data = np.vstack(processed_source)
    g_weights = np.mean(np.array(all_weights), axis=0)
    
    omni_trainer = OmniTransferTrainer(TranAD, device=device)
    omni_trainer.train_offline(big_data, g_weights, n_clusters=5)
    print("      Shape Library Ready.")

    # ==========================================
    # PHASE 2: COMPARISON ON TARGETS
    # ==========================================
    print(f"\n[Phase 2] Running Comparison on {len(target_machines)} Target Machines...")
    results = []
    
    for idx, m in enumerate(target_machines):
        print(f"--- Machine {m} ---")
        
        # Load & Scale Locally
        raw_train = loader.load_dataset('SMD', m)
        local_prep = OmniPreprocess()
        clean_train = local_prep.preprocess(raw_train)
        
        # Test Data
        test_path = os.path.join(args.data_path, 'SMD', 'test', f'{m}.txt')
        label_path = os.path.join(args.data_path, 'SMD', 'test_label', f'{m}.txt')
        if not os.path.exists(test_path): continue
        
        raw_test = np.genfromtxt(test_path, delimiter=',')
        labels = np.genfromtxt(label_path, delimiter=',')
        clean_test = local_prep.scaler.transform(raw_test)
        
        # 1. OMNITRANSFER
        t0 = time.time()
        model_omni = omni_trainer.online_transfer(clean_train, beta_threshold=0.5)
        t_omni = time.time() - t0
        scores_omni = omni_trainer.detect(model_omni, clean_test)
        f1_omni = calc_point_adjusted_f1(scores_omni, labels)
        
        # 2. SCRATCH
        t0 = time.time()
        model_scratch = train_scratch(clean_train, device, epochs=10)
        t_scratch = time.time() - t0
        scores_scratch = omni_trainer.detect(model_scratch, clean_test)
        f1_scratch = calc_point_adjusted_f1(scores_scratch, labels)
        
        print(f"   > Time: {t_omni:.2f}s vs {t_scratch:.2f}s | F1: {f1_omni:.4f} vs {f1_scratch:.4f}")
        
        results.append({
            "Machine": m,
            "Omni_F1": f1_omni,
            "Scratch_F1": f1_scratch,
            "Omni_Time": t_omni,
            "Scratch_Time": t_scratch
        })

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv("results/csv/target_comparison.csv", index=False)
    print("\nSaved results to target_comparison.csv")

if __name__ == "__main__":
    main()