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

# ... (Keep calc_point_adjusted_f1 helper from previous scripts) ...
def calc_point_adjusted_f1(scores, labels):
    if len(scores) < len(labels): labels = labels[-len(scores):]
    else: scores = scores[-len(labels):]
    best_f1 = 0.0
    thresholds = np.percentile(scores, np.linspace(0, 99.9, 50))
    actual = (labels == 1)
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        adjusted_preds = np.array(preds)
        i = 0
        while i < len(labels):
            if actual[i]:
                j = i
                while j < len(labels) and actual[j]: j += 1
                if np.sum(preds[i:j]) > 0: adjusted_preds[i:j] = 1
                i = j
            else: i += 1
        tp = np.sum((adjusted_preds == 1) & (labels == 1))
        fp = np.sum((adjusted_preds == 1) & (labels == 0))
        fn = np.sum((adjusted_preds == 0) & (labels == 1))
        if tp > 0:
            p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
            f1 = 2 * (p * r) / (p + r + 1e-8)
            if f1 > best_f1: best_f1 = f1
    return best_f1

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
    parser.add_argument('--dataset', type=str, default='SMD')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== HEAD-TO-HEAD: {args.dataset} (Omni vs Scratch) ===")
    
    loader = ProcessedDataLoader(processed_dir='processed')
    p_calc = PeriodicityCalculator()
    
    all_entities = loader.get_available_entities(args.dataset)
    if not all_entities:
        print("No data found.")
        return

    split_idx = len(all_entities) // 2
    source_entities = all_entities[:split_idx]
    target_entities = all_entities[split_idx:]
    
    # Phase 1: Offline
    print("\n[Phase 1] Building Shape Library...")
    source_list = []
    weights_list = []
    for e in source_entities:
        d = loader.load(args.dataset, e, 'train')
        source_list.append(d)
        weights_list.append(p_calc.compute_weights(d))
    
    omni_trainer = OmniTransferTrainer(TranAD, device=device)
    omni_trainer.train_offline(np.vstack(source_list), np.mean(weights_list, axis=0), n_clusters=5)
    
    # Phase 2: Compare
    results = []
    for idx, e in enumerate(target_entities):
        print(f"--- Comparing {e} ---")
        train = loader.load(args.dataset, e, 'train')
        test = loader.load(args.dataset, e, 'test')
        labels = loader.load(args.dataset, e, 'labels')
        if len(labels.shape) > 1: labels = labels[:, 0]
        
        # Omni
        t0 = time.time()
        model_omni = omni_trainer.online_transfer(train, beta_threshold=0.5)
        t_omni = time.time() - t0
        s_omni = omni_trainer.detect(model_omni, test)
        f1_omni = calc_point_adjusted_f1(s_omni, labels)
        
        # Scratch
        t0 = time.time()
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
        
    if results:
        df = pd.DataFrame(results)
        df.to_csv(f"results/csv/{args.dataset}_comparison.csv", index=False)
        print(f"Saved to results/csv/{args.dataset}_comparison.csv")

if __name__ == "__main__":
    main()