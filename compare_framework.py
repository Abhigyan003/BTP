import argparse
import torch
import numpy as np
import os
import time
import copy
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_loader import UnifiedDataLoader, OmniPreprocess
from omni_framework import OmniTransferTrainer
from models import TranAD

# --- Helper for Evaluation ---
def get_best_f1(scores, labels):
    labels = labels[-len(scores):]
    best_f1 = 0
    thresholds = np.percentile(scores, np.linspace(0, 100, 50)) # Faster search
    
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * (p * r) / (p + r + 1e-8)
        if f1 > best_f1: best_f1 = f1
    return best_f1

# --- Helper for Training from Scratch ---
def train_from_scratch(model, data, device, epochs=10):
    """
    Standard training loop (No Transfer, No Clustering)
    We give it MORE epochs (10) than Transfer (5) because 
    models from scratch need more time to converge.
    """
    tensor_x = torch.Tensor(data).to(device)
    dataset = TensorDataset(tensor_x, tensor_x)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets') 
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"=== COMPARISON: OmniTransfer vs. Training from Scratch ===")
    
    # 1. DATA LOADING
    loader = UnifiedDataLoader(args.data_path)
    preprocessor = OmniPreprocess()
    target_entity = 'machine-1-6'
    
    # Load Target Data
    print(f"[Setup] Loading Data for {target_entity}...")
    raw_train = loader.load_dataset('SMD', target_entity)
    
    # Fit scaler on target train (Standard practice for Scratch)
    preprocessor.scaler.fit(raw_train)
    clean_train = preprocessor.scaler.transform(raw_train)
    
    # Load Test Data
    test_path = os.path.join(args.data_path, 'SMD/test/machine-1-6.txt')
    test_label_path = os.path.join(args.data_path, 'SMD/test_label/machine-1-6.txt')
    raw_test = np.genfromtxt(test_path, delimiter=',')
    labels = np.genfromtxt(test_label_path, delimiter=',')
    clean_test = preprocessor.scaler.transform(raw_test)

    # ======================================================
    # EXPERIMENT 1: WITHOUT FRAMEWORK (From Scratch)
    # ======================================================
    print("\n" + "="*50)
    print("RUNNING: Baseline (No Framework / Train from Scratch)")
    print("="*50)
    
    # Initialize fresh model
    feat_dim = clean_train.shape[1]
    model_scratch = TranAD(feat_dim=feat_dim).to(device)
    
    # Measure Time
    start_time = time.time()
    
    # Segmentation needed for Transformer input
    # We use the whac helper just for segmentation logic, not clustering
    temp_trainer = OmniTransferTrainer(TranAD, device)
    temp_trainer.whac = temp_trainer.whac or type('obj', (object,), {'segment_data': lambda x, stride=None: x}) # dummy
    from clustering import WHAC_Clustering
    cluster_tool = WHAC_Clustering(np.ones(feat_dim)) # Dummy weights
    train_segments = cluster_tool.segment_data(clean_train)
    
    # Train
    # Note: We train for 10 epochs to give it a fair chance.
    train_from_scratch(model_scratch, train_segments, device, epochs=10)
    
    time_scratch = time.time() - start_time
    print(f"   > Training Time: {time_scratch:.4f} seconds")
    
    # Measure Accuracy
    scores_scratch = temp_trainer.detect(model_scratch, clean_test)
    f1_scratch = get_best_f1(scores_scratch, labels)
    print(f"   > F1 Score: {f1_scratch:.4f}")


    # ======================================================
    # EXPERIMENT 2: WITH OMNITRANSFER
    # ======================================================
    print("\n" + "="*50)
    print("RUNNING: OmniTransfer (Pre-trained + Fine-tuning)")
    print("="*50)
    
    # 1. PRE-REQUISITE: We need the Source Domain trained (Offline)
    # In a real scenario, this is already done. We do it quickly here.
    print("   [Background] Pre-loading Source Domain (Offline Phase)...")
    source_entities = [f'machine-1-{i}' for i in range(1, 6)]
    all_clean = []
    all_weights = []
    for e in source_entities:
        r = loader.load_dataset('SMD', e)
        c = preprocessor.preprocess(r)
        all_clean.append(c)
        all_weights.append(preprocessor.compute_periodic_weights(c))
    
    big_data = np.vstack(all_clean)
    g_weights = np.mean(np.array(all_weights), axis=0)
    
    omni_trainer = OmniTransferTrainer(TranAD, device)
    omni_trainer.train_offline(big_data, g_weights, n_clusters=3)
    print("   [Background] Offline Phase Complete.")
    
    # 2. MEASURE INITIALIZATION TIME (Online Phase)
    start_time = time.time()
    
    # Transfer (Fine-tune for only 5 epochs as per paper benefit)
    model_omni = omni_trainer.online_transfer(clean_train, beta_threshold=0.5)
    
    time_omni = time.time() - start_time
    print(f"   > Initialization Time: {time_omni:.4f} seconds")
    
    # Measure Accuracy
    scores_omni = omni_trainer.detect(model_omni, clean_test)
    f1_omni = get_best_f1(scores_omni, labels)
    print(f"   > F1 Score: {f1_omni:.4f}")


    # ======================================================
    # FINAL COMPARISON
    # ======================================================
    print("\n\n")
    print("="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    print(f"{'Metric':<25} | {'No Framework':<15} | {'OmniTransfer':<15}")
    print("-" * 60)
    print(f"{'Initialization Time (s)':<25} | {time_scratch:<15.4f} | {time_omni:<15.4f}")
    print(f"{'Accuracy (F1 Score)':<25} | {f1_scratch:<15.4f} | {f1_omni:<15.4f}")
    print("-" * 60)
    
    speedup = (time_scratch - time_omni) / time_scratch * 100
    print(f"CONCLUSION: OmniTransfer was {speedup:.2f}% faster to initialize.")
    if f1_omni >= f1_scratch:
        print("            And it achieved EQUAL or BETTER accuracy.")
    else:
        print("            With a slight trade-off in accuracy.")

if __name__ == "__main__":
    main()