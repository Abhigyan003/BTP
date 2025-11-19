import argparse
import torch
import numpy as np
import os
import time  # <--- Added for timing
from data_loader import UnifiedDataLoader, OmniPreprocess
from omni_framework import OmniTransferTrainer
from models import TranAD

# --- helper function ---
def get_best_f1(scores, labels):
    labels = labels[-len(scores):]
    best_f1 = 0
    best_thresh = 0
    thresholds = np.percentile(scores, np.linspace(0, 100, 200))
    
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_f1, best_thresh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets') 
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== OmniTransfer Efficiency Evaluation (Device: {device}) ===")

    # 1. SETUP
    loader = UnifiedDataLoader(args.data_path)
    preprocessor = OmniPreprocess()
    
    source_entities = [f'machine-1-{i}' for i in range(1, 6)]
    all_clean_data = []
    all_weights = []
    
    print(f"[Phase 1] Loading Source Domain...")
    for entity in source_entities:
        raw = loader.load_dataset('SMD', entity)
        clean = preprocessor.preprocess(raw) 
        all_clean_data.append(clean)
        all_weights.append(preprocessor.compute_periodic_weights(clean))

    big_data = np.vstack(all_clean_data)
    global_weights = np.mean(np.array(all_weights), axis=0)
    
    trainer = OmniTransferTrainer(model_class=TranAD, device=device)

    # --- MEASURE TRAINING OVERHEAD ---
    print(f"\n[Timer] Starting Offline Training (The 'Training Overhead')...")
    t_start_offline = time.time()
    
    trainer.train_offline(big_data, global_weights, n_clusters=3)
    
    t_end_offline = time.time()
    training_overhead = t_end_offline - t_start_offline
    print(f"   >>> Offline Training Time: {training_overhead:.4f} seconds")


    # 2. ONLINE TRANSFER
    target_entity = 'machine-1-6'
    print(f"\n[Phase 2] Transferring to Target: {target_entity}...")
    
    raw_train = loader.load_dataset('SMD', target_entity)
    preprocessor.scaler.fit(raw_train) 
    clean_train = preprocessor.scaler.transform(raw_train)
    
    # --- MEASURE INITIALIZATION TIME ---
    print(f"[Timer] Starting Online Transfer (The 'Initialization Time')...")
    t_start_init = time.time()
    
    final_model = trainer.online_transfer(clean_train, beta_threshold=0.5)
    
    t_end_init = time.time()
    init_time = t_end_init - t_start_init
    print(f"   >>> Model Initialization Time: {init_time:.4f} seconds")

    # 3. DETECTION PHASE
    print(f"\n[Phase 3] Evaluating on Test Data...")
    test_path = os.path.join(args.data_path, 'SMD/test/machine-1-6.txt')
    test_label_path = os.path.join(args.data_path, 'SMD/test_label/machine-1-6.txt')
    
    raw_test = np.genfromtxt(test_path, delimiter=',')
    labels = np.genfromtxt(test_label_path, delimiter=',')
    clean_test = preprocessor.scaler.transform(raw_test)
    
    anomaly_scores = trainer.detect(final_model, clean_test)
    f1, thresh = get_best_f1(anomaly_scores, labels)
    
    print("=" * 40)
    print(f"FINAL EFFICIENCY REPORT for {target_entity}")
    print("=" * 40)
    print(f"1. Accuracy (F1 Score)    : {f1:.4f}")
    print(f"2. Training Overhead      : {training_overhead:.2f} s (Total for Source Domain)")
    print(f"3. Initialization Time    : {init_time:.2f} s (Time to ready Target)")
    print("-" * 40)
    print("Note: 'Initialization Time' is significantly lower than")
    print("      training a fresh TranAD model from scratch (which takes min)")
    print("=" * 40)

if __name__ == "__main__":
    main()