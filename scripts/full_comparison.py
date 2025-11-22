import argparse
import torch
import numpy as np
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
from src.entropy import EntropicWeightCalculator

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calc_point_adjusted_f1(scores, labels):
    """Point-Adjusted F1 Score"""
    if len(scores) < len(labels): labels = labels[-len(scores):]
    else: scores = scores[-len(labels):]
    
    best_f1 = 0.0
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
        else: i += 1

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

def train_scratch(data, device, epochs=10):
    """Train TranAD from scratch (baseline)"""
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

def detect_scratch(model, test_data, device):
    """Detection using scratch model"""
    feat_dim = test_data.shape[1]
    dummy_whac = WHAC_Clustering(np.ones(feat_dim), window_size=60)
    segments = dummy_whac.segment_data(test_data, stride=1)
    
    model.eval()
    batch_size = 256
    scores = []
    with torch.no_grad():
        for i in range(0, len(segments), batch_size):
            batch_segs = segments[i : i + batch_size]
            tensor_x = torch.Tensor(batch_segs).to(device)
            rec = model(tensor_x)
            loss = torch.mean((tensor_x - rec) ** 2, dim=2)
            batch_scores = loss[:, -1].cpu().numpy()
            scores.extend(batch_scores)
    return np.array(scores)

# ==========================================
# MAIN COMPARISON
# ==========================================
def run_single_dataset(dataset_name, device):
    """Run comparison for a single dataset"""
    print(f"\n{'='*80}")
    print(f"FULL COMPARISON: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Initialize tools
    loader = ProcessedDataLoader(processed_dir='processed')
    p_calc = PeriodicityCalculator()
    e_calc = EntropicWeightCalculator(alpha=2.0)
    c_calc = CausalWeightCalculator(max_lag=2)
    
    # Get entities
    all_entities = loader.get_available_entities(dataset_name)
    if not all_entities:
        print(f"Error: No data found for {dataset_name}")
        return None
    
    # Split 50/50
    split_idx = len(all_entities) // 2
    source_entities = all_entities[:split_idx]
    target_entities = all_entities[split_idx:]
    
    print(f"Total Entities: {len(all_entities)}")
    print(f"  Source: {len(source_entities)} | Target: {len(target_entities)}\n")
    
    # ==========================================
    # CONFIGURATION DEFINITIONS
    # ==========================================
    configs = {
        'TranAD_Scratch': {'type': 'scratch'},
        'Omni_Periodic': {'type': 'omni', 'use_entropy': False, 'use_causal': False},
        'Omni_Entropy': {'type': 'omni', 'use_entropy': True, 'use_causal': False},
        'Omni_Causal': {'type': 'omni', 'use_entropy': False, 'use_causal': True},
        'Omni_Entropy_Causal': {'type': 'omni', 'use_entropy': True, 'use_causal': True},
    }
    
    all_results = []
    
    for config_name, config in configs.items():
        print(f"\n{'='*80}")
        print(f"[{config_name}]")
        print(f"{'='*80}")
        
        if config['type'] == 'scratch':
            # BASELINE: TranAD from scratch
            for idx, entity in enumerate(target_entities):
                print(f"  {idx+1}/{len(target_entities)}: {entity}...", end=' ')
                
                train = loader.load(dataset_name, entity, 'train')
                test = loader.load(dataset_name, entity, 'test')
                labels = loader.load(dataset_name, entity, 'labels')
                if len(labels.shape) > 1: labels = labels[:, 0]
                
                t0 = time.time()
                model = train_scratch(train, device, epochs=10)
                train_time = time.time() - t0
                
                scores = detect_scratch(model, test, device)
                f1 = calc_point_adjusted_f1(scores, labels)
                
                print(f"Time: {train_time:.2f}s | F1: {f1:.4f}")
                
                all_results.append({
                    'Dataset': dataset_name,
                    'Config': config_name,
                    'Machine': entity,
                    'TrainTime': train_time,
                    'F1': f1
                })
                
        else:
            # OMNITRANSFER VARIANTS
            use_entropy = config['use_entropy']
            use_causal = config['use_causal']
            
            # Build offline shape library
            print(f"  Building Shape Library...")
            source_list = []
            weights_list = []
            
            for entity in source_entities:
                data = loader.load(dataset_name, entity, 'train')
                source_list.append(data)
                
                # Base weight
                if use_entropy:
                    w_base = e_calc.compute_entropic_weights(data)
                else:
                    w_base = p_calc.compute_weights(data)
                
                # Causal boost
                if use_causal:
                    w_c = c_calc.compute_causal_weights(data)
                    w_final = w_base * (1.0 + w_c)
                else:
                    w_final = w_base
                    
                weights_list.append(w_final)
            
            big_data = np.vstack(source_list)
            global_weights = np.mean(np.array(weights_list), axis=0)
            
            trainer = OmniTransferTrainer(TranAD, device=device)
            trainer.train_offline(big_data, global_weights, n_clusters=None)
            print(f"  Shape Library Ready.\n")
            
            # Evaluate on targets
            for idx, entity in enumerate(target_entities):
                print(f"  {idx+1}/{len(target_entities)}: {entity}...", end=' ')
                
                train = loader.load(dataset_name, entity, 'train')
                test = loader.load(dataset_name, entity, 'test')
                labels = loader.load(dataset_name, entity, 'labels')
                if len(labels.shape) > 1: labels = labels[:, 0]
                
                t0 = time.time()
                model = trainer.online_transfer(train)
                train_time = time.time() - t0
                
                scores = trainer.detect(model, test)
                f1 = calc_point_adjusted_f1(scores, labels)
                
                print(f"Time: {train_time:.2f}s | F1: {f1:.4f}")
                
                all_results.append({
                    'Dataset': dataset_name,
                    'Config': config_name,
                    'Machine': entity,
                    'TrainTime': train_time,
                    'F1': f1
                })
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Full OmniTransfer Comparison Suite')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, 
                       help='Dataset names (e.g., SMD MSL SWaT)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    all_datasets_results = []
    
    # Run comparison for each dataset
    for dataset in args.datasets:
        results = run_single_dataset(dataset, device)
        if results:
            all_datasets_results.extend(results)
    
    if not all_datasets_results:
        print("No results generated!")
        return
    
    # ==========================================
    # SAVE RESULTS
    # ==========================================
    df = pd.DataFrame(all_datasets_results)
    
    # Save combined results
    datasets_str = '_'.join(args.datasets)
    csv_path = f"results/csv/{datasets_str}_full_comparison.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"COMBINED RESULTS SAVED: {csv_path}")
    print(f"{'='*80}\n")
    
    # ==========================================
    # SUMMARY TABLES
    # ==========================================
    
    # Overall summary (across all datasets)
    print("\n" + "="*80)
    print("OVERALL SUMMARY (All Datasets)")
    print("="*80)
    overall_summary = df.groupby('Config').agg({
        'F1': 'mean',
        'TrainTime': 'mean'
    }).round(4)
    overall_summary.columns = ['Avg_F1', 'Avg_Time']
    overall_summary = overall_summary.reset_index()
    print(overall_summary.to_string(index=False))
    print("="*80 + "\n")
    
    # Per-dataset summary
    print("\n" + "="*80)
    print("PER-DATASET SUMMARY")
    print("="*80)
    per_dataset_summary = df.groupby(['Dataset', 'Config']).agg({
        'F1': 'mean',
        'TrainTime': 'mean'
    }).round(4)
    per_dataset_summary.columns = ['Avg_F1', 'Avg_Time']
    print(per_dataset_summary.to_string())
    print("="*80 + "\n")
    
    # Save summaries
    overall_summary_path = f"results/csv/{datasets_str}_overall_summary.csv"
    overall_summary.to_csv(overall_summary_path, index=False)
    
    per_dataset_summary_path = f"results/csv/{datasets_str}_per_dataset_summary.csv"
    per_dataset_summary.to_csv(per_dataset_summary_path)  # Keep index
    
    print(f"Overall summary saved: {overall_summary_path}")
    print(f"Per-dataset summary saved: {per_dataset_summary_path}\n")

if __name__ == "__main__":
    main()
