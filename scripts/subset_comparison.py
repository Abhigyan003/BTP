import argparse
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import time
import pandas as pd
import copy
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.data_loader import ProcessedDataLoader, PeriodicityCalculator
from src.omni_framework import OmniTransferTrainer
from src.models import TranAD, RNN_VAE
from src.clustering import WHAC_Clustering
from src.causality import CausalWeightCalculator
from src.entropy import EntropicWeightCalculator
from src.pot import pot_eval
from src.ctf import CTF_Trainer

# For POT evaluation, we need dataset-specific thresholds
POT_THRESHOLDS = {
    'SMD': (0.99995, 1.04),
    'SWaT': (0.993, 1),
    'SMAP': (0.98, 1),
    'MSL': (0.999, 1.04),
    'WADI': (0.999, 1),
}

# ==========================================
# HELPER FUNCTIONS (Copied from full_comparison.py)
# ==========================================
def calc_f1_pot(init_scores, test_scores, labels, dataset='MSL'):
    lm = POT_THRESHOLDS.get(dataset, (0.98, 1.0))
    try:
        result, pred = pot_eval(init_scores, test_scores, labels, q=1e-5, level=lm[0], lm=lm)
        return result['f1']
    except Exception as e:
        print(f"POT evaluation failed: {e}")
        return 0.0

def calc_point_adjusted_f1(scores, labels):
    if len(scores) < len(labels): labels = labels[-len(scores):]
    else: scores = scores[-len(labels):]
    best_f1 = 0.0
    # thresholds = np.percentile(scores, np.linspace(0, 99.9, 50))
    # Use more granular thresholds, especially in the tail
    thresholds = np.percentile(scores, np.concatenate([
        np.linspace(0, 99, 50),
        np.linspace(99, 99.9, 50),
        np.linspace(99.9, 99.99, 50)
    ]))
    
    # Optimize: Pre-calculate max score for each anomaly segment
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
        
    if not anomaly_segments:
        return 0.0
        
    segment_max_scores = []
    segment_lengths = []
    for start, end in anomaly_segments:
        segment_max_scores.append(np.max(scores[start:end]))
        segment_lengths.append(end - start)
        
    segment_max_scores = np.array(segment_max_scores)
    segment_lengths = np.array(segment_lengths)
    
    # Normal scores (for FP calculation)
    normal_scores = np.sort(scores[labels == 0])
    n_normal = len(normal_scores)
    
    for thresh in thresholds:
        # TP: Points in segments that are detected (max_score > thresh)
        detected_segments_mask = (segment_max_scores > thresh)
        tp = np.sum(segment_lengths[detected_segments_mask])
        
        # FN: Points in segments that are NOT detected
        fn = np.sum(segment_lengths[~detected_segments_mask])
        
        # FP: Normal points that are predicted as anomaly
        # Use binary search for speed: O(log N) instead of O(N)
        fp_index = np.searchsorted(normal_scores, thresh, side='right')
        fp = n_normal - fp_index
        
        if tp > 0:
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1 = 2 * (p * r) / (p + r + 1e-8)
            if f1 > best_f1: best_f1 = f1
            
    return best_f1

def train_scratch(data, device, epochs=10):
    feat_dim = data.shape[1]
    model = TranAD(feat_dim=feat_dim).double().to(device)
    dummy_whac = WHAC_Clustering(np.ones(feat_dim), window_size=10)
    segments = dummy_whac.segment_data(data)
    tensor_x = torch.DoubleTensor(segments).to(device)
    dataset = TensorDataset(tensor_x, tensor_x)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    criterion = nn.MSELoss(reduction='none')
    model.train()
    for epoch in range(epochs):
        n = epoch + 1
        for bx, by in loader:
            bx_permuted = bx.permute(1, 0, 2)
            local_bs = bx.shape[0]
            feats = bx.shape[2]
            elem = bx_permuted[-1:, :, :].view(1, local_bs, feats)
            optimizer.zero_grad()
            z = model(bx_permuted, elem)
            l1 = (1 / n) * criterion(z[0], elem) + (1 - 1/n) * criterion(z[1], elem)
            loss = torch.mean(l1)
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
    return model

def detect_scratch(model, test_data, device):
    feat_dim = test_data.shape[1]
    dummy_whac = WHAC_Clustering(np.ones(feat_dim), window_size=10)
    segments = dummy_whac.segment_data(test_data, stride=1)
    model.eval()
    batch_size = 256
    scores = []
    with torch.no_grad():
        for i in range(0, len(segments), batch_size):
            batch_segs = segments[i : i + batch_size]
            tensor_x = torch.DoubleTensor(batch_segs).to(device)
            tensor_x_permuted = tensor_x.permute(1, 0, 2)
            local_bs = tensor_x.shape[0]
            elem = tensor_x_permuted[-1:, :, :].view(1, local_bs, feat_dim)
            z = model(tensor_x_permuted, elem)
            loss = torch.mean((elem - z[1]) ** 2, dim=2)
            batch_scores = loss[0, :].cpu().numpy()
            scores.extend(batch_scores)
    scores = np.array(scores)
    pad_length = 10 - 1
    padded_scores = np.concatenate([np.zeros(pad_length), scores])
    return padded_scores

# ==========================================
# MAIN COMPARISON (Modified for Subset)
# ==========================================
def run_single_dataset(dataset_name, device, selected_configs=None, alpha=2.0, eval_method='point_adjust', limit_entities=20):
    print(f"\n{'='*80}")
    print(f"SUBSET COMPARISON: {dataset_name} (Limit: {limit_entities})")
    print(f"{'='*80}\n")
    
    loader = ProcessedDataLoader(processed_dir='processed')
    p_calc = PeriodicityCalculator()
    e_calc = EntropicWeightCalculator(alpha=alpha)
    c_calc = CausalWeightCalculator(max_lag=2)
    
    all_entities = loader.get_available_entities(dataset_name)
    if not all_entities:
        print(f"Error: No data found for {dataset_name}")
        return None
    
    # LIMIT ENTITIES
    if len(all_entities) > limit_entities:
        print(f"Limiting entities from {len(all_entities)} to {limit_entities}")
        all_entities = all_entities[:limit_entities]
    
    split_idx = len(all_entities) // 2
    source_entities = all_entities[:split_idx]
    target_entities = all_entities[split_idx:]
    
    print(f"Total Entities: {len(all_entities)}")
    print(f"  Source: {len(source_entities)} | Target: {len(target_entities)}\n")
    
    all_configs = {
        'TranAD_Scratch': {'type': 'scratch'},
        'OmniTransfer_Periodic': {'type': 'omni', 'use_entropy': False, 'use_causal': False},
        'OmniTransfer_Entropy': {'type': 'omni', 'use_entropy': True, 'use_causal': False},
        'OmniTransfer_Causal': {'type': 'omni', 'use_entropy': False, 'use_causal': True},
        'OmniTransfer_Entropy_Causal': {'type': 'omni', 'use_entropy': True, 'use_causal': True},
        'OmniTransfer_CTF_Periodic': {'type': 'omni', 'use_entropy': False, 'use_causal': False, 'model': 'RNN_VAE'},
        'OmniTransfer_CTF_Entropy': {'type': 'omni', 'use_entropy': True, 'use_causal': False, 'model': 'RNN_VAE'},
        'CTF': {'type': 'ctf'},
    }
    
    if selected_configs:
        configs = {k: v for k, v in all_configs.items() if k in selected_configs}
    else:
        configs = all_configs
    
    all_results = []
    
    for config_name, config in configs.items():
        print(f"\n{'='*80}")
        print(f"[{config_name}]")
        print(f"{'='*80}")
        
        if config['type'] == 'scratch':
            for idx, entity in enumerate(target_entities):
                print(f"  {idx+1}/{len(target_entities)}: {entity}...", end=' ')
                train = loader.load(dataset_name, entity, 'train')
                test = loader.load(dataset_name, entity, 'test')
                labels = loader.load(dataset_name, entity, 'labels')
                if len(labels.shape) > 1: labels = labels[:, 0]
                
                # Filter low variance features for CTF
                if dataset_name == 'CTF':
                    train, test, _ = loader.filter_low_variance(train, test)
                
                t0 = time.time()
                model = train_scratch(train, device, epochs=10)
                train_time = time.time() - t0
                scores = detect_scratch(model, test, device)
                if eval_method == 'pot':
                    train_scores = detect_scratch(model, train, device)
                    f1 = calc_f1_pot(train_scores, scores, labels, dataset=dataset_name)
                else:
                    f1 = calc_point_adjusted_f1(scores, labels)
                print(f"Time: {train_time:.2f}s | F1: {f1:.4f}")
                all_results.append({
                    'Dataset': dataset_name, 'Config': config_name, 'Machine': entity, 'TrainTime': train_time, 'F1': f1
                })
                
        elif config['type'] == 'ctf':
            print(f"  Training CTF Global Model (Offline)...")
            
            # Populate source_list if needed (e.g. if CTF is run alone)
            # In subset_comparison, source_list is not pre-populated outside configs.
            # We must populate it here.
            source_list = []
            for src_entity in source_entities:
                s_data = loader.load(dataset_name, src_entity, 'train')
                # Note: We don't filter source data here because we need consistent dims for vstack in CTF_Trainer.
                # If we filter per-entity, dims might mismatch.
                # However, target data IS filtered. This is a potential issue if dims mismatch between source and target.
                # But for now, let's stick to the previous logic: load raw source data.
                source_list.append(s_data)
            
            # Global Feature Filtering for CTF
            keep_indices = None
            if dataset_name == 'CTF':
                all_source = np.vstack(source_list)
                _, _, keep_indices = loader.filter_low_variance(all_source, all_source)
                print(f"  [CTF] Global Filtering: Keeping {len(keep_indices)}/{all_source.shape[1]} features.")
                source_list = [d[:, keep_indices] for d in source_list]
            
            source_dict = {entity: data for entity, data in zip(source_entities, source_list)}
            
            # Train Offline ONCE
            trainer = CTF_Trainer(feat_dim=source_list[0].shape[1], device=device)
            periodic_weights = np.ones(source_list[0].shape[1])
            trainer.train_offline(source_dict, periodic_weights, n_clusters=3, epochs=50)
            print(f"  CTF Model Ready.\n")
            
            for idx, entity in enumerate(target_entities):
                print(f"  {idx+1}/{len(target_entities)}: {entity}...", end=' ')
                train = loader.load(dataset_name, entity, 'train')
                test = loader.load(dataset_name, entity, 'test')
                labels = loader.load(dataset_name, entity, 'labels')
                if len(labels.shape) > 1: labels = labels[:, 0]
                
                # Apply Global Filter to Target
                if keep_indices is not None:
                    train = train[:, keep_indices]
                    test = test[:, keep_indices]
                
                t0 = time.time()
                
                # Note: If we filtered target features, we might have a dimension mismatch with the pre-trained model.
                # The pre-trained model expects 'feat_dim' features.
                # If 'keep_indices' reduces features, 'detect' will fail.
                # This is a known issue with filtering per-entity in a transfer setting.
                # However, for CTF, the paper implies we use the same features.
                # If we filter, we must ensure the model can handle it.
                # But the model is fixed size.
                # So if we filter target, we MUST filter source the same way?
                # But source is already trained.
                # Wait, in the previous "slow" version, we re-trained per entity, so we could filter source using target's mask.
                # Now that we train ONCE, we can't filter source differently for each target.
                # So we must either:
                # 1. NOT filter for CTF (but performance was bad).
                # 2. Filter GLOBALLY (intersection of all valid features).
                # 3. Handle missing features in the model (zero padding?).
                
                # Given the user's request for speed, let's assume for now we DON'T filter or we accept the risk.
                # But wait, if I don't filter, performance drops.
                # If I filter target, model crashes (dim mismatch).
                # So I MUST filter globally or not at all.
                # Or... I can't use this "Train Once" optimization with "Per-Entity Filtering".
                # This is a trade-off.
                # But wait, the "low variance" features are likely the SAME across entities (e.g. constant outputs).
                # So maybe I can filter based on the FIRST source entity and assume it holds?
                # Or compute a global mask.
                
                # Let's try to compute a global mask from source data?
                # But the issue is target might have different low-var features.
                
                # For now, to make it run, I will DISABLE filtering in this optimized block 
                # OR I must implement global filtering.
                # Let's try to implement global filtering on source data.
                # If I filter source data, the model will have reduced dims.
                # Then for target, I must apply the SAME mask.
                # So I need to store the mask.
                
                # Let's refine the logic:
                # 1. Load all source data.
                # 2. Compute global mask (features with variance > threshold in ALL/ANY source?).
                #    Let's say in concatenated source.
                # 3. Train model on filtered source.
                # 4. Apply mask to target.
                
                # Implementation:
                # Inside the "Train Offline" block:
                # all_source = np.vstack(source_list)
                # mask = ...
                # source_dict = {e: d[:, mask] ...}
                # trainer = ...
                
                # Then in loop:
                # train = train[:, mask]
                # test = test[:, mask]
                
                # This seems correct and robust.
                
                scores = trainer.detect(test)
                train_time = time.time() - t0
                
                if eval_method == 'pot':
                    train_scores = trainer.detect(train)
                    f1 = calc_f1_pot(train_scores, scores, labels, dataset=dataset_name)
                else:
                    f1 = calc_point_adjusted_f1(scores, labels)
                
                print(f"Time: {train_time:.2f}s | F1: {f1:.4f}")
                all_results.append({
                    'Dataset': dataset_name, 'Config': config_name, 'Machine': entity, 'TrainTime': train_time, 'F1': f1
                })

        else:
            use_entropy = config['use_entropy']
            use_causal = config['use_causal']
            print(f"  Building Shape Library...")
            source_list = []
            weights_list = []
            for entity in source_entities:
                data = loader.load(dataset_name, entity, 'train')
                source_list.append(data)
                if use_entropy: w_base = e_calc.compute_entropic_weights(data)
                else: w_base = p_calc.compute_weights(data)
                if use_causal:
                    w_c = c_calc.compute_causal_weights(data)
                    w_final = w_base * (1.0 + w_c)
                else: w_final = w_base
                weights_list.append(w_final)
            big_data = np.vstack(source_list)
            global_weights = np.mean(np.array(weights_list), axis=0)
            model_class = TranAD
            if config.get('model') == 'RNN_VAE': model_class = RNN_VAE
            trainer = OmniTransferTrainer(model_class, device=device)
            trainer.train_offline(big_data, global_weights, n_clusters=None, epochs=10)
            print(f"  Shape Library Ready.\n")
            for idx, entity in enumerate(target_entities):
                print(f"  {idx+1}/{len(target_entities)}: {entity}...", end=' ')
                train = loader.load(dataset_name, entity, 'train')
                test = loader.load(dataset_name, entity, 'test')
                labels = loader.load(dataset_name, entity, 'labels')
                if len(labels.shape) > 1: labels = labels[:, 0]
                t0 = time.time()
                model = trainer.online_transfer(train, epochs=10)
                train_time = time.time() - t0
                scores = trainer.detect(model, test)
                if eval_method == 'pot':
                    train_scores = trainer.detect(model, train)
                    f1 = calc_f1_pot(train_scores, scores, labels, dataset=dataset_name)
                else:
                    f1 = calc_point_adjusted_f1(scores, labels)
                print(f"Time: {train_time:.2f}s | F1: {f1:.4f}")
                all_results.append({
                    'Dataset': dataset_name, 'Config': config_name, 'Machine': entity, 'TrainTime': train_time, 'F1': f1
                })
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Subset OmniTransfer Comparison')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='Dataset names')
    parser.add_argument('--configs', type=str, nargs='+', default=['TranAD_Scratch'], help='Configs to run')
    parser.add_argument('--limit', type=int, default=20, help='Max entities to use (default: 20)')
    parser.add_argument('--alpha', type=float, default=2.0, help='Alpha for entropy')
    parser.add_argument('--eval_method', type=str, default='point_adjust', choices=['point_adjust', 'pot'])
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print(f"RUNNING SUBSET COMPARISON (Limit: {args.limit})")
    print(f"CONFIGS: {', '.join(args.configs)}")
    print(f"{'='*80}\n")
    
    all_datasets_results = []
    for dataset in args.datasets:
        results = run_single_dataset(dataset, device, selected_configs=args.configs, alpha=args.alpha, eval_method=args.eval_method, limit_entities=args.limit)
        if results: all_datasets_results.extend(results)
    
    if not all_datasets_results:
        print("No results generated!")
        return
    
    df = pd.DataFrame(all_datasets_results)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets_str = '_'.join(args.datasets)
    csv_path = f"results/csv/{datasets_str}_subset_comparison_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"COMBINED RESULTS SAVED: {csv_path}")
    print(f"{'='*80}\n")
    
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    overall_summary = df.groupby('Config').agg({'F1': 'mean', 'TrainTime': 'mean'}).round(4)
    overall_summary.columns = ['Avg_F1', 'Avg_Time']
    print(overall_summary.reset_index().to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
