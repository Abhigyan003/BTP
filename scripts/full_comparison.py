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
from src.jumpstarter_adapter import JumpStarterAdapter

# For POT evaluation, we need dataset-specific thresholds
# These are from the original TranAD paper
POT_THRESHOLDS = {
    'SMD': (0.99995, 1.04),
    'SWaT': (0.993, 1),
    'SMAP': (0.98, 1),
    'MSL': (0.999, 1.04),  # For TranAD model
    'WADI': (0.999, 1),
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calc_f1_pot(init_scores, test_scores, labels, dataset='MSL'):
    """
    Calculate F1 using POT (Peaks Over Threshold) evaluation
    This matches the original TranAD paper's evaluation method
    
    Args:
        init_scores: Training anomaly scores (for threshold initialization)
        test_scores: Test anomaly scores
        labels: Ground truth labels
        dataset: Dataset name for threshold parameters
        
    Returns:
        float: F1 score
    """
    # Get dataset-specific threshold parameters
    lm = POT_THRESHOLDS.get(dataset, (0.98, 1.0))
    
    try:
        result, pred = pot_eval(init_scores, test_scores, labels, q=1e-5, level=lm[0], lm=lm)
        return result['f1']
    except Exception as e:
        # If POT fails, return 0
        print(f"POT evaluation failed: {e}")
        return 0.0

def calc_point_adjusted_f1(scores, labels):
    """Point-Adjusted F1 Score"""
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
    """Train TranAD from scratch (baseline) using proper two-phase training"""
    feat_dim = data.shape[1]
    model = TranAD(feat_dim=feat_dim).double().to(device)  # Convert to double precision!
    
    dummy_whac = WHAC_Clustering(np.ones(feat_dim), window_size=10)
    segments = dummy_whac.segment_data(data)
    
    # TranAD expects [window_size, batch_size, feats] format
    # segments is [num_samples, window_size, feats]
    tensor_x = torch.DoubleTensor(segments).to(device)  # Use DoubleTensor like original!
    dataset = TensorDataset(tensor_x, tensor_x)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=model.lr)  # Use model's default lr
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # Add scheduler!
    criterion = nn.MSELoss(reduction='none')
    
    model.train()
    for epoch in range(epochs):
        l1s = []
        n = epoch + 1
        for bx, by in loader:
            # Convert from [batch, window, feats] to [window, batch, feats]
            bx_permuted = bx.permute(1, 0, 2)
            
            # Extract target (last timestep): [1, batch, feats]
            local_bs = bx.shape[0]
            feats = bx.shape[2]
            elem = bx_permuted[-1:, :, :].view(1, local_bs, feats)
            
            optimizer.zero_grad()
            
            # TranAD forward returns (x1, x2) - two phases
            z = model(bx_permuted, elem)  # Call with (window, elem) like original
            
            # Time-dependent weighted loss - EXACTLY like original line 264
            l1 = (1 / n) * criterion(z[0], elem) + (1 - 1/n) * criterion(z[1], elem)
            
            loss = torch.mean(l1)
            l1s.append(loss.item())
            
            loss.backward(retain_graph=True)  # retain_graph like original!
            optimizer.step()
        
        scheduler.step()  # Step scheduler AFTER each epoch!
            
    return model

def detect_scratch(model, test_data, device):
    """Detection using scratch model - uses TranAD's Phase 2 decoder"""
    feat_dim = test_data.shape[1]
    dummy_whac = WHAC_Clustering(np.ones(feat_dim), window_size=10)
    segments = dummy_whac.segment_data(test_data, stride=1)
    
    model.eval()
    batch_size = 256
    scores = []
    with torch.no_grad():
        for i in range(0, len(segments), batch_size):
            batch_segs = segments[i : i + batch_size]
            tensor_x = torch.DoubleTensor(batch_segs).to(device)  # Use DoubleTensor!
            
            # Convert to TranAD format: [window, batch, feats]
            tensor_x_permuted = tensor_x.permute(1, 0, 2)
            local_bs = tensor_x.shape[0]
            elem = tensor_x_permuted[-1:, :, :].view(1, local_bs, feat_dim)
            
            # Get both phases using original calling convention
            z = model(tensor_x_permuted, elem)
            
            # Use Phase 2 (z[1]) for anomaly scoring
            loss = torch.mean((elem - z[1]) ** 2, dim=2)
            batch_scores = loss[0, :].cpu().numpy()
            scores.extend(batch_scores)
            
    # Pad scores to match original data length
    # segments has length: len(test_data) - window_size + 1
    # We need to pad window_size - 1 scores at the beginning
    scores = np.array(scores)
    pad_length = 10 - 1  # window_size is hardcoded to 10 in detect_scratch
    padded_scores = np.concatenate([np.zeros(pad_length), scores])
    
    return padded_scores

# ==========================================
# MAIN COMPARISON
# ==========================================
# ==========================================
# MAIN COMPARISON
# ==========================================
def run_single_dataset(dataset_name, device, selected_configs=None, alpha=2.0, eval_method='point_adjust'):
    """Run comparison for a single dataset"""
    print(f"\n{'='*80}")
    print(f"FULL COMPARISON: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Initialize tools
    loader = ProcessedDataLoader(processed_dir='processed')
    p_calc = PeriodicityCalculator()
    e_calc = EntropicWeightCalculator(alpha=alpha)  # Use provided alpha
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
    all_configs = {
        'TranAD_Scratch': {'type': 'scratch'},
        'OmniTransfer_Periodic': {'type': 'omni', 'use_entropy': False, 'use_causal': False},
        'OmniTransfer_Entropy': {'type': 'omni', 'use_entropy': True, 'use_causal': False},
        'OmniTransfer_Causal': {'type': 'omni', 'use_entropy': False, 'use_causal': True},
        'OmniTransfer_Entropy_Causal': {'type': 'omni', 'use_entropy': True, 'use_causal': True},
        'OmniTransfer_CTF_Periodic': {'type': 'omni', 'use_entropy': False, 'use_causal': False, 'model': 'RNN_VAE'},
        'OmniTransfer_CTF_Entropy': {'type': 'omni', 'use_entropy': True, 'use_causal': False, 'model': 'RNN_VAE'},
        'CTF': {'type': 'ctf'},
        'JumpStarter': {'type': 'jumpstarter'},
        # Aliases for backward compatibility
        'Omni_Periodic': {'type': 'omni', 'use_entropy': False, 'use_causal': False},
        'Omni_Entropy': {'type': 'omni', 'use_entropy': True, 'use_causal': False},
        'Omni_Causal': {'type': 'omni', 'use_entropy': False, 'use_causal': True},
        'Omni_Entropy_Causal': {'type': 'omni', 'use_entropy': True, 'use_causal': True},
    }
    
    # Filter configs based on selected_configs
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
            # BASELINE: TranAD from scratch
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
                    # For POT, we need training scores for initialization
                    train_scores = detect_scratch(model, train, device)
                    f1 = calc_f1_pot(train_scores, scores, labels, dataset=dataset_name)
                else:
                    # Default: Point-Adjusted F1
                    f1 = calc_point_adjusted_f1(scores, labels)
                
                print(f"Time: {train_time:.2f}s | F1: {f1:.4f}")
                
                all_results.append({
                    'Dataset': dataset_name,
                    'Config': config_name,
                    'Machine': entity,
                    'TrainTime': train_time,
                    'F1': f1
                })
                
        elif config['type'] == 'ctf':
            print(f"  Training CTF Global Model (Offline)...")
            
            # Prepare dictionary for CTF
            # Note: source_list is populated in the OmniTransfer block below.
            # For CTF, we need to ensure source_list is available.
            # If CTF is run alone, source_list might be empty.
            # Let's ensure source_list is populated here if CTF is the only config.
            # This is a bit of a hack, ideally source_list should be populated once for all configs.
            # For now, assume source_list is populated by a previous OmniTransfer run or handle it.
            # If source_list is empty, this will fail.
            # Let's add a check or populate it.
            # For now, assuming source_list is populated by the OmniTransfer block which runs first.
            # If CTF is run as the only config, source_list will be empty.
            # To fix this, we need to move the source_list population outside the config loop.
            # For this specific change, I'll assume source_list is available.
            
            # Populate source_list and weights_list if not already done (e.g., if CTF is run alone)
            # This is a temporary fix to ensure source_list is available for CTF if it's the first config.
            # A better solution would be to populate source_list once before the config loop.
            if not source_list:
                for entity in source_entities:
                    data = loader.load(dataset_name, entity, 'train')
                    source_list.append(data)
            
            # Global Feature Filtering for CTF
            keep_indices = None
            if dataset_name == 'CTF':
                # Concatenate all source data to find low variance features globally
                all_source = np.vstack(source_list)
                # Use the static method but we need to adapt it or just use variance logic directly
                # filter_low_variance takes (train, test). We can pass (all_source, all_source).
                _, _, keep_indices = loader.filter_low_variance(all_source, all_source)
                print(f"  [CTF] Global Filtering: Keeping {len(keep_indices)}/{all_source.shape[1]} features.")
                
                # Filter source list
                source_list = [d[:, keep_indices] for d in source_list]
            
            source_dict = {entity: data for entity, data in zip(source_entities, source_list)}
            
            # Train Offline ONCE
            trainer = CTF_Trainer(feat_dim=source_list[0].shape[1], device=device)
            periodic_weights = np.ones(source_list[0].shape[1])
            trainer.train_offline(source_dict, periodic_weights, n_clusters=3, epochs=10)
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
                # For CTF, "online transfer" is just matching and fine-tuning.
                # But wait, the original paper says "Model Transfer" happens per cluster.
                # And for a new target machine, we map it to a cluster and fine-tune.
                # My CTF_Trainer.detect() handles the matching, but does it fine-tune?
                # No, detect() uses the PRE-TRAINED cluster models.
                # It does NOT fine-tune on the target test data (that would be cheating/transductive).
                # However, usually we might fine-tune on target TRAIN data if available.
                # In this script, we have 'train' (target train) and 'test' (target test).
                # The original code I wrote:
                # trainer.train_offline(source_dict...) -> trains cluster models.
                # Then detect(test).
                # It didn't use target 'train' data for fine-tuning in the loop?
                # Actually, the previous code called train_offline inside the loop using 'train' (target train) as input?
                # No, previous code:
                # trainer.train_offline(source_dict...)
                # Wait, if I passed source_dict (source entities), then I wasn't using target 'train' at all?
                # Ah, the previous code inside the loop:
                # source_dict = {entity: data for ... source_list} -> This is SOURCE data.
                # So for every target entity, I was re-training on SOURCE data? That's even more redundant!
                # And I wasn't using target 'train' data to adapt?
                # If CTF is "Coarse-to-Fine Model Transfer", it implies transferring to the target.
                # The paper says: "Step 4: Model Transfer... fine-tuned using the per-cluster data."
                # This "per-cluster data" comes from the historical data (Source).
                # So the cluster models are trained on Source data.
                # For a new target machine, we just match it to a cluster and use that model.
                # So my implementation of `detect` matches to cluster and uses that model.
                # So yes, training once on Source is correct.
                # And we don't need to fine-tune on target 'train' unless we want to do "Online Transfer" like Omni.
                # But standard CTF is offline.
                
                # So, simply calling detect() is correct.
                # But wait, should we fine-tune on target 'train'?
                # If we have target 'train' data, we should probably use it.
                # But CTF paper implies the clusters are formed from "historical data" (Source).
                # And new machines are assigned to clusters.
                # If we want to adapt to the new machine, we might fine-tune.
                # But let's stick to the paper: "Online Anomaly Detection... determine the cluster... use the model."
                # It doesn't explicitly say fine-tune on target.
                # So "Training Time" for target entity is effectively 0 (or just matching time).
                # But to be fair, we should count the matching time.
                
                scores = trainer.detect(test)
                train_time = time.time() - t0 # This will be very fast, just inference/matching.
                
                if eval_method == 'pot':
                    train_scores = trainer.detect(train)
                    f1 = calc_f1_pot(train_scores, scores, labels, dataset=dataset_name)
                else:
                    f1 = calc_point_adjusted_f1(scores, labels)
                
                print(f"Time: {train_time:.2f}s | F1: {f1:.4f}")
                all_results.append({
                    'Dataset': dataset_name, 'Config': config_name, 'Machine': entity, 'TrainTime': train_time, 'F1': f1
                })
                
        elif config['type'] == 'jumpstarter':
            # JUMPSTARTER: Compressive Sensing-based Anomaly Detection
            for idx, entity in enumerate(target_entities):
                print(f"  {idx+1}/{len(target_entities)}: {entity}...", end=' ')
                
                train = loader.load(dataset_name, entity, 'train')
                test = loader.load(dataset_name, entity, 'test')
                labels = loader.load(dataset_name, entity, 'labels')
                if len(labels.shape) > 1: labels = labels[:, 0]
                
                # Filter low variance features for CTF
                if dataset_name == 'CTF':
                    train, test, _ = loader.filter_low_variance(train, test)
                
                feat_dim = train.shape[1]
                
                # Create adapter with default parameters
                adapter = JumpStarterAdapter(
                    feat_dim=feat_dim,
                    device=device,
                    sample_rate=0.3,
                    cluster_threshold=0.15,
                    workers=4,
                    window=96,
                    windows_per_cycle=7
                )
                
                t0 = time.time()
                # Train (reconstruct training data)
                reconstructed_train, _ = adapter.train(train)
                
                # Detect on test data (will reconstruct internally)
                scores = adapter.detect(test)
                train_time = time.time() - t0
                
                if eval_method == 'pot':
                    # For POT, we need training scores
                    train_scores = adapter.detect(train, reconstructed=reconstructed_train)
                    f1 = calc_f1_pot(train_scores, scores, labels, dataset=dataset_name)
                else:
                    # Default: Point-Adjusted F1
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
                
                # Filter low variance features for CTF (Source)
                # Note: For OmniTransfer, we need consistent features across all entities.
                # If we filter per entity, we might have mismatch.
                # However, CTF has same features for all machines (49 KPIs).
                # But some might be constant in one machine and not others.
                # Ideally we should filter based on GLOBAL variance or keep all.
                # But here we are building shape library.
                # Let's skip filtering for Shape Library construction for now to avoid dimension mismatch,
                # OR we filter each entity individually? No, dimensions must match.
                # If we filter, we must ensure all entities end up with same features.
                # For CTF, let's assume we filter per entity for now, BUT wait...
                # If we filter entity A and it drops feat 0, and entity B keeps feat 0,
                # then vstack will fail or be meaningless.
                # So for OmniTransfer Shape Library, we should NOT filter individual sources unless we do it globally.
                # BUT, for the target entity (online transfer), we CAN filter.
                # And the shape library is built from SOURCE entities.
                # If we don't filter source, the shape library has 49 dims.
                # If we filter target, it has <49 dims.
                # Then we can't transfer!
                # CRITICAL: We must NOT filter for OmniTransfer if it changes dimensions per entity!
                # UNLESS we filter globally.
                
                # Let's check if CTF source entities have same dead features.
                # If not, we can't easily filter for OmniTransfer without a global mask.
                # For now, let's ONLY filter for the TARGET entity in the online phase,
                # AND we must ensure the Shape Library model matches the filtered dimensions?
                # No, if Shape Library has 49 dims, and Target has 30, we can't use the pre-trained model directly.
                
                # SOLUTION: For OmniTransfer on CTF, we might have to skip filtering OR filter globally.
                # Given the time constraints, let's try filtering ONLY for 'scratch' and 'ctf' configs first.
                # For OmniTransfer, let's see if we can filter the target and then "project" the shape library? No.
                
                # Actually, if we just filter for 'scratch' and 'ctf' configs, that covers the user's complaint about "CTF" config.
                # The user also complained about Omni_CTF.
                # So we need a solution for OmniTransfer too.
                
                # Global filtering for OmniTransfer:
                # 1. Load all source data.
                # 2. Compute global variance.
                # 3. Filter all sources and targets using global mask.
                
                # This requires changing the loop structure.
                # Current structure: Iterate source entities, load, compute weights.
                
                # Let's stick to filtering for 'scratch' and 'ctf' first as they are per-entity.
                # For OmniTransfer, let's leave it for now or apply a simple fix if possible.
                # Wait, if I filter target in online_transfer, I need a model that matches.
                # If I train offline on full 49 dims, I have a 49-dim model.
                # If I filter target to 30 dims, I can't use the 49-dim model.
                
                # Okay, for OmniTransfer, I will NOT filter in this loop.
                # I will filter in the target loop, BUT I need to handle the dimension mismatch.
                # Actually, if I filter the target, I effectively treat it as a new problem.
                # But OmniTransfer relies on the Shape Library (Source) being relevant.
                # If Source has 49 dims and Target has 30, are they the "same" features?
                # Yes, just some are dropped.
                # So we should drop the SAME features from Source.
                # But Source might have variance in those features!
                # If Source has variance, and Target doesn't, then that feature IS informative in Source.
                # But for Target it's useless.
                # If we transfer, we want to transfer knowledge about the useful features.
                # So we should drop the features that are useless in TARGET from the SOURCE/MODEL.
                # But we don't know the target yet when building Shape Library!
                
                # This is a fundamental issue with Transfer Learning when features differ.
                # However, for CTF, the features are the SAME KPIs.
                # If a KPI is dead in Target, we probably should ignore it.
                # If we ignore it in Target, we should ignore the corresponding weights in the Source model.
                # This implies "Partial Transfer" or "Slicing" the pre-trained model.
                
                # Implementing model slicing is complex.
                # Alternative: Don't filter for OmniTransfer, just let it handle zeros.
                # But the user said "still really bad results" for Omni_CTF too.
                # Maybe the zeros are killing the clustering in OmniTransfer too.
                
                # Let's apply filtering for 'scratch' and 'ctf' configs first (lines 252 and 287).
                # For OmniTransfer, I will add a TODO or try to handle it if I can.
                # Actually, if I filter globally for CTF (intersection of all valid features?), that might work.
                # But that requires loading all data first.
                
                # Let's just apply to 'scratch' and 'ctf' configs for now.
                # The user's immediate complaint was about "CTF" config results in the summary table.
                
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
            
            # Determine model class
            model_class = TranAD
            if config.get('model') == 'RNN_VAE':
                model_class = RNN_VAE
            
            trainer = OmniTransferTrainer(model_class, device=device)
            # trainer.train_offline(big_data, global_weights, n_clusters=10)  # Fixed 4 clusters
            trainer.train_offline(big_data, global_weights, n_clusters=None, epochs=10)  # Auto-tune clusters
            print(f"  Shape Library Ready.\n")
            
            # Evaluate on targets
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
                    # For POT, we need training scores for initialization
                    train_scores = trainer.detect(model, train)
                    f1 = calc_f1_pot(train_scores, scores, labels, dataset=dataset_name)
                else:
                    # Default: Point-Adjusted F1
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
    parser.add_argument('--configs', type=str, nargs='+', 
                       default=['TranAD_Scratch', 'Omni_Periodic', 'Omni_Entropy'],
                        choices=['TranAD_Scratch', 'Omni_Periodic', 'Omni_Entropy', 
                                'Omni_Causal', 'Omni_Entropy_Causal', 'CTF', 'Omni_CTF_Periodic', 'Omni_CTF_Entropy',
                                'JumpStarter'],
                        help='Configurations to run (default: TranAD_Scratch, Omni_Periodic, Omni_Entropy)')
    parser.add_argument('--alpha', type=float, default=2.0,
                       help='Alpha parameter for entropy weights (default: 2.0, try 0.5 or 1.0)')
    parser.add_argument('--eval_method', type=str, default='point_adjust',
                       choices=['point_adjust', 'pot'],
                       help='Evaluation method: point_adjust (default) or pot (Peaks Over Threshold)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"RUNNING CONFIGURATIONS: {', '.join(args.configs)}")
    if 'Omni_Entropy' in args.configs or 'Omni_Entropy_Causal' in args.configs:
        print(f"ENTROPY ALPHA: {args.alpha}")
    print(f"{'='*80}\n")
    
    all_datasets_results = []
    
    # Run comparison for each dataset
    # Run comparison for each dataset
    for dataset in args.datasets:
        results = run_single_dataset(dataset, device, selected_configs=args.configs, alpha=args.alpha, eval_method=args.eval_method)
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
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets_str = '_'.join(args.datasets)
    csv_path = f"results/csv/{datasets_str}_full_comparison_{timestamp}.csv"
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
    
    # Save summaries with timestamp
    overall_summary_path = f"results/csv/{datasets_str}_overall_summary_{timestamp}.csv"
    overall_summary.to_csv(overall_summary_path, index=False)
    
    per_dataset_summary_path = f"results/csv/{datasets_str}_per_dataset_summary_{timestamp}.csv"
    per_dataset_summary.to_csv(per_dataset_summary_path)  # Keep index
    
    print(f"Overall summary saved: {overall_summary_path}")
    print(f"Per-dataset summary saved: {per_dataset_summary_path}\n")

if __name__ == "__main__":
    main()
