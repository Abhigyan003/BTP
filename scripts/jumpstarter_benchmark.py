"""
JumpStarter Standalone Benchmark Script

Evaluates JumpStarter's compressive sensing-based anomaly detection
on multiple datasets from the processed directory.

This script uses the JumpStarter implementation from ../JumpStarter directory.
"""

import sys
import os

# Add JumpStarter directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)  # Go up to code/
jumpstarter_dir = os.path.join(code_dir, 'JumpStarter')
sys.path.insert(0, jumpstarter_dir)

import numpy as np
import pandas as pd
import time
import argparse
from datetime import datetime

from detector import CSAnomalyDetector


def normalization(data):
    """Min-max normalization per feature [0, 1]"""
    _range = np.max(data) - np.min(data)
    if _range == 0:
        return np.zeros_like(data) + 0.5
    return (data - np.min(data)) / _range


def lesinn_score(incoming_data, historical_data, random_state=42):
    """LESINN-based sampling confidence"""
    from algorithm.lesinn import online_lesinn
    distances = online_lesinn(
        incoming_data, historical_data,
        random_state=random_state,
        t=40,
        phi=20
    )
    # p_normalize
    p_min = 0.05
    x = 1 / (distances + 1e-8)
    x_max, x_min = np.max(x), np.min(x)
    x_min *= (1 - p_min)
    return (x - x_min) / (x_max - x_min + 1e-8)


def anomaly_score_example(source, reconstructed):
    """Distance-based anomaly scoring"""
    n, d = source.shape
    d_dis = np.zeros((d,))
    
    percentage = 90
    topn = 2
    
    for i in range(d):
        dis = np.abs(source[:, i] - reconstructed[:, i])
        dis = dis - np.mean(dis)
        d_dis[i] = np.percentile(dis, percentage)
    
    if d <= topn:
        return d / (np.sum(1 / (d_dis + 1e-8)))
    
    topn_vals = 1 / (d_dis[np.argsort(d_dis)][-topn:] + 1e-8)
    return topn / (np.sum(topn_vals))


def calc_point_adjusted_f1(scores, labels):
    """Point-Adjusted F1 Score"""
    if len(scores) < len(labels):
        labels = labels[-len(scores):]
    else:
        scores = scores[-len(labels):]
    
    best_f1 = 0.0
    thresholds = np.percentile(scores, np.concatenate([
        np.linspace(0, 99, 50),
        np.linspace(99, 99.9, 50),
        np.linspace(99.9, 99.99, 50)
    ]))
    
    actual = (labels == 1)
    
    # Pre-calculate anomaly segments
    anomaly_segments = []
    i = 0
    while i < len(labels):
        if actual[i]:
            j = i
            while j < len(labels) and actual[j]:
                j += 1
            anomaly_segments.append((i, j))
            i = j
        else:
            i += 1
    
    if not anomaly_segments:
        return 0.0
    
    segment_max_scores = []
    segment_lengths = []
    for start, end in anomaly_segments:
        segment_max_scores.append(np.max(scores[start:end]))
        segment_lengths.append(end - start)
    
    segment_max_scores = np.array(segment_max_scores)
    segment_lengths = np.array(segment_lengths)
    
    normal_scores = np.sort(scores[labels == 0])
    n_normal = len(normal_scores)
    
    for thresh in thresholds:
        detected_segments_mask = (segment_max_scores > thresh)
        tp = np.sum(segment_lengths[detected_segments_mask])
        fn = np.sum(segment_lengths[~detected_segments_mask])
        fp_index = np.searchsorted(normal_scores, thresh, side='right')
        fp = n_normal - fp_index
        
        if tp > 0:
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1 = 2 * (p * r) / (p + r + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
    
    return best_f1


def load_data(dataset_name, entity, split, data_dir='processed'):
    """Load preprocessed data"""
    file_path = os.path.join(data_dir, dataset_name, f'{entity}_{split}.npy')
    return np.load(file_path)


def get_available_entities(dataset_name, data_dir='processed'):
    """Get list of available entities for a dataset"""
    dataset_dir = os.path.join(data_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        return []
    
    files = os.listdir(dataset_dir)
    entities = set()
    for f in files:
        if f.endswith('_train.npy'):
            entity = f.replace('_train.npy', '')
            entities.add(entity)
    
    return sorted(list(entities))


def run_benchmark(dataset_name, entities=None,
                  window=96, windows_per_cycle=7,
                  sample_rate=0.3, cluster_threshold=0.15,
                  workers=4, random_state=42):
    """
    Run JumpStarter benchmark on a dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., 'SMD', 'MSL', 'SMAP')
        entities: List of entities to test (None = all)
        window: Reconstruction window size
        windows_per_cycle: Number of windows per cycle
        sample_rate: Sampling rate for compressive sensing
        cluster_threshold: Feature clustering threshold
        workers: Number of parallel workers
        random_state: Random seed
    """
    print(f"\n{'='*80}")
    print(f"JUMPSTARTER BENCHMARK: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Get entities
    all_entities = get_available_entities(dataset_name)
    if not all_entities:
        print(f"Error: No data found for {dataset_name}")
        return None
    
    if entities:
        test_entities = [e for e in entities if e in all_entities]
    else:
        test_entities = all_entities
    
    print(f"Testing {len(test_entities)} entities\n")
    print(f"Parameters:")
    print(f"  Window: {window}")
    print(f"  Windows/Cycle: {windows_per_cycle}")
    print(f"  Sample Rate: {sample_rate}")
    print(f"  Cluster Threshold: {cluster_threshold}")
    print(f"  Workers: {workers}\n")
    
    results = []
    
    for idx, entity in enumerate(test_entities):
        print(f"  [{idx+1}/{len(test_entities)}] {entity}...", end=' ', flush=True)
        
        try:
            # Load data
            train = load_data(dataset_name, entity, 'train')
            test = load_data(dataset_name, entity, 'test')
            labels = load_data(dataset_name, entity, 'labels')
            
            if len(labels.shape) > 1:
                labels = labels[:, 0]
            
            # Normalize
            n, d = train.shape
            for i in range(d):
                train[:, i] = normalization(train[:, i])
                test[:, i] = normalization(test[:, i])
            
            # Create detector
            detector = CSAnomalyDetector(
                workers=workers,
                cluster_threshold=cluster_threshold,
                sample_rate=sample_rate,
                sample_score_method=lesinn_score,
                distance=anomaly_score_example,
                scale=5.0,
                rho=0.1,
                sigma=1/24,
                random_state=random_state,
                retry_limit=10,
                without_grouping=None,
                without_localize_sampling=False,
                latest_windows=96
            )
            
            # Train (reconstruct) - use stride=50 for reasonable speed
            t0 = time.time()
            train_rec, retries = detector.reconstruct(
                train, window=window,
                windows_per_cycle=windows_per_cycle,
                stride=50
            )
            
            # Detect on test - use stride=50 for reasonable speed
            test_rec, _ = detector.reconstruct(
                test, window=window,
                windows_per_cycle=windows_per_cycle,
                stride=50
            )
            
            scores = detector.predict(
                test, test_rec,
                window=10, stride=50
            )
            train_time = time.time() - t0
            
            # Evaluate
            f1 = calc_point_adjusted_f1(scores, labels)
            
            print(f"Time: {train_time:.2f}s | F1: {f1:.4f} | Retries: {retries}")
            
            results.append({
                'Dataset': dataset_name,
                'Entity': entity,
                'TrainTime': train_time,
                'F1': f1,
                'Retries': retries
            })
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({
                'Dataset': dataset_name,
                'Entity': entity,
                'TrainTime': 0.0,
                'F1': 0.0,
                'Retries': 0,
                'Error': str(e)
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='JumpStarter Standalone Benchmark')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                       help='Dataset names (e.g., SMD MSL SMAP)')
    parser.add_argument('--entities', type=str, nargs='+',
                       help='Specific entities to test (default: all)')
    parser.add_argument('--window', type=int, default=96,
                       help='Reconstruction window size (default: 96)')
    parser.add_argument('--windows_per_cycle', type=int, default=7,
                       help='Windows per cycle (default: 7)')
    parser.add_argument('--sample_rate', type=float, default=0.4,
                       help='Sampling rate (default: 0.4)')
    parser.add_argument('--cluster_threshold', type=float, default=0.15,
                       help='Cluster threshold (default: 0.15)')
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of workers (default: 16)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    all_results = []
    
    for dataset in args.datasets:
        results = run_benchmark(
            dataset,
            entities=args.entities,
            window=args.window,
            windows_per_cycle=args.windows_per_cycle,
            sample_rate=args.sample_rate,
            cluster_threshold=args.cluster_threshold,
            workers=args.workers,
            random_state=args.random_state
        )
        
        if results:
            all_results.extend(results)
    
    if not all_results:
        print("No results generated!")
        return
    
    # Save results
    df = pd.DataFrame(all_results)
    
    # Create results directory if it doesn't exist
    results_dir = 'results/csv'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets_str = '_'.join(args.datasets)
    csv_path = f"{results_dir}/jumpstarter_{datasets_str}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"RESULTS SAVED: {csv_path}")
    print(f"={'='*80}\n")
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    summary = df.groupby('Dataset').agg({
        'F1': 'mean',
        'TrainTime': 'mean',
        'Retries': 'mean'
    }).round(4)
    summary.columns = ['Avg_F1', 'Avg_Time', 'Avg_Retries']
    print(summary.to_string())
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
