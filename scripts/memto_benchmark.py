"""
MEMTO Standalone Benchmark Script

Evaluates MEMTO (Memory-Enhanced Multivariate Time Series Anomaly Detection)
on multiple datasets from the processed directory.

MEMTO requires 2-phase training:
1. First train: Train transformer with random memory initialization
2. Memory initial: Initialize memory items using k-means clustering
3. Second train: Fine-tune with initialized memory items
4. Test: Evaluate on test data

This script handles all phases automatically for each dataset.
"""

import sys
import os

# Add MEMTO to path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(script_dir)  # code/
memto_dir = os.path.join(code_dir, 'MEMTO', 'MEMTO')
sys.path.insert(0, memto_dir)

import numpy as np
import pandas as pd
import time
import argparse
from datetime import datetime
import torch
from torch.backends import cudnn

from utils.utils import mkdir
from solver import Solver


# Dataset-specific configurations (from test.sh)
DATASET_CONFIGS = {
    'SMD': {
        'input_c': 38,
        'output_c': 38,
        'batch_size': 256,
        'anormly_ratio': 0.5,
        'd_model': 512
    },
    'SMAP': {
        'input_c': 25,
        'output_c': 25,
        'batch_size': 256,
        'anormly_ratio': 1.0,
        'd_model': 512
    },
    'MSL': {
        'input_c': 55,
        'output_c': 55,
        'batch_size': 8,
        'anormly_ratio': 1.0,
        'd_model': 64
    },
    'PSM': {
        'input_c': 25,
        'output_c': 25,
        'batch_size': 256,
        'anormly_ratio': 1.0,
        'd_model': 512
    },
    'SWaT': {
        'input_c': 51,
        'output_c': 51,
        'batch_size': 256,
        'anormly_ratio': 0.1,
        'd_model': 512
    }
}


def run_memto_benchmark(dataset_name, num_epochs_train=100, num_epochs_memory=100,
                        win_size=100, n_memory=10, lambd=0.01,
                        lr_first=1e-4, lr_second=5e-5,
                        device='cuda:0', random_state=42):
    """
    Run full MEMTO training and testing pipeline for a dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., 'SMD', 'MSL', 'SMAP')
        num_epochs_train: Epochs for first training phase
        num_epochs_memory: Epochs for memory initialization phase
        win_size: Window size for sequences
        n_memory: Number of memory items
        lambd: Lambda parameter for entropy loss
        lr_first: Learning rate for first training
        lr_second: Learning rate for second training
        device: CUDA device
        random_state: Random seed
    
    Returns:
        dict with results (accuracy, precision, recall, f1_score, train_time)
    """
    print(f"\n{'='*80}")
    print(f"MEMTO BENCHMARK: {dataset_name}")
    print(f"={'='*80}\n")
    
    # Get dataset-specific config
    if dataset_name not in DATASET_CONFIGS:
        print(f"ERROR: No configuration found for {dataset_name}")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        return None
    
    config = DATASET_CONFIGS[dataset_name]
    
    # Prepare data path - MEMTO expects data in specific format
    # We need to convert from code/processed/{dataset}/*.npy 
    # to MEMTO/data/{dataset}/{dataset}/ format
    data_path = os.path.join(memto_dir, 'data', dataset_name, dataset_name + '/')
    
    if not os.path.exists(data_path):
        print(f"ERROR: MEMTO data not found at {data_path}")
        print(f"Please prepare MEMTO data format first")
        return None
    
    print(f"Parameters:")
    print(f"  Input/Output Channels: {config['input_c']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Window Size: {win_size}")
    print(f"  Memory Items: {n_memory}")
    print(f"  Epochs (Train/Memory): {num_epochs_train}/{num_epochs_memory}")
    print(f"  Learning Rates: {lr_first}/{lr_second}")
    print(f"  Device: {device}\n")
    
    # Set random seeds
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    cudnn.benchmark = True
    
    model_save_path = f'checkpoints_{dataset_name}'
    if not os.path.exists(model_save_path):
        mkdir(model_save_path)
    
    total_start_time = time.time()
    
    try:
        # PHASE 1: First Training (random memory initialization)
        print("="*80)
        print("PHASE 1: First Training (Random Memory Initialization)")
        print("="*80)
        
        config_dict = {
            'lr': lr_first,
            'num_epochs': num_epochs_train,
            'k': 5,
            'win_size': win_size,
            'input_c': config['input_c'],
            'output_c': config['output_c'],
            'batch_size': config['batch_size'],
            'temp_param': 0.05,
            'lambd': lambd,
            'pretrained_model': None,
            'dataset': dataset_name,
            'mode': 'train',
            'data_path': data_path,
            'model_save_path': model_save_path,
            'anormly_ratio': config['anormly_ratio'],
            'device': device,
            'n_memory': n_memory,
            'num_workers': 4 * torch.cuda.device_count() if torch.cuda.is_available() else 4,
            'd_model': config['d_model'],
            'temperature': 0.1,
            'memory_initial': False,
            'phase_type': None
        }
        
        solver = Solver(config_dict)
        solver.train(training_type='first_train')
        
        # PHASE 2: Memory Initialization
        print("\n" + "="*80)
        print("PHASE 2: Memory Initialization (K-Means Clustering)")
        print("="*80)
        
        config_dict.update({
            'lr': lr_second,
            'num_epochs': num_epochs_memory,
            'mode': 'memory_initial',
            'memory_initial': True,
            'phase_type': 'second_train'
        })
        
        solver = Solver(config_dict)
        solver.get_memory_initial_embedding(training_type='second_train')
        
        # PHASE 3: Testing
        print("\n" + "="*80)
        print("PHASE 3: Testing")
        print("="*80)
        
        config_dict.update({
            'mode': 'test',
            'memory_initial': False,
            'phase_type': 'test'
        })
        
        solver = Solver(config_dict)
        accuracy, precision, recall, f1_score = solver.test()
        
        total_time = time.time() - total_start_time
        
        print(f"\n{'='*80}")
        print(f"RESULTS for {dataset_name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1_score:.4f}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"={'='*80}\n")
        
        return {
            'Dataset': dataset_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1_score,
            'TotalTime': total_time
        }
        
    except Exception as e:
        print(f"\nERROR during {dataset_name} benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'Dataset': dataset_name,
            'Accuracy': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'TotalTime': 0.0,
            'Error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='MEMTO Standalone Benchmark')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                       help='Dataset names (e.g., SMD MSL SMAP)')
    parser.add_argument('--num_epochs_train', type=int, default=100,
                       help='Epochs for first training (default: 100)')
    parser.add_argument('--num_epochs_memory', type=int, default=100,
                       help='Epochs for memory initialization (default: 100)')
    parser.add_argument('--win_size', type=int, default=100,
                       help='Window size (default: 100)')
    parser.add_argument('--n_memory', type=int, default=10,
                       help='Number of memory items (default: 10)')
    parser.add_argument('--lambd', type=float, default=0.01,
                       help='Lambda for entropy loss (default: 0.01)')
    parser.add_argument('--lr_first', type=float, default=1e-4,
                       help='Learning rate for first training (default: 1e-4)')
    parser.add_argument('--lr_second', type=float, default=5e-5,
                       help='Learning rate for second training (default: 5e-5)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device (default: cuda:0)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    all_results = []
    
    for dataset in args.datasets:
        result = run_memto_benchmark(
            dataset,
            num_epochs_train=args.num_epochs_train,
            num_epochs_memory=args.num_epochs_memory,
            win_size=args.win_size,
            n_memory=args.n_memory,
            lambd=args.lambd,
            lr_first=args.lr_first,
            lr_second=args.lr_second,
            device=args.device,
            random_state=args.random_state
        )
        
        if result:
            all_results.append(result)
    
    if not all_results:
        print("No results generated!")
        return
    
    # Save results
    df = pd.DataFrame(all_results)
    
    # Create results directory
    results_dir = 'results/csv'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets_str = '_'.join(args.datasets)
    csv_path = f"{results_dir}/memto_{datasets_str}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"RESULTS SAVED: {csv_path}")
    print(f"={'='*80}\n")
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(df[['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1', 'TotalTime']].to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
