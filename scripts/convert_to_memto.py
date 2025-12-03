"""
Data Converter: code/processed → MEMTO format

Converts per-entity .npy files from code/processed/{dataset}/
to MEMTO's expected combined format in MEMTO/data/{dataset}/{dataset}/
"""

import numpy as np
import os
import argparse


def convert_dataset_to_memto(dataset_name, processed_dir='processed', memto_data_dir='MEMTO/MEMTO/data'):
    """
    Convert per-entity dataset to MEMTO format.
    
    MEMTO expects:
        - {dataset}_train.npy: shape (n_total_timesteps, n_features)
        - {dataset}_test.npy: shape (n_total_timesteps, n_features)
        - {dataset}_test_label.npy: shape (n_total_timesteps,)
    
    We have:
        - {entity}_train.npy, {entity}_test.npy, {entity}_labels.npy per entity
    
    Strategy: Combine all entities by concatenating along timestep axis
    """
    print(f"\n{'='*80}")
    print(f"Converting {dataset_name} to MEMTO format")
    print(f"{'='*80}\n")
    
    # Get list of entities
    dataset_dir = os.path.join(processed_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Directory not found: {dataset_dir}")
        return False
    
    files = os.listdir(dataset_dir)
    entities = set()
    for f in files:
        if f.endswith('_train.npy'):
            entity = f.replace('_train.npy', '')
            entities.add(entity)
    
    entities = sorted(list(entities))
    print(f"Found {len(entities)} entities: {entities[:5]}{'...' if len(entities) > 5 else ''}\n")
    
    # Combine data from all entities
    all_train = []
    all_test = []
    all_labels = []
    
    for entity in entities:
        print(f"  Loading {entity}...", end=' ')
        
        train_file = os.path.join(dataset_dir, f'{entity}_train.npy')
        test_file = os.path.join(dataset_dir, f'{entity}_test.npy')
        label_file = os.path.join(dataset_dir, f'{entity}_labels.npy')
        
        train = np.load(train_file)
        test = np.load(test_file)
        labels = np.load(label_file)
        
        # Handle labels shape
        if len(labels.shape) > 1:
            labels = labels[:, 0]
        
        print(f"train: {train.shape}, test: {test.shape}, labels: {labels.shape}")
        
        all_train.append(train)
        all_test.append(test)
        all_labels.append(labels)
    
    # Concatenate along timestep axis (axis=0)
    combined_train = np.concatenate(all_train, axis=0)
    combined_test = np.concatenate(all_test, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nCombined shapes:")
    print(f"  Train: {combined_train.shape}")
    print(f"  Test: {combined_test.shape}")
    print(f"  Labels: {combined_labels.shape}")
    
    # Create MEMTO data directory
    output_dir = os.path.join(memto_data_dir, dataset_name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in MEMTO format
    train_path = os.path.join(output_dir, f'{dataset_name}_train.npy')
    test_path = os.path.join(output_dir, f'{dataset_name}_test.npy')
    label_path = os.path.join(output_dir, f'{dataset_name}_test_label.npy')
    
    np.save(train_path, combined_train)
    np.save(test_path, combined_test)
    np.save(label_path, combined_labels)
    
    print(f"\n✓ Saved to:")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print(f"  {label_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert datasets to MEMTO format')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                       help='Dataset names to convert (e.g., SMD MSL SMAP)')
    parser.add_argument('--processed_dir', type=str, default='processed',
                       help='Source directory with per-entity data (default: processed)')
    parser.add_argument('--memto_data_dir', type=str, default='MEMTO/MEMTO/data',
                       help='MEMTO data directory (default: MEMTO/MEMTO/data)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("DATA CONVERTER: code/processed → MEMTO format")
    print(f"{'='*80}")
    
    success_count = 0
    for dataset in args.datasets:
        if convert_dataset_to_memto(dataset, args.processed_dir, args.memto_data_dir):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"CONVERSION COMPLETE: {success_count}/{len(args.datasets)} datasets successful")
    print(f"{'='*80}\n")
    
    if success_count == len(args.datasets):
        print("✓ All datasets ready for MEMTO benchmark!")
        print("\nYou can now run:")
        print(f"  python scripts/memto_benchmark.py --datasets {' '.join(args.datasets)}")
    else:
        print("⚠ Some datasets failed to convert")


if __name__ == "__main__":
    main()
