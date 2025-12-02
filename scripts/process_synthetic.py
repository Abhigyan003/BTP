import pandas as pd
import numpy as np
import os
import gzip
import argparse
from tqdm import tqdm

def process_dataset(input_path, output_dir, train_days, test_days):
    print(f"Processing {input_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Read compressed CSV
    # Use chunksize to handle large files if needed, but for 400 entities it might fit in memory
    # Let's read in chunks to be safe
    
    chunksize = 100000
    chunks = []
    with gzip.open(input_path, 'rt') as f:
        for chunk in tqdm(pd.read_csv(f, chunksize=chunksize), desc="Reading CSV"):
            chunks.append(chunk)
            
    df = pd.concat(chunks, ignore_index=True)
    
    # Get unique entities
    entities = df['entity_id'].unique()
    print(f"Found {len(entities)} entities.")
    
    samples_per_day = 288 # 5 min interval
    train_len = train_days * samples_per_day
    
    for entity in tqdm(entities, desc="Saving Entities"):
        entity_df = df[df['entity_id'] == entity].sort_values('timestamp')
        
        # Extract metrics and labels
        # Columns: timestamp, entity_id, metric_0...metric_M, label
        metric_cols = [c for c in entity_df.columns if c.startswith('metric_')]
        data = entity_df[metric_cols].values
        labels = entity_df['label'].values
        
        # Split
        if len(data) < train_len:
            print(f"Warning: Entity {entity} has insufficient data ({len(data)} < {train_len}). Skipping.")
            continue
            
        train_data = data[:train_len]
        test_data = data[train_len:]
        test_labels = labels[train_len:]
        
        # Save
        # Format: processed/{Dataset}/entity_{id}_{mode}.npy
        # entity name is already 'entity_XXXX'
        
        np.save(os.path.join(output_dir, f"{entity}_train.npy"), train_data)
        np.save(os.path.join(output_dir, f"{entity}_test.npy"), test_data)
        np.save(os.path.join(output_dir, f"{entity}_labels.npy"), test_labels)
        
    print(f"Saved processed files to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='processed/raw_synthetic')
    parser.add_argument('--out_dir', type=str, default='processed')
    args = parser.parse_args()
    
    # Dataset 1
    ds1_in = os.path.join(args.raw_dir, 'Dataset1_synthetic.csv.gz')
    ds1_out = os.path.join(args.out_dir, 'Synthetic_DS1')
    if os.path.exists(ds1_in):
        # DS1: 10 days total. 8 days train (7 base + 1 transfer), 2 days test
        process_dataset(ds1_in, ds1_out, train_days=8, test_days=2)
    else:
        print(f"Skipping Dataset1 (not found at {ds1_in})")
        
    # Dataset 2
    ds2_in = os.path.join(args.raw_dir, 'Dataset2_synthetic.csv.gz')
    ds2_out = os.path.join(args.out_dir, 'Synthetic_DS2')
    if os.path.exists(ds2_in):
        # DS2: 22 days total. 15 days train (14 base + 1 transfer), 7 days test
        process_dataset(ds2_in, ds2_out, train_days=15, test_days=7)
    else:
        print(f"Skipping Dataset2 (not found at {ds2_in})")

    # State_DS1
    state_ds1_in = os.path.join(args.raw_dir, 'State_DS1_synthetic.csv.gz')
    state_ds1_out = os.path.join(args.out_dir, 'State_DS1')
    if os.path.exists(state_ds1_in):
        # State_DS1: 10 days total. 8 days train, 2 days test
        process_dataset(state_ds1_in, state_ds1_out, train_days=8, test_days=2)
    else:
        print(f"Skipping State_DS1 (not found at {state_ds1_in})")

    # State_DS2
    state_ds2_in = os.path.join(args.raw_dir, 'State_DS2_synthetic.csv.gz')
    state_ds2_out = os.path.join(args.out_dir, 'State_DS2')
    if os.path.exists(state_ds2_in):
        # State_DS2: 22 days total. 15 days train, 7 days test (similar to DS2)
        process_dataset(state_ds2_in, state_ds2_out, train_days=15, test_days=7)
    else:
        print(f"Skipping State_DS2 (not found at {state_ds2_in})")

if __name__ == "__main__":
    main()
