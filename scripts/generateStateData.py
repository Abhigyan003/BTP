#!/usr/bin/env python3
"""
generateStateData.py

Generate synthetic datasets driven by hidden states (Markov Chain) rather than strict periodicity.
This is designed to test the 'Entropy' based transfer learning, which should adapt better to 
changing complexity (states) than the 'Periodic' baseline.

States:
0. Stable: Low noise, constant baseline.
1. Volatile: High noise, random walk.
2. Drifting: Linear trend (up or down).
3. Oscillating: High frequency sine wave.

Output: gzipped CSV files, rows: timestamp,entity_id,metric_0..metric_{M-1},label
"""

import os
import gzip
import csv
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def make_timestamps(start_date, n_steps, delta_minutes=5):
    start = start_date
    return [(start + timedelta(minutes=delta_minutes * i)).isoformat() for i in range(n_steps)]

def generate_markov_states(n_steps, transition_matrix, rng):
    """
    Generate a sequence of states based on a transition matrix.
    """
    states = np.zeros(n_steps, dtype=int)
    current_state = 0
    for t in range(1, n_steps):
        probs = transition_matrix[current_state]
        current_state = rng.choice(len(probs), p=probs)
        states[t] = current_state
    return states

def generate_entity_state_data(n_metrics, n_steps, states, rng):
    """
    Generate data based on states that are Low Entropy (predictable) but Low Periodicity.
    This favors Entropy-based weighting over Periodic-based weighting.
    
    States:
    0: Stable (Low Entropy)
    1: Volatile (High Entropy)
    2: Oscillating (Periodic)
    """
    data = np.zeros((n_steps, n_metrics), dtype=float)
    
    # Base values
    current_values = rng.uniform(0, 10, size=n_metrics)
    
    for t in range(n_steps):
        state = states[t]
        
        if state == 0: # Stable (Low Entropy)
            noise = rng.normal(0, 0.1, size=n_metrics)
            # Target value around 0
            target = np.zeros(n_metrics)
            current_values = 0.9 * current_values + 0.1 * target + noise
            
        elif state == 1: # Volatile (High Entropy)
            noise = rng.normal(0, 5.0, size=n_metrics) # High noise
            # Target value around 0 (same mean as Stable, but high variance)
            target = np.zeros(n_metrics)
            current_values = 0.5 * current_values + 0.5 * target + noise
            
        elif state == 2: # Oscillating (Periodic)
            noise = rng.normal(0, 0.1, size=n_metrics)
            # Sine wave
            osc = np.sin(2 * np.pi * 0.05 * t) * 5.0
            current_values = osc + noise
            
        elif state == 3: # Unused/Same as Stable
             noise = rng.normal(0, 0.1, size=n_metrics)
             current_values = noise
            
        data[t] = current_values

    return data

def inject_anomalies(data, labels, anomaly_rate=0.05, rng=None):
    """
    Inject anomalies (spikes) independent of state.
    """
    n_steps, n_metrics = data.shape
    n_anom = int(n_steps * anomaly_rate)
    
    idxs = rng.choice(n_steps, size=n_anom, replace=False)
    
    for idx in idxs:
        # Global spike
        mag = rng.uniform(10, 20) * rng.choice([-1, 1])
        data[idx] += mag
        labels[idx] = 1
        
    return data, labels

def stream_generate(n_entities, n_metrics, total_days, out_path, seed=42, delta_minutes=5):
    rng = np.random.default_rng(seed)
    samples_per_day = int(1440 // delta_minutes)
    n_steps = total_days * samples_per_day
    start_time = datetime(2023, 1, 1)
    timestamps = make_timestamps(start_time, n_steps, delta_minutes)
    
    header = ['timestamp', 'entity_id'] + [f'metric_{i}' for i in range(n_metrics)] + ['label']
    
    # Transition Matrix (4 states)
    # High probability to stay in same state, low prob to switch
    # 0: Stable, 1: Volatile, 2: Drifting, 3: Oscillating
    T = np.array([
        [0.95, 0.02, 0.02, 0.01], # From Stable
        [0.05, 0.90, 0.05, 0.00], # From Volatile
        [0.02, 0.02, 0.95, 0.01], # From Drifting
        [0.05, 0.00, 0.05, 0.90], # From Oscillating
    ])
    
    with gzip.open(out_path, 'wt', newline='') as gzfile:
        writer = csv.writer(gzfile)
        writer.writerow(header)
        
        for e in range(n_entities):
            states = generate_markov_states(n_steps, T, rng)
            data = generate_entity_state_data(n_metrics, n_steps, states, rng)
            labels = np.zeros(n_steps, dtype=int)
            
            # Inject anomalies
            data, labels = inject_anomalies(data, labels, anomaly_rate=0.05, rng=rng)
            
            # Normalize
            mins = data.min(axis=0)
            maxs = data.max(axis=0)
            denom = maxs - mins
            denom[denom == 0] = 1.0
            data = (data - mins) / denom
            
            entity_id = f'entity_{e:04d}'
            for t in range(n_steps):
                row = [timestamps[t], entity_id] + [float(x) for x in data[t].tolist()] + [int(labels[t])]
                writer.writerow(row)
                
            if (e + 1) % 20 == 0:
                print(f"Generated {e+1}/{n_entities} entities...")
                gzfile.flush()
                
    return {
        'file': out_path,
        'n_entities': n_entities,
        'n_metrics': n_metrics,
        'n_steps': n_steps
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='processed/raw_synthetic')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Generate State_DS1
    # 100 entities, 10 metrics, 10 days
    out_path1 = os.path.join(args.outdir, 'State_DS1_synthetic.csv.gz')
    print(f"Generating State_DS1 to {out_path1}...")
    stream_generate(100, 10, 10, out_path1, seed=args.seed)
    print("State_DS1 Done.")

    # Generate State_DS2
    # 400 entities, 50 metrics, 22 days
    out_path2 = os.path.join(args.outdir, 'State_DS2_synthetic.csv.gz')
    print(f"Generating State_DS2 to {out_path2}...")
    stream_generate(400, 50, 22, out_path2, seed=args.seed + 100)
    print("State_DS2 Done.")

if __name__ == "__main__":
    main()
