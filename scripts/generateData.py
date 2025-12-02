#!/usr/bin/env python3
"""
generate_omni_datasets.py

Generate synthetic datasets that match "Dataset1" and "Dataset2" specs from
the OmniTransfer paper:
  - Dataset1: 400 entities, 19 metrics, 10 days total (7d base +1d transfer +2d test)
  - Dataset2: 400 entities, 25 metrics, 22 days total (14d base +1d transfer +7d test)

Sampling: 5-minute interval -> 288 samples/day.
Outputs: gzipped CSV files, rows: timestamp,entity_id,metric_0..metric_{M-1},label

Usage:
  python3 generate_omni_datasets.py --outdir ./omni_synthetic --workers 1

Notes:
 - Script streams rows to gzip to avoid large memory usage.
 - It injects mixed anomalies (global spike, contextual, pattern shift, frequency, trend).
 - Per-metric "periodicity strength" and "phase shifts" are included to emulate realistic variety.
 - Use a machine with sufficient disk space for full datasets (compressed files will be several GB).
"""

import os
import gzip
import csv
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# -------------------------
# Helper / Generator logic
# -------------------------

def make_timestamps(start_date, n_steps, delta_minutes=5):
    start = start_date
    return [(start + timedelta(minutes=delta_minutes * i)).isoformat() for i in range(n_steps)]

def generate_entity_array(n_metrics, n_steps, period_strengths, phase_shifts, trend_scale=0.0, noise_scale=0.05, rng=None):
    """
    Create a (n_steps, n_metrics) array for one entity.
    - period_strengths: array in [0,1] length n_metrics, larger -> more periodic
    - phase_shifts: integer array of sample shifts (mod samples_per_day)
    - trend_scale: small float for per-entity trend amplitude
    """
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(n_steps)
    data = np.zeros((n_steps, n_metrics), dtype=float)
    samples_per_day = 288  # 5-minute sampling

    for m in range(n_metrics):
        p = float(period_strengths[m])
        phase = int(phase_shifts[m] % samples_per_day)
        # daily + weekly sinusoidal components
        daily = np.sin(2.0 * np.pi * (t + phase) / samples_per_day)
        weekly = np.sin(2.0 * np.pi * (t + phase) / (samples_per_day * 7))
        seasonal = p * (0.7 * daily + 0.3 * weekly)

        # low-frequency trend
        trend = (trend_scale * (t / float(max(1, n_steps)) ) ) * (rng.normal() * 0.5 + 0.5)

        # aperiodic (smoothed noise) when p small
        ar = 0.0
        if p < 0.4:
            noise = rng.normal(scale=0.6, size=n_steps)
            # simple smoothing to make aperiodic behavior
            ar = pd.Series(noise).rolling(window=12, min_periods=1, center=True).mean().to_numpy()

        amp = 1.0 + 0.4 * rng.normal()
        data[:, m] = amp * (seasonal + ar) + trend
        data[:, m] += noise_scale * rng.normal(size=n_steps)
    return data

def inject_anomalies_array(data, anomaly_rate=0.055, rng=None):
    """
    Modify `data` in-place to insert anomalies. Return per-timestep binary labels (1 if any metric anomalous).
    Types: global_spike, contextual (single metric), pattern_shift (longer), frequency, trend
    """
    if rng is None:
        rng = np.random.default_rng()
    n_steps, n_metrics = data.shape
    labels = np.zeros(n_steps, dtype=int)
    target_anom_points = max(1, int(round(n_steps * anomaly_rate)))
    inserted = 0
    attempts = 0

    # choose segments until we reach target anomaly points (avoid infinite loop)
    while inserted < target_anom_points and attempts < target_anom_points * 10:
        attempts += 1
        start = int(rng.integers(0, n_steps))
        length = int(rng.choice([1,2,3,5,10,20,30], p=[0.45,0.2,0.15,0.1,0.06,0.03,0.01]))
        end = min(n_steps, start + length)
        metric = int(rng.integers(0, n_metrics))
        typ = rng.choice(['global_spike','contextual','pattern_shift','frequency','trend'], p=[0.25,0.25,0.2,0.15,0.15])

        if typ == 'global_spike':
            mag = float(max(1.5, rng.normal(5.0,2.0)))
            # apply spike to many metrics
            data[start:end, :] += mag * (1.0 + 0.2 * rng.normal(size=(end-start, n_metrics)))
            labels[start:end] = 1
            inserted += (end - start)

        elif typ == 'contextual':
            mag = float(max(1.0, rng.normal(4.0,1.2)))
            data[start:end, metric] += mag * (1.0 + 0.3 * rng.normal(size=(end-start)))
            labels[start:end] = 1
            inserted += (end - start)

        elif typ == 'pattern_shift':
            mag = float(max(0.8, rng.normal(1.5,0.5)))
            seglen = end - start
            if seglen <= 0:
                continue
            # add a slow shaped perturbation (sin-based)
            data[start:end, metric] += mag * np.sin(np.linspace(0, np.pi * max(1, seglen)/5.0, seglen))
            labels[start:end] = 1
            inserted += seglen

        elif typ == 'frequency':
            freq = int(rng.choice([6,12,24]))  # higher frequency in samples
            mag = float(max(0.4, rng.normal(1.0,0.4)))
            idx = np.arange(start, end)
            if start < end:
                data[start:end, metric] += mag * np.sin(2.0 * np.pi * idx / float(freq))
                labels[start:end] = 1
                inserted += (end - start)

        elif typ == 'trend':
            mag = float(max(0.8, rng.normal(2.0,0.6)))
            seglen = end - start
            if seglen <= 0:
                continue
            data[start:end, metric] += mag * np.linspace(0.0, 1.0, seglen)
            labels[start:end] = 1
            inserted += seglen

    return labels

# -------------------------
# Streaming writer
# -------------------------

def stream_generate(n_entities, n_metrics, total_days, out_path, seed=42,
                    periodic_metric_ratio=0.6, anomaly_rate=0.055, delta_minutes=5):
    """
    Stream-generate a gzipped CSV for the dataset.
    CSV columns: timestamp, entity_id, metric_0 ... metric_{M-1}, label
    """
    rng = np.random.default_rng(seed)
    samples_per_day = int(1440 // delta_minutes)
    n_steps = total_days * samples_per_day
    start_time = datetime(2023, 1, 1)
    timestamps = make_timestamps(start_time, n_steps, delta_minutes)

    header = ['timestamp', 'entity_id'] + [f'metric_{i}' for i in range(n_metrics)] + ['label']

    # open gz writer
    with gzip.open(out_path, 'wt', newline='') as gzfile:
        writer = csv.writer(gzfile)
        writer.writerow(header)

        for e in range(n_entities):
            # per-entity randomized periodic strengths
            period_strengths = rng.uniform(0.0, 1.0, size=n_metrics)
            # enforce ratio of strongly periodic metrics
            n_strong = max(1, int(round(periodic_metric_ratio * n_metrics)))
            strong_idx = rng.choice(n_metrics, size=n_strong, replace=False)
            # boost strong indices
            period_strengths[strong_idx] = rng.uniform(0.7, 1.0, size=len(strong_idx))
            # weaker for others
            for i in range(n_metrics):
                if i not in strong_idx:
                    period_strengths[i] = rng.uniform(0.0, 0.6)

            # phase shifts (up to one day)
            phase_shifts = rng.integers(0, samples_per_day, size=n_metrics)

            trend_scale = float(rng.uniform(-0.4, 0.4))

            # generate data and labels
            data = generate_entity_array(n_metrics, n_steps, period_strengths, phase_shifts,
                                         trend_scale=trend_scale, noise_scale=0.10, rng=rng)
            labels = inject_anomalies_array(data, anomaly_rate=anomaly_rate, rng=rng)

            # per-metric normalization per-entity to [0,1] for stability (like paper preprocessing)
            mins = data.min(axis=0)
            maxs = data.max(axis=0)
            denom = (maxs - mins)
            denom[denom == 0.0] = 1.0
            data = (data - mins) / denom

            entity_id = f'entity_{e:04d}'
            # stream write rows
            for t in range(n_steps):
                row = [timestamps[t], entity_id] + [float(x) for x in data[t].tolist()] + [int(labels[t])]
                writer.writerow(row)

            # flush occasionally
            if (e + 1) % 20 == 0:
                gzfile.flush()

    return {
        'file': out_path,
        'n_entities': n_entities,
        'n_metrics': n_metrics,
        'n_steps': n_steps,
    }

# -------------------------
# CLI / Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate OmniTransfer-like synthetic datasets.")
    parser.add_argument('--outdir', type=str, default='./omni_synthetic', help='output directory')
    parser.add_argument('--workers', type=int, default=1, help='not used (single-threaded streaming)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--delta_minutes', type=int, default=5, help='sampling minutes (default 5 -> 288/day)')
    parser.add_argument('--preview', action='store_true', help='generate small preview files instead of full')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.preview:
        # Small preview for quick inspection: 5 entities, few days
        ds1_entities = 5; ds1_metrics = 19; ds1_days = 2
        ds2_entities = 5; ds2_metrics = 25; ds2_days = 3
        print("Generating previews...")
        p1 = os.path.join(args.outdir, 'Dataset1_preview.csv.gz')
        p2 = os.path.join(args.outdir, 'Dataset2_preview.csv.gz')
        meta1 = stream_generate(ds1_entities, ds1_metrics, ds1_days, p1, seed=args.seed,
                                periodic_metric_ratio=0.6, anomaly_rate=0.055, delta_minutes=args.delta_minutes)
        meta2 = stream_generate(ds2_entities, ds2_metrics, ds2_days, p2, seed=args.seed+1,
                                periodic_metric_ratio=0.55, anomaly_rate=0.052, delta_minutes=args.delta_minutes)
        print("Previews written to:", p1, p2)
        print("Meta1:", meta1)
        print("Meta2:", meta2)
        return

    # Full sizes to match paper's Table 3
    # Dataset1: 400 entities, 19 metrics, 10 days (7 base +1 transfer +2 test)
    ds1_entities = 400; ds1_metrics = 19; ds1_days = 10
    # Dataset2: 400 entities, 25 metrics, 22 days (14 base +1 transfer +7 test)
    ds2_entities = 400; ds2_metrics = 25; ds2_days = 22

    ds1_path = os.path.join(args.outdir, 'Dataset1_synthetic.csv.gz')
    ds2_path = os.path.join(args.outdir, 'Dataset2_synthetic.csv.gz')

    print("Generating Dataset1 ->", ds1_path)
    meta1 = stream_generate(ds1_entities, ds1_metrics, ds1_days, ds1_path, seed=args.seed,
                            periodic_metric_ratio=0.60, anomaly_rate=0.055, delta_minutes=args.delta_minutes)
    print("Dataset1 done:", meta1)

    print("Generating Dataset2 ->", ds2_path)
    meta2 = stream_generate(ds2_entities, ds2_metrics, ds2_days, ds2_path, seed=args.seed+123,
                            periodic_metric_ratio=0.55, anomaly_rate=0.052, delta_minutes=args.delta_minutes)
    print("Dataset2 done:", meta2)

    # Write small metadata overview file
    overview = {
        'Dataset1': meta1,
        'Dataset2': meta2,
        'notes': 'Timestamps are ISO8601. Labels are binary per time point (1 = anomaly).'
    }
    import json
    with open(os.path.join(args.outdir, 'datasets_overview.json'), 'w') as f:
        json.dump(overview, f, indent=2)

    print("All done. Files written to:", args.outdir)

if __name__ == "__main__":
    main()
