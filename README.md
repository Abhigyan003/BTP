# OmniTransfer: Universal Transfer Learning for Time-Series Anomaly Detection

A comprehensive framework for transfer learning in anomaly detection, integrating the **TranAD** model with advanced weighting strategies (**Periodicity**, **Entropy**, **Causality**) and **Coarse-to-Fine (CTF)** modeling for efficient knowledge transfer across time-series datasets.

## Overview

OmniTransfer enables rapid deployment of anomaly detection models on new systems by leveraging knowledge from previously seen data. The framework combines:

- **TranAD Model**: Transformer-based anomaly detection with dual decoders.
- **Coarse-to-Fine (CTF) Transfer**: A novel approach using `RNN_VAE` for coarse-grained transfer and `TranAD` for fine-grained anomaly detection.
- **Entropic Clustering**: Grouping entities by complexity (entropy) rather than just periodicity, enabling robust transfer in state-dependent systems.
- **Hierarchical Clustering**: WHAC (Weighted Hierarchical Agglomerative Clustering) for shape library construction.

## Project Structure

```
.
├── src/                      # Core framework modules
│   ├── models.py            # TranAD & RNN_VAE model implementations
│   ├── omni_framework.py    # OmniTransfer trainer
│   ├── ctf.py               # Coarse-to-Fine (CTF) trainer
│   ├── clustering.py        # WHAC clustering algorithm
│   ├── entropy.py           # Entropy-based weighting
│   ├── causality.py         # Causal weighting calculator
│   └── data_loader.py       # Data loading utilities
├── scripts/                  # Experiment scripts
│   ├── full_comparison.py   # Full benchmark suite
│   ├── subset_comparison.py # Fast subset verification
│   ├── generateData.py      # Synthetic periodic data generation
│   ├── generateStateData.py # Synthetic state-dependent data generation
│   └── process_synthetic.py # Data processing pipeline
├── datasets/                 # Raw dataset storage
├── processed/                # Preprocessed data (.npy files)
├── results/                  # Experiment outputs
│   ├── csv/                 # Result tables
│   └── image/               # Visualizations
└── requirements.txt         # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd code

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Generation & Processing

Generate synthetic datasets to test specific capabilities:

```bash
# Generate Periodic Data (Synthetic_DS1/DS2)
python scripts/generateData.py

# Generate State-Dependent Data (State_DS1/DS2)
python scripts/generateStateData.py

# Process raw CSVs into .npy format
python scripts/process_synthetic.py
```

### 2. Run Benchmarks

Compare different configurations (TranAD Scratch, OmniTransfer Periodic/Entropy, Omni_CTF):

```bash
# Run full comparison on State-Dependent Data
python scripts/full_comparison.py --datasets State_DS2 --configs OmniTransfer_Entropy OmniTransfer_Periodic

# Run fast subset verification (e.g., first 20 entities)
python scripts/subset_comparison.py --datasets State_DS2 --limit 20
```

**Available Configurations:**
- `TranAD_Scratch`: Baseline trained from scratch per entity.
- `OmniTransfer_Periodic`: Transfer learning using periodicity-based clustering.
- `OmniTransfer_Entropy`: Transfer learning using entropy-based clustering (Best for state-dependent data).
- `Omni_CTF`: Coarse-to-Fine transfer using RNN_VAE + TranAD.

## Supported Datasets

| Dataset | Type | Entities | Dimensions | Description |
| :--- | :--- | :---: | :---: | :--- |
| **MSL** | Real-World | 27 | 55 | Mars Science Laboratory (Rover) |
| **SMAP** | Real-World | 55 | 25 | Soil Moisture Active Passive (Satellite) |
| **SMD** | Real-World | 28 | 38 | Server Machine Dataset |
| **Synthetic_DS1/2** | Synthetic | 400 | 19/25 | Strictly periodic data |
| **State_DS1/2** | Synthetic | 100/400 | 10/50 | State-dependent data (Stable, Volatile, Oscillating) |

## Key Insights

### 1. Entropy vs. Periodicity
Traditional transfer learning relies on **Periodicity** (grouping by time-of-day). However, in **State-Dependent Systems** (like `State_DS2`), behavior depends on the operating mode (e.g., "High Load"), not the clock.
- **OmniTransfer_Entropy** uses entropy as a prior to identify these states.
- **Result:** On `State_DS2`, Entropy found **4 clusters** (matching ground truth) while Periodic found only 2, leading to higher F1 scores.

### 2. Efficiency
Clustering reduces redundant training. Instead of training 400 separate models, OmniTransfer trains ~4 "Base Models" and fine-tunes them.
- **Speedup:** ~15x faster than training from scratch on large datasets (e.g., SMD).

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{tranad2022,
  title={TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data},
  author={Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R},
  journal={Proceedings of the VLDB Endowment},
  year={2022}
}
```

## License

This project is provided for research and educational purposes.
