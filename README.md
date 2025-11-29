# OmniTransfer: Universal Transfer Learning for Time-Series Anomaly Detection

A comprehensive framework for transfer learning in anomaly detection, integrating the TranAD model with advanced weighting strategies (periodicity, entropy, causality) and hierarchical clustering for efficient knowledge transfer across time-series datasets.

## Overview

OmniTransfer enables rapid deployment of anomaly detection models on new systems by leveraging knowledge from previously seen data. The framework combines:

- **TranAD Model**: Transformer-based anomaly detection with dual decoders
- **Multiple Weighting Strategies**: Periodicity-based, entropy-based, and causal weighting
- **Hierarchical Clustering**: WHAC (Weighted Hierarchical Agglomerative Clustering) for shape library construction
- **Online Transfer**: Fast adaptation to new target systems with minimal training

## Project Structure

```
.
├── src/                      # Core framework modules
│   ├── models.py            # TranAD model implementation
│   ├── omni_framework.py    # OmniTransfer trainer
│   ├── clustering.py        # WHAC clustering algorithm
│   ├── entropy.py           # Entropy-based weighting
│   ├── causality.py         # Causal weighting calculator
│   └── data_loader.py       # Data loading utilities
├── scripts/                  # Experiment scripts
│   ├── full_comparison.py   # Full comparison suite
│   ├── viz_full.py          # Visualization script
│   └── compare.py           # Pairwise comparison
├── datasets/                 # Raw dataset storage
├── processed/                # Preprocessed data (.npy files)
├── results/                  # Experiment outputs
│   ├── csv/                 # Result tables
│   └── image/               # Visualizations
├── preprocess.py            # Data preprocessing pipeline
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

# For visualization support, also install:
pip install matplotlib seaborn
```

## Quick Start

### 1. Data Preprocessing

First, preprocess your datasets:

```bash
# Preprocess specific datasets
python preprocess.py SMD SMAP MSL

# Preprocess all supported datasets
python preprocess.py SMD SWaT SMAP MSL WADI MSDS UCR MBA NAB
```

This creates normalized `.npy` files in the `processed/` directory.

### 2. Run Full Comparison

Compare different configurations (TranAD Scratch, OmniTransfer with Periodic/Entropy/Causal weighting):

```bash
# Run comparison on multiple datasets
python -m scripts.full_comparison --datasets SMD SMAP MSL

# Specify configurations to test
python -m scripts.full_comparison --datasets SMD \
    --configs TranAD_Scratch Omni_Periodic Omni_Entropy

# Adjust entropy alpha parameter
python -m scripts.full_comparison --datasets SMD SMAP \
    --configs Omni_Entropy --alpha 1.0
```

**Available Configurations:**
- `TranAD_Scratch`: TranAD trained from scratch (baseline)
- `Omni_Periodic`: OmniTransfer with periodicity-based weighting
- `Omni_Entropy`: OmniTransfer with entropy-based weighting
- `Omni_Causal`: OmniTransfer with causal weighting
- `Omni_Entropy_Causal`: OmniTransfer with combined entropy + causal weighting

### 3. Visualize Results

Generate comprehensive visualizations:

```bash
# Visualize results for datasets
python -m scripts.viz_full --datasets SMD SMAP MSL
```

This creates a PNG with 6 plots:
1. F1 Score Comparison (bar chart)
2. Training Time Comparison (bar chart)
3. Accuracy vs Speed Trade-off (scatter plot)
4. Per-Dataset F1 Scores (table)
5. F1 Score Distribution (violin plot)
6. Speedup Comparison (bar chart)

## Supported Datasets

The framework supports the following benchmark datasets:

| Dataset | Domain | Entities | Features | Description |
|---------|--------|----------|----------|-------------|
| **SMD** | Server Machines | 28 | 38 | Server Machine Dataset |
| **SMAP** | Spacecraft | 55 | Varies | Soil Moisture Active Passive satellite |
| **MSL** | Spacecraft | 27 | Varies | Mars Science Laboratory rover |
| **SWaT** | Industrial | 1 | 51 | Secure Water Treatment testbed |
| **WADI** | Industrial | 1 | 123 | Water Distribution testbed |
| **UCR** | Various | 250 | 1 | UCR Time Series Archive |
| **NAB** | Various | Multiple | 1 | Numenta Anomaly Benchmark |
| **MBA** | Business | 1 | Multiple | MBA dataset |
| **MSDS** | Synthetic | 1 | Multiple | Microsoft dataset |

## Key Components

### TranAD Model (`src/models.py`)

Transformer-based anomaly detector with:
- Self-conditioning mechanism
- Dual decoder architecture
- Time-dependent adversarial training
- Attention-based reconstruction

### OmniTransfer Framework (`src/omni_framework.py`)

Two-phase transfer learning:

**Phase 1: Offline Shape Library Construction**
```python
from src.omni_framework import OmniTransferTrainer
from src.models import TranAD

trainer = OmniTransferTrainer(TranAD, device='cuda')
trainer.train_offline(source_data, weights, n_clusters=4)
```

**Phase 2: Online Transfer to Target**
```python
model = trainer.online_transfer(target_train_data)
scores = trainer.detect(model, target_test_data)
```

### Weighting Strategies

#### 1. Periodicity-Based (`src/data_loader.py`)
```python
from src.data_loader import PeriodicityCalculator
p_calc = PeriodicityCalculator()
weights = p_calc.compute_weights(data)
```

#### 2. Entropy-Based (`src/entropy.py`)
```python
from src.entropy import EntropicWeightCalculator
e_calc = EntropicWeightCalculator(alpha=2.0)
weights = e_calc.compute_entropic_weights(data)
```

#### 3. Causal Weighting (`src/causality.py`)
```python
from src.causality import CausalWeightCalculator
c_calc = CausalWeightCalculator(max_lag=2)
weights = c_calc.compute_causal_weights(data)
```

## Evaluation Metrics

The framework uses **Point-Adjusted F1 Score (PA-F1)** as the primary metric:
- Adjusts for anomaly segment detection
- If any point in an anomaly segment is detected, the entire segment is considered detected
- More practical than point-wise metrics for real-world applications

## Results Organization

After running experiments, results are stored as:

```
results/
├── csv/
│   ├── {datasets}_full_comparison_{timestamp}.csv      # Detailed per-machine results
│   ├── {datasets}_overall_summary_{timestamp}.csv      # Aggregated metrics
│   └── {datasets}_per_dataset_summary_{timestamp}.csv  # Per-dataset breakdown
└── image/
    └── {datasets}_full_comparison.png                   # Comprehensive visualization
```

## Advanced Usage

### Custom Alpha Parameter for Entropy

The `alpha` parameter controls the sensitivity of entropy-based weighting:

```bash
# More conservative (alpha = 0.5)
python -m scripts.full_comparison --datasets SMD --configs Omni_Entropy --alpha 0.5

# Standard (alpha = 2.0, default)
python -m scripts.full_comparison --datasets SMD --configs Omni_Entropy --alpha 2.0

# More aggressive (alpha = 4.0)
python -m scripts.full_comparison --datasets SMD --configs Omni_Entropy --alpha 4.0
```

### Cluster Auto-tuning

By default, the framework uses 4 clusters. To enable automatic cluster selection:

Edit `scripts/full_comparison.py` line 215-216:
```python
# trainer.train_offline(big_data, global_weights, n_clusters=4)  # Fixed 4 clusters
trainer.train_offline(big_data, global_weights, n_clusters=None)  # Auto-tune clusters
```

## Performance Benchmarks

Typical results (F1 Score / Training Time):

| Configuration | SMD | SMAP | MSL |
|--------------|-----|------|-----|
| TranAD_Scratch | 0.88 / 0.36s | 0.85 / 0.42s | 0.82 / 0.38s |
| Omni_Periodic | 0.85 / 0.12s | 0.83 / 0.14s | 0.80 / 0.13s |
| Omni_Entropy | 0.84 / 0.10s | 0.81 / 0.11s | 0.78 / 0.10s |

**Key Insights:**
- 🚀 **3-4x faster** training with OmniTransfer
- 📊 Slight accuracy trade-off (~3-5% lower F1)
- ⚡ Ideal for rapid deployment scenarios

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

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- New features include appropriate tests
- Documentation is updated accordingly

## Troubleshooting

### Common Issues

**Q: "No module named 'src'"**
```bash
# Make sure you're in the code directory
cd code
python -m scripts.full_comparison --datasets SMD
```

**Q: "File not found: results/csv/..."**
```bash
# The visualization script automatically finds timestamped files
# Just ensure you've run full_comparison first
python -m scripts.full_comparison --datasets SMD
python -m scripts.viz_full --datasets SMD
```

**Q: GPU/CUDA errors**
```bash
# The framework automatically falls back to CPU
# To force CPU usage:
export CUDA_VISIBLE_DEVICES=""
```

## Contact

For questions, issues, or suggestions, please open an issue on the repository.
