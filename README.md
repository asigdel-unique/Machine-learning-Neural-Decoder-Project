# Neural Decoder for Brain-Computer Interfaces

Predicting 2D hand movement velocity from motor cortex neural recordings using machine learning. This project evaluates model robustness and assesses practical viability for brain-computer interface (BCI) applications.

## Dataset

- **Source**: [DANDI Archive — Dandiset 000129](https://dandiarchive.org/dandiset/000129)
- **Subject**: Macaque "Indy"
- **Task**: Self-paced random target reaching
- **Recording**: 96-channel Utah array in primary motor cortex (M1)
- **License**: CC-BY-4.0

## Setup

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate

pip install numpy pandas matplotlib scipy scikit-learn
pip install dandi pynwb h5py
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Usage

```bash
python check_environment.py        # Verify dependencies
python data_loader.py              # Download and load dataset
python exploratory_analysis.py     # Visualize neural and behavioral data
python preprocessing.py            # Bin spikes, extract features, train/test split
python baseline_model.py           # Train Ridge regression decoder
python train_neural_network.py     # Train neural network decoder
python drift_analysis.py           # Simulate signal drift and recalibration
python reliability_analysis.py     # Cross-temporal, speed, direction, dropout tests
python run_demo.py                 # Run offline decoding demo
python generate_figures.py         # Generate final comparison figures
```

## Data Processing

- Spike times binned into 20 ms windows
- 6 bins of history (120 ms context) concatenated per sample
- 98 observed units (32 held out by the NLB benchmark)
- Target: 2D finger velocity (X, Y)
- Train/test split: 80/20 by time (no shuffling)
- Features normalized using training statistics only

## Results

| Model | MAE | RMSE | R² (Vel X) | R² (Vel Y) |
|-------|-----|------|------------|------------|
| Baseline (mean) | 36.84 | 72.10 | — | — |
| Ridge Regression | 37.15 | 56.71 | 0.389 | 0.368 |
| Neural Network | 29.52 | 50.56 | 0.507 | 0.510 |

The neural network (3-layer feedforward, 256→128→2, ReLU, dropout 0.3) outperforms Ridge regression on all metrics.

### Key Findings

- Neural network decoder explains ~51% of velocity variance from neural activity
- Inference time per window: 0.064 ms (300x faster than the 20 ms real-time requirement)
- Robust to 20% channel dropout and moderate noise
- Gain drift >50% requires recalibration; retraining the output layer on 60s of labeled data recovers most performance
- Fast movements (>150 mm/s) show the highest prediction error
- Consistent performance across time within session (R² 0.51–0.71 across 5 folds)

### Limitations

1. Trained and tested on a single session from one subject
2. Offline evaluation only — not a real-time closed-loop BCI
3. Predictions underestimate fast movement amplitudes
4. Prediction lag of ~20–40 ms behind actual movement onset
5. Performance degrades with >30% channel dropout
6. Assumes stationary neural tuning within the session

## Project Structure

```
├── check_environment.py        # Environment verification
├── data_loader.py              # Data loading from DANDI/NWB
├── exploratory_analysis.py     # EDA and visualization
├── preprocessing.py            # Feature engineering pipeline
├── baseline_model.py           # Ridge regression decoder
├── train_neural_network.py     # Neural network training and evaluation
├── drift_analysis.py           # Signal drift simulation
├── reliability_analysis.py     # Robustness and reliability testing
├── run_demo.py                 # Offline decoding demo
├── generate_figures.py         # Final comparison figures
├── demo_live.py                # Live visualization demo
├── demo_visual.py              # Visual demo
├── technical_report.md         # Detailed technical report
├── processed_data/             # Preprocessed arrays
├── results/                    # Trained models and metrics
└── figures/                    # Generated plots
```

## Citation

If you use this dataset, please cite:

- Neural Latents Benchmark: Pei et al., "Neural Latents Benchmark '21: Evaluating latent variable models of neural population activity", NeurIPS 2021
- Dataset: DANDI Archive, Dandiset 000129
