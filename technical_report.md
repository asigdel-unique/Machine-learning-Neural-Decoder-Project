# Neural Decoder Technical Report

## 1. Objective

Predict 2D hand movement velocity from motor cortex neural recordings using machine learning, evaluate model robustness, and assess practical viability for brain-computer interface applications.

## 2. Dataset

**Source**: MC_RTT (Random Target Task) from the Neural Latents Benchmark, DANDI Archive Dandiset 000129. Licensed under CC-BY-4.0.

**Subject**: Macaque "Indy", session 2017-02-02.

**Recording**: 96-channel Utah microelectrode array implanted in primary motor cortex (M1). This is the same electrode technology used in human BCI clinical trials (BrainGate).

**Task**: Self-paced reaching to randomly placed targets on a screen.

**Data contents**:
- 130 sorted neural units (98 observed, 32 held out by the NLB benchmark)
- Behavioral signals at 1000 Hz: cursor position (2D), finger position (3D), finger velocity (2D), target position (2D)
- Total duration: 649 seconds (~10.8 minutes)

## 3. Data Processing

**Spike binning**: Spike times were binned into 20 ms non-overlapping windows, producing spike count vectors of 98 dimensions per bin.

**Feature construction**: Each sample concatenates 6 consecutive bins (current + 5 preceding), creating a 120 ms temporal context window. This yields 588 features per sample (98 neurons × 6 bins).

**Target variable**: 2D finger velocity (X, Y), averaged within each 20 ms bin.

**Train/test split**: 80/20 split by time (first 519s for training, last 130s for testing). No shuffling was applied, preserving the temporal structure and preventing data leakage.

**Normalization**: Features were z-scored using training set mean and standard deviation. Test features were normalized with the same training statistics.

**Missing data**: 300 out of 649,100 timepoints (0.05%) contained NaN values, likely from gaps between continuous recording segments. Bins containing any NaN values were excluded.

## 4. Models

### 4.1 Baseline: Mean Predictor
Predicts the training set mean velocity for all test samples. This establishes the lower bound of performance.

### 4.2 Ridge Regression
L2-regularized linear regression (α=1.0) mapping the 588-dimensional feature vector to 2D velocity. Provides a strong linear baseline commonly used in BCI decoding literature.

### 4.3 Feedforward Neural Network
Architecture:
- Input layer: 588 units
- Hidden layer 1: 256 units, ReLU activation, 30% dropout
- Hidden layer 2: 128 units, ReLU activation, 30% dropout
- Output layer: 2 units (velocity X, Y)

Training:
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Learning rate scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
- Loss function: Mean squared error
- Early stopping: patience=20, best model at epoch 10
- Batch size: 256
- Validation set: last 15% of training data (time-based split)

## 5. Results

### 5.1 Model Comparison

| Model | MAE | RMSE | R² (Vel X) | R² (Vel Y) |
|-------|-----|------|------------|------------|
| Baseline (mean) | 36.84 | 72.10 | — | — |
| Ridge Regression | 37.15 | 56.71 | 0.389 | 0.368 |
| Neural Network | 29.52 | 50.56 | 0.507 | 0.510 |

The neural network outperforms Ridge regression on all metrics:
- 20.5% lower MAE
- 10.8% lower RMSE
- R² improved from ~0.38 to ~0.51 (31% relative improvement)

### 5.2 Cross-Temporal Stability

Evaluated across 5 time-based folds of the full dataset:
- Mean R² (Vel X): 0.601 ± 0.063
- Mean R² (Vel Y): 0.626 ± 0.072
- Performance is highest in the middle of the session and degrades slightly toward the end, consistent with neural drift within a session.

### 5.3 Overfitting Assessment

Average R² gap between train and test: 0.133. This indicates moderate overfitting, which is expected and acceptable given early stopping was used to select the best model.

### 5.4 Performance by Movement Speed

| Speed Range | Samples | MAE | RMSE |
|-------------|---------|-----|------|
| Stationary (0–10 mm/s) | 2,723 | 9.28 | 16.72 |
| Slow (10–50 mm/s) | 1,729 | 21.19 | 30.99 |
| Medium (50–150 mm/s) | 1,122 | 44.19 | 55.14 |
| Fast (150+ mm/s) | 913 | 87.66 | 108.50 |

Fast movements are the primary failure mode, with errors approximately 10× higher than during stationary periods. The model consistently underestimates peak velocity amplitudes.

### 5.5 Performance by Movement Direction

All 8 cardinal/diagonal directions were decoded, with MAE ranging from 34.71 (Down) to 61.55 (Right). The directional bias may reflect non-uniform sampling of movement directions in the self-paced reaching task.

## 6. Robustness Testing

### 6.1 Signal Drift (Simulated)

All drift scenarios below are simulated modifications to the normalized test features, not real observed drift.

| Drift Type | MAE Change | R² Change |
|------------|------------|-----------|
| Scale +20% | +7.2% | -7.4% |
| Scale +50% | +24.6% | -39.8% |
| Scale -30% | -2.0% | -10.0% |
| Offset shift (+0.5σ) | +26.6% | -27.6% |
| 20% channel dropout | +2.3% | -8.5% |
| Gaussian noise (σ=0.3) | +3.3% | -4.4% |

The model is robust to noise and moderate channel loss, but sensitive to gain scaling and offset shifts — the most common real-world drift modes in chronic implants.

### 6.2 Recalibration

After simulating +50% gain drift, retraining only the output layer on 60 seconds (3,000 samples) of labeled data recovered R² from ~0.30 to ~0.48, close to the original ~0.51. This demonstrates that simple recalibration strategies can effectively compensate for drift without full model retraining.

### 6.3 Channel Dropout Robustness

| Channels Dropped | MAE | Mean R² |
|-----------------|-----|---------|
| 0% (0) | 29.52 | 0.509 |
| 5% (4) | 29.67 | 0.500 |
| 10% (9) | 29.74 | 0.488 |
| 20% (19) | 30.09 | 0.473 |
| 30% (29) | 31.58 | 0.385 |
| 50% (49) | 33.04 | 0.284 |

Graceful degradation up to 20% dropout; significant performance loss begins at 30%.

## 7. Processing Speed

Per-window inference time on CPU:
- Mean: 0.064 ms
- 95th percentile: 0.085 ms
- Maximum: 0.955 ms

All windows are processed well within the 20 ms real-time budget (300× margin), indicating the model architecture is viable for real-time BCI applications from a computational standpoint.

## 8. Limitations

1. **Single session, single subject**: The model was trained and tested on one recording session from one macaque. Generalization to other subjects, sessions, or electrode implant locations was not tested.

2. **Offline evaluation only**: All results are from replaying pre-recorded data. Closed-loop BCI performance, where the subject's behavior depends on decoded output, was not assessed.

3. **Amplitude underestimation**: The model systematically underestimates peak velocities during fast movements, which would reduce control precision during rapid reaching.

4. **Prediction lag**: The decoder relies on a 120 ms history window, introducing inherent latency. The effective prediction lag is approximately 20–40 ms behind actual movement onset.

5. **Simulated drift only**: Signal drift was simulated through artificial modifications. Real chronic electrode drift involves more complex patterns including changes in neural tuning, electrode impedance, and unit sorting stability.

6. **No comparison with state-of-the-art**: The feedforward neural network is a simple architecture. Recurrent networks (LSTM, GRU), transformers, and Kalman filter approaches have demonstrated higher performance on similar datasets.

7. **Limited behavioral variables**: Only finger velocity was decoded. Position, force, and grasp-related signals were not modeled.

## 9. Conclusion

A 3-layer feedforward neural network successfully decodes 2D hand movement velocity from 98 motor cortex neurons, explaining approximately 51% of velocity variance (R² ≈ 0.51). The model substantially outperforms Ridge regression (R² ≈ 0.38) and runs 300× faster than real-time requirements. Robustness testing reveals acceptable tolerance to noise and channel loss, with simple output-layer recalibration able to recover performance after significant gain drift. The primary failure mode is underestimation of fast movement amplitudes. Future work could explore recurrent architectures, multi-session adaptation, and closed-loop evaluation.
