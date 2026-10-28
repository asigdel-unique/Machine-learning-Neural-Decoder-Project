"""
V2 preprocessing: sqrt transform + exponential smoothing + longer history.
Changes from improved_preprocessing.py:
  1. Square-root transform on spike counts (variance stabilization)
  2. 20 bins = 400ms context (was 15 = 300ms)
Saves to processed_data/v2_*.npz
"""
import numpy as np
from pynwb import NWBHDF5IO
import os

TRAIN_FILE = "data/000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb"
BIN_SIZE_S = 0.02
N_HISTORY = 20  # 400ms context (was 15 = 300ms)
SMOOTH_TAU_MS = 50
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading NWB file...")
with NWBHDF5IO(TRAIN_FILE, "r") as io:
    nwb = io.read()
    units = nwb.units
    n_units = len(units)
    behavior = nwb.processing["behavior"]
    finger_vel = behavior["finger_vel"].data[:]
    rate = behavior["finger_vel"].rate
    heldout = units["heldout"][:]
    all_spike_times = [units["spike_times"][i] for i in range(n_units)]

observed_idx = [i for i in range(n_units) if not heldout[i]]
n_neurons = len(observed_idx)
print(f"Using {n_neurons} observed units")

# --- Bin spikes ---
total_time = finger_vel.shape[0] / rate
n_bins = int(total_time / BIN_SIZE_S)
bin_edges = np.arange(n_bins + 1) * BIN_SIZE_S

spike_counts = np.zeros((n_bins, n_neurons), dtype=np.float32)
for j, unit_i in enumerate(observed_idx):
    st = all_spike_times[unit_i]
    counts, _ = np.histogram(st, bins=bin_edges)
    spike_counts[:, j] = counts

print(f"Spike counts: {spike_counts.shape}")
print(f"  Mean count: {spike_counts.mean():.3f}, Max: {spike_counts.max():.0f}")

# --- Square-root transform (variance stabilization for Poisson-like counts) ---
spike_sqrt = np.sqrt(spike_counts)
print(f"Applied sqrt transform (stabilizes Poisson variance)")
print(f"  Mean sqrt: {spike_sqrt.mean():.3f}, Std: {spike_sqrt.std():.3f}")

# --- Exponential smoothing ---
alpha = 1 - np.exp(-BIN_SIZE_S / (SMOOTH_TAU_MS / 1000))
smoothed = np.zeros_like(spike_sqrt)
smoothed[0] = spike_sqrt[0]
for t in range(1, len(spike_sqrt)):
    smoothed[t] = alpha * spike_sqrt[t] + (1 - alpha) * smoothed[t - 1]

print(f"Applied exponential smoothing (tau={SMOOTH_TAU_MS}ms, alpha={alpha:.4f})")

# --- Bin velocity ---
samples_per_bin = int(BIN_SIZE_S * rate)
n_bins_vel = finger_vel.shape[0] // samples_per_bin
vel_binned = np.zeros((n_bins_vel, 2), dtype=np.float32)

for i in range(n_bins_vel):
    start = i * samples_per_bin
    end = start + samples_per_bin
    chunk = finger_vel[start:end]
    if np.any(np.isnan(chunk)):
        vel_binned[i] = np.nan
    else:
        vel_binned[i] = chunk.mean(axis=0)

n_common = min(smoothed.shape[0], vel_binned.shape[0])
smoothed = smoothed[:n_common]
vel_binned = vel_binned[:n_common]

valid_mask = ~np.any(np.isnan(vel_binned), axis=1)
smoothed = smoothed[valid_mask]
vel_binned = vel_binned[valid_mask]
print(f"Valid bins: {smoothed.shape[0]}")

# --- Create sequences ---
n_samples = smoothed.shape[0] - N_HISTORY + 1
X_seq = np.zeros((n_samples, N_HISTORY, n_neurons), dtype=np.float32)
for i in range(n_samples):
    X_seq[i] = smoothed[i:i + N_HISTORY]

y = vel_binned[N_HISTORY - 1:]
print(f"Sequences: X={X_seq.shape} (samples x timesteps x neurons), y={y.shape}")

X_flat = X_seq.reshape(n_samples, -1)
print(f"Flattened: X={X_flat.shape}")

# --- Train/test split by time ---
split_idx = int(n_samples * 0.8)
X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
X_flat_train, X_flat_test = X_flat[:split_idx], X_flat[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --- Normalize using training stats ---
train_mean = X_seq_train.reshape(-1, n_neurons).mean(axis=0)
train_std = X_seq_train.reshape(-1, n_neurons).std(axis=0)
train_std[train_std == 0] = 1.0

X_seq_train = (X_seq_train - train_mean) / train_std
X_seq_test = (X_seq_test - train_mean) / train_std

X_flat_train = X_seq_train.reshape(X_seq_train.shape[0], -1)
X_flat_test = X_seq_test.reshape(X_seq_test.shape[0], -1)

print(f"\nTrain: {X_seq_train.shape[0]} samples")
print(f"Test:  {X_seq_test.shape[0]} samples")
print(f"History: {N_HISTORY} bins ({N_HISTORY * BIN_SIZE_S * 1000:.0f}ms)")
print(f"Features per timestep: {n_neurons}")

# --- Save ---
np.savez(os.path.join(OUTPUT_DIR, "v2_train.npz"),
         X_seq=X_seq_train, X_flat=X_flat_train, y=y_train,
         train_mean=train_mean, train_std=train_std)
np.savez(os.path.join(OUTPUT_DIR, "v2_test.npz"),
         X_seq=X_seq_test, X_flat=X_flat_test, y=y_test)

print(f"\nSaved to {OUTPUT_DIR}/v2_train.npz and v2_test.npz")
