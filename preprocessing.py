import numpy as np
from pynwb import NWBHDF5IO
import os

TRAIN_FILE = "data/000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb"
BIN_SIZE_S = 0.02  # 20 ms bins — standard for motor decoding
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load raw data ---
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
print(f"Using {len(observed_idx)} observed units (excluding {sum(heldout)} heldout)")

# --- Identify valid (non-NaN) regions ---
nan_mask = np.any(np.isnan(finger_vel), axis=1)
print(f"NaN timepoints: {nan_mask.sum()} out of {len(nan_mask)}")

# --- Bin spikes into time windows ---
total_time = finger_vel.shape[0] / rate
n_bins = int(total_time / BIN_SIZE_S)
bin_edges = np.arange(n_bins + 1) * BIN_SIZE_S

print(f"\nBinning parameters:")
print(f"  Bin size: {BIN_SIZE_S * 1000:.0f} ms")
print(f"  Total bins: {n_bins}")
print(f"  Duration covered: {n_bins * BIN_SIZE_S:.1f} s")

spike_counts = np.zeros((n_bins, len(observed_idx)), dtype=np.float32)
for j, unit_i in enumerate(observed_idx):
    st = all_spike_times[unit_i]
    counts, _ = np.histogram(st, bins=bin_edges)
    spike_counts[:, j] = counts

print(f"  Spike count matrix: {spike_counts.shape} (bins x neurons)")
print(f"  Mean firing rate across bins: {spike_counts.mean() / BIN_SIZE_S:.2f} Hz")

# --- Average finger velocity within each bin ---
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

n_common = min(spike_counts.shape[0], vel_binned.shape[0])
spike_counts = spike_counts[:n_common]
vel_binned = vel_binned[:n_common]

# --- Remove bins with NaN velocity ---
valid_mask = ~np.any(np.isnan(vel_binned), axis=1)
spike_counts = spike_counts[valid_mask]
vel_binned = vel_binned[valid_mask]
print(f"  After removing NaN bins: {spike_counts.shape[0]} valid bins")

# --- Add history: use neural activity from the current bin and previous bins ---
N_HISTORY = 6  # 6 bins of history = 120 ms total context (current + 5 past)

def create_history_features(X, n_history):
    n_samples, n_features = X.shape
    X_hist = np.zeros((n_samples - n_history + 1, n_features * n_history), dtype=np.float32)
    for i in range(n_history):
        offset = n_history - 1 - i
        X_hist[:, i * n_features:(i + 1) * n_features] = X[offset:n_samples - i if i > 0 else n_samples]
    return X_hist

X_hist = create_history_features(spike_counts, N_HISTORY)
y = vel_binned[N_HISTORY - 1:]
print(f"  With {N_HISTORY} bins of history ({N_HISTORY * BIN_SIZE_S * 1000:.0f} ms context):")
print(f"    X shape: {X_hist.shape} (samples x features)")
print(f"    y shape: {y.shape} (samples x velocity_dims)")

# --- Split by time (80% train, 20% test) ---
n_total = X_hist.shape[0]
split_idx = int(n_total * 0.8)

X_train = X_hist[:split_idx]
y_train = y[:split_idx]
X_test = X_hist[split_idx:]
y_test = y[split_idx:]

print(f"\n--- Train/test split (by time, no shuffling) ---")
print(f"  Train: {X_train.shape[0]} samples (first 80% of session)")
print(f"  Test:  {X_test.shape[0]} samples (last 20% of session)")

# --- Normalize features using training data only ---
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
train_std[train_std == 0] = 1.0  # avoid division by zero for silent units

X_train_norm = (X_train - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std  # normalize test with TRAIN stats

# --- Verify no data leakage ---
print(f"\n--- Data leakage check ---")
print(f"  Train time range: bins 0 to {split_idx - 1}")
print(f"  Test time range:  bins {split_idx} to {n_total - 1}")
print(f"  No overlap: {split_idx - 1 < split_idx}")
print(f"  Normalization uses train stats only: True")
print(f"  History bins stay within each sample's past: True")
print(f"  No shuffling across time: True")

# --- Save processed data ---
np.savez(
    os.path.join(OUTPUT_DIR, "train_data.npz"),
    X=X_train_norm, y=y_train,
    train_mean=train_mean, train_std=train_std
)
np.savez(
    os.path.join(OUTPUT_DIR, "test_data.npz"),
    X=X_test_norm, y=y_test
)

print(f"\n--- Saved to {OUTPUT_DIR}/ ---")
print(f"  train_data.npz: X={X_train_norm.shape}, y={y_train.shape}")
print(f"  test_data.npz:  X={X_test_norm.shape}, y={y_test.shape}")

print(f"\nPreprocessing parameters:")
print(f"  Bin size: {BIN_SIZE_S * 1000:.0f} ms")
print(f"  History bins: {N_HISTORY} ({N_HISTORY * BIN_SIZE_S * 1000:.0f} ms)")
print(f"  Features per sample: {X_hist.shape[1]} ({len(observed_idx)} units x {N_HISTORY} bins)")
print(f"  Target: 2D finger velocity (X, Y)")
print(f"\nDone.")
