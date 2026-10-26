"""
Offline Neural Decoder Demo

Loads the trained neural network model and processes held-out neural data
in sequential time windows, displaying predicted vs actual movement.

NOTE: This is an OFFLINE demo — it replays pre-recorded data.
It is NOT a real-time closed-loop brain-computer interface.
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os


class NeuralDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


# --- Load model and data ---
test = np.load("processed_data/test_data.npz")
train = np.load("processed_data/train_data.npz")
X_test, y_test = test["X"], test["y"]

n_features = X_test.shape[1]
model = NeuralDecoder(n_features)
model.load_state_dict(torch.load("results/nn_best.pt", weights_only=True))
model.eval()

# --- Process data in windows and measure timing ---
BIN_SIZE_S = 0.02
WINDOW_DISPLAY = 250  # 5 seconds of display context
N_SAMPLES = len(X_test)

print("=" * 60)
print("OFFLINE NEURAL DECODER DEMO")
print("This demo replays pre-recorded neural data.")
print("It is NOT running in a real-time closed-loop.")
print("=" * 60)
print(f"\nModel: 3-layer feedforward neural network")
print(f"Input: {n_features} features (98 neurons x 6 time bins)")
print(f"Output: 2D finger velocity (X, Y)")
print(f"Test samples: {N_SAMPLES} ({N_SAMPLES * BIN_SIZE_S:.1f} seconds)")

# Measure per-window processing time
predictions = np.zeros_like(y_test)
times_per_window = []

for i in range(N_SAMPLES):
    x = torch.FloatTensor(X_test[i:i+1])
    t_start = time.perf_counter()
    with torch.no_grad():
        pred = model(x).numpy()
    t_end = time.perf_counter()
    predictions[i] = pred[0]
    times_per_window.append(t_end - t_start)

times_ms = np.array(times_per_window) * 1000
print(f"\nPer-window processing time:")
print(f"  Mean:   {times_ms.mean():.3f} ms")
print(f"  Median: {np.median(times_ms):.3f} ms")
print(f"  95th:   {np.percentile(times_ms, 95):.3f} ms")
print(f"  Max:    {times_ms.max():.3f} ms")
print(f"  Bin size: {BIN_SIZE_S * 1000:.0f} ms")
if times_ms.mean() < BIN_SIZE_S * 1000:
    print(f"  >> Processing is faster than real-time ({times_ms.mean():.3f} ms < {BIN_SIZE_S*1000:.0f} ms)")
else:
    print(f"  >> WARNING: Processing slower than real-time")

# --- Generate demo figure ---
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.2, 0.8], hspace=0.35, wspace=0.3)

t = np.arange(N_SAMPLES) * BIN_SIZE_S

# Full test set: Velocity X
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, y_test[:, 0], label="Actual", linewidth=0.5, alpha=0.7, color="steelblue")
ax1.plot(t, predictions[:, 0], label="Predicted", linewidth=0.5, alpha=0.7, color="coral")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Velocity X (mm/s)")
ax1.set_title("Full Test Session — Velocity X")
ax1.legend(loc="upper right")

# Full test set: Velocity Y
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t, y_test[:, 1], label="Actual", linewidth=0.5, alpha=0.7, color="steelblue")
ax2.plot(t, predictions[:, 1], label="Predicted", linewidth=0.5, alpha=0.7, color="coral")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Velocity Y (mm/s)")
ax2.set_title("Full Test Session — Velocity Y")
ax2.legend(loc="upper right")

# Zoomed window (5-second segment)
zoom_start = 1000
zoom_end = zoom_start + WINDOW_DISPLAY
t_zoom = t[zoom_start:zoom_end]

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t_zoom, y_test[zoom_start:zoom_end, 0], label="Actual", linewidth=1.5, color="steelblue")
ax3.plot(t_zoom, predictions[zoom_start:zoom_end, 0], label="Predicted", linewidth=1.5, color="coral")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Velocity X (mm/s)")
ax3.set_title("Zoomed Window — Velocity X (5 seconds)")
ax3.legend()

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t_zoom, y_test[zoom_start:zoom_end, 1], label="Actual", linewidth=1.5, color="steelblue")
ax4.plot(t_zoom, predictions[zoom_start:zoom_end, 1], label="Predicted", linewidth=1.5, color="coral")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Velocity Y (mm/s)")
ax4.set_title("Zoomed Window — Velocity Y (5 seconds)")
ax4.legend()

# 2D trajectory reconstruction
actual_pos = np.cumsum(y_test * BIN_SIZE_S, axis=0)
pred_pos = np.cumsum(predictions * BIN_SIZE_S, axis=0)

traj_start = 0
traj_end = 1500  # 30 seconds

ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(actual_pos[traj_start:traj_end, 0], actual_pos[traj_start:traj_end, 1],
         label="Actual trajectory", linewidth=0.8, color="steelblue")
ax5.plot(pred_pos[traj_start:traj_end, 0], pred_pos[traj_start:traj_end, 1],
         label="Predicted trajectory", linewidth=0.8, color="coral")
ax5.set_xlabel("X position (integrated)")
ax5.set_ylabel("Y position (integrated)")
ax5.set_title("2D Trajectory (first 30s, integrated from velocity)")
ax5.legend()
ax5.set_aspect("equal")

# Processing time histogram
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(times_ms, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
ax6.axvline(BIN_SIZE_S * 1000, color="red", linestyle="--", linewidth=2,
            label=f"Real-time limit ({BIN_SIZE_S*1000:.0f} ms)")
ax6.axvline(times_ms.mean(), color="orange", linestyle="--", linewidth=2,
            label=f"Mean ({times_ms.mean():.3f} ms)")
ax6.set_xlabel("Processing time per window (ms)")
ax6.set_ylabel("Count")
ax6.set_title("Processing Time Distribution")
ax6.legend()

fig.suptitle("Offline Neural Decoder Demo\n(Replayed pre-recorded data — NOT real-time closed-loop)",
             fontsize=14, fontweight="bold")
plt.savefig("figures/demo_output.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved figures/demo_output.png")
print("\nDEMO COMPLETE (offline replay of pre-recorded neural data)")
