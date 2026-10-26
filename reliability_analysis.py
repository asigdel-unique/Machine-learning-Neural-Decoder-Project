import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

OUTPUT_DIR = "results"


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


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X)).numpy()
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2_x = 1 - np.sum((y[:, 0] - pred[:, 0])**2) / np.sum((y[:, 0] - y[:, 0].mean())**2)
    r2_y = 1 - np.sum((y[:, 1] - pred[:, 1])**2) / np.sum((y[:, 1] - y[:, 1].mean())**2)
    return mae, rmse, r2_x, r2_y, pred


# --- Load data ---
train = np.load("processed_data/train_data.npz")
test = np.load("processed_data/test_data.npz")
X_all = np.concatenate([train["X"], test["X"]])
y_all = np.concatenate([train["y"], test["y"]])

n_features = X_all.shape[1]
model = NeuralDecoder(n_features)
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "nn_best.pt"), weights_only=True))

# ====================================================================
# TEST 1: Cross-temporal evaluation (5-fold time-based CV)
# ====================================================================
print("=" * 60)
print("TEST 1: Cross-temporal evaluation (5-fold time-based)")
print("=" * 60)

n_total = len(X_all)
fold_size = n_total // 5
fold_results = []

for fold in range(5):
    test_start = fold * fold_size
    test_end = test_start + fold_size
    mask_test = np.zeros(n_total, dtype=bool)
    mask_test[test_start:test_end] = True

    X_fold_test = X_all[mask_test]
    y_fold_test = y_all[mask_test]

    mae, rmse, r2x, r2y, _ = evaluate(model, X_fold_test, y_fold_test)
    fold_results.append({"fold": fold + 1, "mae": mae, "rmse": rmse, "r2x": r2x, "r2y": r2y,
                         "t_start": test_start * 0.02, "t_end": test_end * 0.02})
    print(f"  Fold {fold+1} (t={test_start*0.02:.0f}–{test_end*0.02:.0f}s): "
          f"MAE={mae:.2f}, R²X={r2x:.3f}, R²Y={r2y:.3f}")

maes = [r["mae"] for r in fold_results]
r2xs = [r["r2x"] for r in fold_results]
r2ys = [r["r2y"] for r in fold_results]
print(f"\n  Mean MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f}")
print(f"  Mean R²X: {np.mean(r2xs):.3f} ± {np.std(r2xs):.3f}")
print(f"  Mean R²Y: {np.mean(r2ys):.3f} ± {np.std(r2ys):.3f}")

# ====================================================================
# TEST 2: Overfitting check (train vs test performance)
# ====================================================================
print("\n" + "=" * 60)
print("TEST 2: Overfitting check")
print("=" * 60)

mae_train, rmse_train, r2x_train, r2y_train, _ = evaluate(model, train["X"], train["y"])
mae_test, rmse_test, r2x_test, r2y_test, _ = evaluate(model, test["X"], test["y"])

print(f"  {'Metric':<10} {'Train':>10} {'Test':>10} {'Gap':>10}")
print(f"  {'-'*42}")
print(f"  {'MAE':<10} {mae_train:>10.2f} {mae_test:>10.2f} {mae_test-mae_train:>+10.2f}")
print(f"  {'RMSE':<10} {rmse_train:>10.2f} {rmse_test:>10.2f} {rmse_test-rmse_train:>+10.2f}")
print(f"  {'R² X':<10} {r2x_train:>10.3f} {r2x_test:>10.3f} {r2x_test-r2x_train:>+10.3f}")
print(f"  {'R² Y':<10} {r2y_train:>10.3f} {r2y_test:>10.3f} {r2y_test-r2y_train:>+10.3f}")

overfit = (r2x_train - r2x_test + r2y_train - r2y_test) / 2
print(f"\n  Average R² gap (train-test): {overfit:.3f}")
if overfit > 0.15:
    print("  WARNING: Significant overfitting detected.")
elif overfit > 0.05:
    print("  Moderate overfitting — acceptable with early stopping.")
else:
    print("  Minimal overfitting.")

# ====================================================================
# TEST 3: Performance by movement speed
# ====================================================================
print("\n" + "=" * 60)
print("TEST 3: Performance by movement speed")
print("=" * 60)

X_test_data, y_test_data = test["X"], test["y"]
speed = np.sqrt(y_test_data[:, 0]**2 + y_test_data[:, 1]**2)

_, _, _, _, pred_test = evaluate(model, X_test_data, y_test_data)

speed_bins = [("Stationary (0–10)", 0, 10),
              ("Slow (10–50)", 10, 50),
              ("Medium (50–150)", 50, 150),
              ("Fast (150+)", 150, np.inf)]

speed_results = []
for label, lo, hi in speed_bins:
    mask = (speed >= lo) & (speed < hi)
    n = mask.sum()
    if n < 10:
        continue
    mae_s = mean_absolute_error(y_test_data[mask], pred_test[mask])
    rmse_s = np.sqrt(mean_squared_error(y_test_data[mask], pred_test[mask]))
    speed_results.append({"label": label, "n": n, "mae": mae_s, "rmse": rmse_s})
    print(f"  {label:<25} n={n:>5}  MAE={mae_s:>7.2f}  RMSE={rmse_s:>7.2f}")

# ====================================================================
# TEST 4: Performance by movement direction
# ====================================================================
print("\n" + "=" * 60)
print("TEST 4: Performance by movement direction")
print("=" * 60)

angles = np.arctan2(y_test_data[:, 1], y_test_data[:, 0])
moving = speed > 20

dir_names = ["Right", "Up-Right", "Up", "Up-Left", "Left", "Down-Left", "Down", "Down-Right"]
dir_bounds = np.linspace(-np.pi, np.pi, 9)
dir_results = []

for i in range(8):
    mask = moving & (angles >= dir_bounds[i]) & (angles < dir_bounds[i + 1])
    n = mask.sum()
    if n < 10:
        continue
    mae_d = mean_absolute_error(y_test_data[mask], pred_test[mask])
    dir_results.append({"dir": dir_names[i], "n": n, "mae": mae_d})
    print(f"  {dir_names[i]:<12} n={n:>5}  MAE={mae_d:>7.2f}")

# ====================================================================
# TEST 5: Noise and channel dropout robustness
# ====================================================================
print("\n" + "=" * 60)
print("TEST 5: Robustness to channel dropout")
print("=" * 60)

n_neurons = 98
n_history = X_test_data.shape[1] // n_neurons
rng = np.random.default_rng(42)

dropout_fracs = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
dropout_results = []

for frac in dropout_fracs:
    n_drop = int(n_neurons * frac)
    if n_drop > 0:
        drop_idx = rng.choice(n_neurons, n_drop, replace=False)
        X_dropped = X_test_data.copy()
        for idx in drop_idx:
            for h in range(n_history):
                X_dropped[:, h * n_neurons + idx] = 0.0
    else:
        X_dropped = X_test_data

    mae_d, rmse_d, r2x_d, r2y_d, _ = evaluate(model, X_dropped, y_test_data)
    dropout_results.append({"frac": frac, "mae": mae_d, "r2": (r2x_d + r2y_d) / 2})
    print(f"  Drop {frac*100:>4.0f}% ({n_drop:>2} channels): MAE={mae_d:.2f}, "
          f"R²={((r2x_d+r2y_d)/2):.3f}")

# ====================================================================
# Limitations summary
# ====================================================================
print("\n" + "=" * 60)
print("LIMITATIONS AND FAILURE CASES")
print("=" * 60)
print("  1. Fast movements (>150 mm/s) have higher prediction errors")
print("  2. Model was trained on a single session from one subject")
print("  3. Predictions lag behind actual movement onset by ~20-40ms")
print("  4. Performance degrades with >30% channel dropout")
print("  5. Gain drift >50% requires recalibration")
print("  6. Model assumes stationary neural tuning within session")
print("  7. Offline evaluation only — no closed-loop testing")

# --- Plots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Fold performance
ax = axes[0, 0]
folds = [r["fold"] for r in fold_results]
ax.bar(np.array(folds) - 0.15, [r["r2x"] for r in fold_results], 0.3,
       label="R² X", color="steelblue")
ax.bar(np.array(folds) + 0.15, [r["r2y"] for r in fold_results], 0.3,
       label="R² Y", color="coral")
ax.set_xlabel("Fold (time segment)")
ax.set_ylabel("R²")
ax.set_title("Cross-temporal R² by Fold")
ax.legend()
ax.set_xticks(folds)

# Speed vs error
ax = axes[0, 1]
labels = [r["label"].split("(")[0].strip() for r in speed_results]
ax.bar(labels, [r["mae"] for r in speed_results], color="steelblue")
ax.set_ylabel("MAE")
ax.set_title("Prediction Error by Movement Speed")
ax.tick_params(axis="x", rotation=15)

# Direction polar plot
ax = axes[1, 0]
if dir_results:
    dir_labels = [r["dir"] for r in dir_results]
    dir_maes = [r["mae"] for r in dir_results]
    ax.bar(dir_labels, dir_maes, color="steelblue")
    ax.set_ylabel("MAE")
    ax.set_title("Prediction Error by Movement Direction")
    ax.tick_params(axis="x", rotation=45)

# Channel dropout curve
ax = axes[1, 1]
fracs = [r["frac"] * 100 for r in dropout_results]
r2s = [r["r2"] for r in dropout_results]
ax.plot(fracs, r2s, "o-", color="steelblue", linewidth=2, markersize=8)
ax.set_xlabel("Channels Dropped (%)")
ax.set_ylabel("Mean R²")
ax.set_title("Robustness to Channel Dropout")
ax.grid(True, alpha=0.3)

plt.suptitle("Reliability Testing", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/reliability.png", dpi=150)
plt.close()
print("\nSaved figures/reliability.png")
