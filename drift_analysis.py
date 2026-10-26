import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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
test = np.load("processed_data/test_data.npz")
train = np.load("processed_data/train_data.npz")
X_test, y_test = test["X"], test["y"]
X_train, y_train = train["X"], train["y"]
train_mean, train_std = train["train_mean"], train["train_std"]

n_features = X_test.shape[1]

model = NeuralDecoder(n_features)
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "nn_best.pt"), weights_only=True))

# --- Baseline performance (no drift) ---
mae_orig, rmse_orig, r2x_orig, r2y_orig, pred_orig = evaluate(model, X_test, y_test)
print(f"Original performance (no drift):")
print(f"  MAE={mae_orig:.2f}, RMSE={rmse_orig:.2f}, R²X={r2x_orig:.3f}, R²Y={r2y_orig:.3f}")

# --- Simulate drift scenarios ---
# All drift is applied to normalized test features (simulating changes in firing rates)
drift_scenarios = {
    "Scale +20%": lambda X: X * 1.2,
    "Scale +50%": lambda X: X * 1.5,
    "Scale -30%": lambda X: X * 0.7,
    "Offset shift": lambda X: X + 0.5,
    "Random drop 20% channels": lambda X: drop_channels(X, 0.2),
    "Gaussian noise (σ=0.3)": lambda X: X + np.random.default_rng(42).normal(0, 0.3, X.shape),
}


def drop_channels(X, frac):
    rng = np.random.default_rng(42)
    n_neurons = 98
    n_history = X.shape[1] // n_neurons
    drop_idx = rng.choice(n_neurons, int(n_neurons * frac), replace=False)
    X_dropped = X.copy()
    for idx in drop_idx:
        for h in range(n_history):
            X_dropped[:, h * n_neurons + idx] = 0.0
    return X_dropped


print("\n" + "=" * 70)
print("SIGNAL DRIFT SIMULATION (all drift is simulated, not real)")
print("=" * 70)

results = {}
for name, drift_fn in drift_scenarios.items():
    X_drifted = drift_fn(X_test.copy()).astype(np.float32)
    mae, rmse, r2x, r2y, pred = evaluate(model, X_drifted, y_test)
    pct_mae = (mae - mae_orig) / mae_orig * 100
    pct_r2 = ((r2x + r2y) / 2 - (r2x_orig + r2y_orig) / 2) / ((r2x_orig + r2y_orig) / 2) * 100
    results[name] = {"mae": mae, "rmse": rmse, "r2x": r2x, "r2y": r2y,
                     "pct_mae": pct_mae, "pct_r2": pct_r2, "pred": pred,
                     "X_drifted": X_drifted}
    print(f"\n  {name}:")
    print(f"    MAE={mae:.2f} ({pct_mae:+.1f}%), R²X={r2x:.3f}, R²Y={r2y:.3f} ({pct_r2:+.1f}%)")


# --- Simple recalibration: retrain last layer on small amount of drifted data ---
print("\n" + "=" * 70)
print("RECALIBRATION: Retrain output layer on 60s of drifted+labeled data")
print("=" * 70)

RECAL_SAMPLES = 3000  # ~60 seconds at 20ms bins

worst_drift = "Scale +50%"
X_drifted = results[worst_drift]["X_drifted"]

X_recal = X_drifted[:RECAL_SAMPLES]
y_recal = y_test[:RECAL_SAMPLES]
X_eval = X_drifted[RECAL_SAMPLES:]
y_eval = y_test[RECAL_SAMPLES:]

# Freeze all layers except the last
model_adapted = NeuralDecoder(n_features)
model_adapted.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "nn_best.pt"), weights_only=True))

for name_p, param in model_adapted.named_parameters():
    if "net.6" not in name_p:  # only unfreeze last linear layer
        param.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_adapted.parameters()),
                             lr=1e-3)
criterion = nn.MSELoss()

recal_ds = TensorDataset(torch.FloatTensor(X_recal), torch.FloatTensor(y_recal))
recal_loader = DataLoader(recal_ds, batch_size=256, shuffle=True)

model_adapted.train()
for epoch in range(30):
    for X_batch, y_batch in recal_loader:
        pred = model_adapted(X_batch)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

mae_before, rmse_before, r2x_before, r2y_before, pred_before = evaluate(
    model, X_eval, y_eval)
mae_after, rmse_after, r2x_after, r2y_after, pred_after = evaluate(
    model_adapted, X_eval, y_eval)

print(f"\n  Drift scenario: {worst_drift}")
print(f"  Recalibration data: {RECAL_SAMPLES} samples ({RECAL_SAMPLES * 0.02:.0f}s)")
print(f"  Evaluated on remaining {len(X_eval)} test samples\n")
print(f"  {'Metric':<12} {'Before':>10} {'After':>10} {'Change':>10}")
print(f"  {'-'*44}")
print(f"  {'MAE':<12} {mae_before:>10.2f} {mae_after:>10.2f} {(mae_after-mae_before):>+10.2f}")
print(f"  {'RMSE':<12} {rmse_before:>10.2f} {rmse_after:>10.2f} {(rmse_after-rmse_before):>+10.2f}")
print(f"  {'R² X':<12} {r2x_before:>10.3f} {r2x_after:>10.3f} {(r2x_after-r2x_before):>+10.3f}")
print(f"  {'R² Y':<12} {r2y_before:>10.3f} {r2y_after:>10.3f} {(r2y_after-r2y_before):>+10.3f}")

# --- Plots ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Drift impact bar chart
ax = axes[0, 0]
names = list(results.keys())
maes = [results[n]["mae"] for n in names]
ax.barh(names, maes, color="coral")
ax.axvline(mae_orig, color="steelblue", linestyle="--", linewidth=2, label=f"No drift ({mae_orig:.1f})")
ax.set_xlabel("MAE")
ax.set_title("Impact of Simulated Signal Drift on MAE")
ax.legend()

# R² degradation
ax = axes[0, 1]
r2_avgs = [(results[n]["r2x"] + results[n]["r2y"]) / 2 for n in names]
ax.barh(names, r2_avgs, color="steelblue")
ax.axvline((r2x_orig + r2y_orig) / 2, color="coral", linestyle="--", linewidth=2,
           label=f"No drift ({(r2x_orig+r2y_orig)/2:.3f})")
ax.set_xlabel("Mean R²")
ax.set_title("Impact of Simulated Signal Drift on R²")
ax.legend()

# Time series: worst drift before/after adaptation
t = np.arange(len(y_eval)) * 0.02
mask = t <= 10
ax = axes[1, 0]
ax.plot(t[mask], y_eval[mask, 0], label="Actual", linewidth=0.8)
ax.plot(t[mask], pred_before[mask, 0], label="Before recal", linewidth=0.8, alpha=0.7)
ax.plot(t[mask], pred_after[mask, 0], label="After recal", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity X")
ax.set_title(f"Recalibration Effect ({worst_drift}, Vel X)")
ax.legend()

ax = axes[1, 1]
ax.plot(t[mask], y_eval[mask, 1], label="Actual", linewidth=0.8)
ax.plot(t[mask], pred_before[mask, 1], label="Before recal", linewidth=0.8, alpha=0.7)
ax.plot(t[mask], pred_after[mask, 1], label="After recal", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity Y")
ax.set_title(f"Recalibration Effect ({worst_drift}, Vel Y)")
ax.legend()

plt.suptitle("Signal Drift Simulation & Recalibration\n(All drift is simulated, not real)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/signal_drift.png", dpi=150)
plt.close()
print("\nSaved figures/signal_drift.png")
print("\nNOTE: All drift scenarios are simulated. No real signal drift was observed.")
