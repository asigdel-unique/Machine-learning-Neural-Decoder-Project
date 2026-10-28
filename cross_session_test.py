"""
Cross-session generalization test.
Simulates testing on a different session by splitting the recording in half:
  - Session A: first 50% of the data (earlier in time)
  - Session B: last 50% of the data (later in time)
Tests how well models trained on one time period generalize to another,
which simulates the neural drift problem in real BCIs.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")


class GRUDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def calc_r2(y_true, y_pred):
    r2x = 1 - np.sum((y_true[:, 0] - y_pred[:, 0])**2) / np.sum((y_true[:, 0] - y_true[:, 0].mean())**2)
    r2y = 1 - np.sum((y_true[:, 1] - y_pred[:, 1])**2) / np.sum((y_true[:, 1] - y_true[:, 1].mean())**2)
    return r2x, r2y, (r2x + r2y) / 2


def train_gru(X_train, y_train, X_val, y_val, n_neurons, save_name, n_epochs=80, patience=20):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = GRUDecoder(n_neurons)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    best_val = float("inf")
    wait = 0
    save_path = os.path.join(OUTPUT_DIR, save_name)

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vloss, nv = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                vloss += criterion(model(xb), yb).item() * len(xb)
                nv += len(xb)
        vl = vloss / nv
        scheduler.step()

        if vl < best_val:
            best_val = vl
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: val={vl:.2f}")

        if wait >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model


def predict_gru(model, X):
    model.eval()
    ds = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(ds, batch_size=256)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb).numpy())
    return np.concatenate(preds)


# ============================================================
# Load and split data
# ============================================================
print("Loading V2 preprocessed data...")
train_data = np.load(os.path.join(BASE_DIR, "processed_data", "v2_train.npz"))
test_data = np.load(os.path.join(BASE_DIR, "processed_data", "v2_test.npz"))

X_all = np.concatenate([train_data["X_seq"], test_data["X_seq"]], axis=0)
y_all = np.concatenate([train_data["y"], test_data["y"]], axis=0)

n_total = len(y_all)
n_half = n_total // 2
n_neurons = X_all.shape[2]

X_sessionA = X_all[:n_half]
y_sessionA = y_all[:n_half]
X_sessionB = X_all[n_half:]
y_sessionB = y_all[n_half:]

print(f"Total samples: {n_total}")
print(f"Session A (early): {len(y_sessionA)} samples")
print(f"Session B (late):  {len(y_sessionB)} samples")
print(f"Sequence: {X_all.shape[1]} timesteps x {n_neurons} neurons")

# Normalize each session independently
def normalize_session(X_train, X_test):
    n = X_train.shape[2]
    mean = X_train.reshape(-1, n).mean(axis=0)
    std = X_train.reshape(-1, n).std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std

# ============================================================
# Test 1: Train on A, test on B (forward generalization)
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: Train on Session A (early) -> Test on Session B (late)")
print("=" * 70)

val_split = int(len(X_sessionA) * 0.85)
X_trainA = X_sessionA[:val_split]
X_valA = X_sessionA[val_split:]
y_trainA = y_sessionA[:val_split]
y_valA = y_sessionA[val_split:]

X_trainA_n, X_valA_n = normalize_session(X_trainA, X_valA)
_, X_testB_n = normalize_session(X_trainA, X_sessionB)

# Ridge baseline
X_flat_train = X_trainA_n.reshape(len(X_trainA_n), -1)
X_flat_test = X_testB_n.reshape(len(X_testB_n), -1)
ridge = Ridge(alpha=1.0)
ridge.fit(X_flat_train, y_trainA)
ridge_pred = ridge.predict(X_flat_test)
ridge_r2x, ridge_r2y, ridge_r2 = calc_r2(y_sessionB, ridge_pred)
print(f"\n  Ridge:  R2 = {ridge_r2:.4f} (X={ridge_r2x:.4f}, Y={ridge_r2y:.4f})")

# GRU
print("  Training GRU on Session A...")
gru_ab = train_gru(X_trainA_n, y_trainA, X_valA_n, y_valA, n_neurons, "cross_gru_ab.pt")
gru_pred_ab = predict_gru(gru_ab, X_testB_n)
gru_r2x, gru_r2y, gru_r2 = calc_r2(y_sessionB, gru_pred_ab)
print(f"  GRU:    R2 = {gru_r2:.4f} (X={gru_r2x:.4f}, Y={gru_r2y:.4f})")

del gru_ab

# ============================================================
# Test 2: Train on B, test on A (backward generalization)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Train on Session B (late) -> Test on Session A (early)")
print("=" * 70)

val_split = int(len(X_sessionB) * 0.85)
X_trainB = X_sessionB[:val_split]
X_valB = X_sessionB[val_split:]
y_trainB = y_sessionB[:val_split]
y_valB = y_sessionB[val_split:]

X_trainB_n, X_valB_n = normalize_session(X_trainB, X_valB)
_, X_testA_n = normalize_session(X_trainB, X_sessionA)

# Ridge
X_flat_train = X_trainB_n.reshape(len(X_trainB_n), -1)
X_flat_test = X_testA_n.reshape(len(X_testA_n), -1)
ridge = Ridge(alpha=1.0)
ridge.fit(X_flat_train, y_trainB)
ridge_pred_ba = ridge.predict(X_flat_test)
ridge_r2x_ba, ridge_r2y_ba, ridge_r2_ba = calc_r2(y_sessionA, ridge_pred_ba)
print(f"\n  Ridge:  R2 = {ridge_r2_ba:.4f} (X={ridge_r2x_ba:.4f}, Y={ridge_r2y_ba:.4f})")

# GRU
print("  Training GRU on Session B...")
gru_ba = train_gru(X_trainB_n, y_trainB, X_valB_n, y_valB, n_neurons, "cross_gru_ba.pt")
gru_pred_ba = predict_gru(gru_ba, X_testA_n)
gru_r2x_ba, gru_r2y_ba, gru_r2_ba = calc_r2(y_sessionA, gru_pred_ba)
print(f"  GRU:    R2 = {gru_r2_ba:.4f} (X={gru_r2x_ba:.4f}, Y={gru_r2y_ba:.4f})")

del gru_ba

# ============================================================
# Test 3: Same-session baseline (train on 80% of A, test on 20% of A)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Same-session baseline (80/20 split within Session A)")
print("=" * 70)

split = int(len(X_sessionA) * 0.8)
X_same_tr = X_sessionA[:split]
y_same_tr = y_sessionA[:split]
X_same_te = X_sessionA[split:]
y_same_te = y_sessionA[split:]

val_s = int(len(X_same_tr) * 0.85)
X_str, X_sval = X_same_tr[:val_s], X_same_tr[val_s:]
y_str, y_sval = y_same_tr[:val_s], y_same_tr[val_s:]

X_str_n, X_sval_n = normalize_session(X_str, X_sval)
_, X_same_te_n = normalize_session(X_str, X_same_te)

print("  Training GRU (same-session)...")
gru_same = train_gru(X_str_n, y_str, X_sval_n, y_sval, n_neurons, "cross_gru_same.pt")
gru_pred_same = predict_gru(gru_same, X_same_te_n)
same_r2x, same_r2y, same_r2 = calc_r2(y_same_te, gru_pred_same)
print(f"  GRU:    R2 = {same_r2:.4f} (X={same_r2x:.4f}, Y={same_r2y:.4f})")

del gru_same

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("CROSS-SESSION GENERALIZATION SUMMARY")
print("=" * 70)

print(f"\n{'Test Condition':<45} {'Ridge R2':>10} {'GRU R2':>10}")
print("-" * 67)
print(f"{'Same-session (A train -> A test)':<45} {'---':>10} {same_r2:>10.4f}")
print(f"{'Cross-session (A train -> B test)':<45} {ridge_r2:>10.4f} {gru_r2:>10.4f}")
print(f"{'Cross-session (B train -> A test)':<45} {ridge_r2_ba:>10.4f} {gru_r2_ba:>10.4f}")
avg_cross = (gru_r2 + gru_r2_ba) / 2
print(f"{'Cross-session average':<45} {(ridge_r2+ridge_r2_ba)/2:>10.4f} {avg_cross:>10.4f}")

drop = same_r2 - avg_cross
print(f"\n  R2 drop from same-session to cross-session: {drop:.4f} ({drop/same_r2*100:.1f}%)")
print(f"  This simulates neural drift between recording sessions.")
if drop < 0.05:
    print("  -> Minimal drift: model generalizes well across time.")
elif drop < 0.15:
    print("  -> Moderate drift: model is somewhat robust but loses performance.")
else:
    print("  -> Significant drift: recalibration would be needed for a real BCI.")

# ============================================================
# Figure
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Bar chart comparison
ax = axes[0]
conditions = ["Same\nsession", "A->B\n(forward)", "B->A\n(backward)"]
gru_vals = [same_r2, gru_r2, gru_r2_ba]
colors = ["#2ecc71", "#e74c3c", "#3498db"]
bars = ax.bar(conditions, gru_vals, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("R2 Average")
ax.set_title("GRU Cross-Session Generalization")
ax.set_ylim(0, max(gru_vals) * 1.2)
for bar, val in zip(bars, gru_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

# 2. Velocity X: A->B prediction
ax = axes[1]
t = np.arange(min(500, len(y_sessionB))) * 0.02
n = len(t)
ax.plot(t, y_sessionB[:n, 0], label="Actual", linewidth=0.8, color="black")
ax.plot(t, gru_pred_ab[:n, 0], label="GRU (trained on A)", linewidth=0.8, color="#e74c3c", alpha=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity X (mm/s)")
ax.set_title("Cross-Session Prediction (A->B, first 10s)")
ax.legend(fontsize=8)

# 3. Velocity Y: A->B prediction
ax = axes[2]
ax.plot(t, y_sessionB[:n, 1], label="Actual", linewidth=0.8, color="black")
ax.plot(t, gru_pred_ab[:n, 1], label="GRU (trained on A)", linewidth=0.8, color="#3498db", alpha=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity Y (mm/s)")
ax.set_title("Cross-Session Prediction (A->B, first 10s)")
ax.legend(fontsize=8)

plt.suptitle("Cross-Session Generalization Test", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "cross_session_results.png"), dpi=150)
plt.close()
print(f"\nSaved figures/cross_session_results.png")

# Clean up temp model files
for f in ["cross_gru_ab.pt", "cross_gru_ba.pt", "cross_gru_same.pt"]:
    p = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(p):
        os.remove(p)
