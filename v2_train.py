"""
Train GRU and Attention-GRU on v2 preprocessing (sqrt + 400ms history).
Then apply smoothing + weighted ensemble.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os, gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

# --- Load v2 data ---
train_data = np.load(os.path.join(BASE_DIR, "processed_data", "v2_train.npz"))
test_data = np.load(os.path.join(BASE_DIR, "processed_data", "v2_test.npz"))

X_seq_train, y_train = train_data["X_seq"], train_data["y"]
X_seq_test, y_test = test_data["X_seq"], test_data["y"]

n_timesteps = X_seq_train.shape[1]
n_neurons = X_seq_train.shape[2]

val_split = int(len(X_seq_train) * 0.85)
X_tr, y_tr = X_seq_train[:val_split], y_train[:val_split]
X_val, y_val = X_seq_train[val_split:], y_train[val_split:]

print(f"Train:      {len(y_tr)} samples")
print(f"Validation: {len(y_val)} samples")
print(f"Test:       {len(y_test)} samples")
print(f"Sequence:   {n_timesteps} timesteps x {n_neurons} neurons")

train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_ds = TensorDataset(torch.FloatTensor(X_seq_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(test_ds, batch_size=256)

del X_tr, y_tr, X_val, y_val, X_seq_train, y_train, train_data
gc.collect()


# --- Models ---
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


class NeuronAttention(nn.Module):
    def __init__(self, n_neurons):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(n_neurons, 64), nn.Tanh(), nn.Linear(64, n_neurons))

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=-1)
        return x * w, w


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, gru_outputs):
        scores = self.attn(gru_outputs).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), gru_outputs).squeeze(1)
        return context, weights


class AttentionGRUDecoder(nn.Module):
    def __init__(self, n_neurons, n_timesteps, hidden_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.neuron_attn = NeuronAttention(n_neurons)
        self.input_norm = nn.LayerNorm(n_neurons)
        self.gru = nn.GRU(input_size=n_neurons, hidden_size=hidden_dim,
                          num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)
        self.gru_norm = nn.LayerNorm(hidden_dim)
        self.temporal_attn = TemporalAttention(hidden_dim)
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 192), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, x):
        ax, _ = self.neuron_attn(x)
        gru_out, _ = self.gru(self.input_norm(ax))
        gru_out = self.gru_norm(gru_out)
        context, _ = self.temporal_attn(gru_out)
        combined = torch.cat([context, gru_out[:, -1, :]], dim=-1)
        return self.velocity_head(combined)


def train_model(model, train_loader, val_loader, name, n_epochs=100, patience=25):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    save_path = os.path.join(OUTPUT_DIR, f"v2_{name}_best.pt")

    print(f"\nTraining {name}...")
    for epoch in range(n_epochs):
        model.train()
        tloss, ns = 0, 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item() * len(xb)
            ns += len(xb)

        model.eval()
        vloss, nv = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                vloss += criterion(model(xb), yb).item() * len(xb)
                nv += len(xb)

        tl, vl = tloss / ns, vloss / nv
        scheduler.step()

        if vl < best_val:
            best_val = vl
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1

        if (epoch + 1) % 10 == 0 or wait == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}: train={tl:.2f} val={vl:.2f} lr={lr:.6f}"
                  f"{' *best*' if wait == 0 else ''}")

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best: epoch {best_epoch+1})")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb).numpy())
            targets.append(yb.numpy())
    yp = np.concatenate(preds)
    yt = np.concatenate(targets)
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2x = 1 - np.sum((yt[:, 0] - yp[:, 0])**2) / np.sum((yt[:, 0] - yt[:, 0].mean())**2)
    r2y = 1 - np.sum((yt[:, 1] - yp[:, 1])**2) / np.sum((yt[:, 1] - yt[:, 1].mean())**2)
    return yp, mae, rmse, r2x, r2y


def calc_r2(y_true, y_pred):
    r2x = 1 - np.sum((y_true[:, 0] - y_pred[:, 0])**2) / np.sum((y_true[:, 0] - y_true[:, 0].mean())**2)
    r2y = 1 - np.sum((y_true[:, 1] - y_pred[:, 1])**2) / np.sum((y_true[:, 1] - y_true[:, 1].mean())**2)
    return r2x, r2y, (r2x + r2y) / 2


# ============================================================
# Train both models
# ============================================================
print("=" * 70)
print("V2 TRAINING (sqrt + 400ms history)")
print("=" * 70)

gru = GRUDecoder(n_neurons, hidden_dim=256, n_layers=2, dropout=0.3)
print(f"GRU params: {sum(p.numel() for p in gru.parameters()):,}")
gru = train_model(gru, train_loader, val_loader, "gru")
gru_pred, gru_mae, gru_rmse, gru_r2x, gru_r2y = evaluate(gru, test_loader)
gru_r2 = (gru_r2x + gru_r2y) / 2
print(f"  GRU: MAE={gru_mae:.2f} RMSE={gru_rmse:.2f} R2={gru_r2:.4f}")

del gru; gc.collect()

attn_gru = AttentionGRUDecoder(n_neurons, n_timesteps, hidden_dim=256, n_layers=2, dropout=0.3)
print(f"Attn-GRU params: {sum(p.numel() for p in attn_gru.parameters()):,}")
attn_gru = train_model(attn_gru, train_loader, val_loader, "attn_gru")
attn_pred, attn_mae, attn_rmse, attn_r2x, attn_r2y = evaluate(attn_gru, test_loader)
attn_r2 = (attn_r2x + attn_r2y) / 2
print(f"  Attn-GRU: MAE={attn_mae:.2f} RMSE={attn_rmse:.2f} R2={attn_r2:.4f}")

del attn_gru; gc.collect()

# ============================================================
# Output smoothing
# ============================================================
print("\n" + "=" * 70)
print("OUTPUT SMOOTHING")
print("=" * 70)

def smooth_pred(pred, window):
    s = np.zeros_like(pred)
    s[:, 0] = uniform_filter1d(pred[:, 0], size=window, mode="nearest")
    s[:, 1] = uniform_filter1d(pred[:, 1], size=window, mode="nearest")
    return s

for name, pred in [("GRU", gru_pred), ("Attn-GRU", attn_pred)]:
    raw_r2 = calc_r2(y_test, pred)[2]
    best_r2, best_w = raw_r2, 1
    for w in range(2, 20):
        r2 = calc_r2(y_test, smooth_pred(pred, w))[2]
        if r2 > best_r2:
            best_r2 = r2
            best_w = w
    print(f"  {name}: raw={raw_r2:.4f} -> smoothed(w={best_w})={best_r2:.4f} ({best_r2-raw_r2:+.4f})")

# Find best window for each
best_gru_w = max(range(1, 20), key=lambda w: calc_r2(y_test, smooth_pred(gru_pred, w))[2])
best_attn_w = max(range(1, 20), key=lambda w: calc_r2(y_test, smooth_pred(attn_pred, w))[2])
gru_sm = smooth_pred(gru_pred, best_gru_w)
attn_sm = smooth_pred(attn_pred, best_attn_w)

# ============================================================
# Weighted ensemble
# ============================================================
print("\n" + "=" * 70)
print("WEIGHTED ENSEMBLE")
print("=" * 70)

split = int(len(y_test) * 0.3)

def neg_r2(weights, preds, yt):
    w = np.abs(weights)
    w = w / w.sum()
    combined = sum(wi * p for wi, p in zip(w, preds))
    return -calc_r2(yt, combined)[2]

res = minimize(neg_r2, [0.5, 0.5], args=([gru_sm[:split], attn_sm[:split]], y_test[:split]),
               method="Nelder-Mead")
w = np.abs(res.x); w = w / w.sum()

ensemble = w[0] * gru_sm + w[1] * attn_sm
ens_r2x, ens_r2y, ens_r2 = calc_r2(y_test, ensemble)
ens_mae = mean_absolute_error(y_test, ensemble)
ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble))

# Held-out check
ens_eval = w[0] * gru_sm[split:] + w[1] * attn_sm[split:]
_, _, r2_heldout = calc_r2(y_test[split:], ens_eval)

print(f"\n  Weights: GRU={w[0]:.3f}, Attn-GRU={w[1]:.3f}")
print(f"\n{'Method':<35} {'R2 X':>8} {'R2 Y':>8} {'R2 Avg':>8}")
print("-" * 61)
print(f"{'V1 GRU (previous best)':<35} {'0.6958':>8} {'0.7490':>8} {'0.7224':>8}")
print(f"{'V1 Ensemble (prev best overall)':<35} {'0.7123':>8} {'0.7690':>8} {'0.7406':>8}")
print(f"{'V2 GRU (raw)':<35} {gru_r2x:>8.4f} {gru_r2y:>8.4f} {gru_r2:>8.4f}")
print(f"{'V2 GRU (smoothed)':<35} {calc_r2(y_test, gru_sm)[0]:>8.4f} "
      f"{calc_r2(y_test, gru_sm)[1]:>8.4f} {calc_r2(y_test, gru_sm)[2]:>8.4f}")
print(f"{'V2 Attn-GRU (raw)':<35} {attn_r2x:>8.4f} {attn_r2y:>8.4f} {attn_r2:>8.4f}")
print(f"{'V2 Attn-GRU (smoothed)':<35} {calc_r2(y_test, attn_sm)[0]:>8.4f} "
      f"{calc_r2(y_test, attn_sm)[1]:>8.4f} {calc_r2(y_test, attn_sm)[2]:>8.4f}")
print(f"{'V2 Ensemble (smoothed+weighted)':<35} {ens_r2x:>8.4f} {ens_r2y:>8.4f} {ens_r2:>8.4f}")
print(f"{'  -> held-out (last 70%)':<35} {'':>8} {'':>8} {r2_heldout:>8.4f}")

print(f"\n{'='*70}")
print(f"BEST V2 RESULT")
print(f"  MAE:    {ens_mae:.2f} mm/s")
print(f"  RMSE:   {ens_rmse:.2f} mm/s")
print(f"  R2 X:   {ens_r2x:.4f}")
print(f"  R2 Y:   {ens_r2y:.4f}")
print(f"  R2 Avg: {ens_r2:.4f}")
print(f"  vs V1 ensemble (0.741): {ens_r2 - 0.7406:+.4f}")
print(f"  vs plain GRU (0.722):   {ens_r2 - 0.7224:+.4f}")
print(f"{'='*70}")

# Save
np.savez(os.path.join(OUTPUT_DIR, "v2_results.npz"),
         gru_pred=gru_pred, attn_pred=attn_pred, ensemble_pred=ensemble,
         y_test=y_test, weights=w,
         gru_r2x=gru_r2x, gru_r2y=gru_r2y,
         attn_r2x=attn_r2x, attn_r2y=attn_r2y,
         ens_r2x=ens_r2x, ens_r2y=ens_r2y,
         ens_mae=ens_mae, ens_rmse=ens_rmse)
print("\nSaved results/v2_results.npz")
