"""
Improved decoder: LSTM + exponential smoothing + 300ms history.
Compares against the original feedforward NN and Ridge.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("figures", exist_ok=True)

# --- Load improved data ---
train = np.load("processed_data/improved_train.npz")
test = np.load("processed_data/improved_test.npz")

X_seq_train, y_train = train["X_seq"], train["y"]
X_seq_test, y_test = test["X_seq"], test["y"]
X_flat_train = train["X_flat"]
X_flat_test = test["X_flat"]

n_timesteps = X_seq_train.shape[1]
n_features = X_seq_train.shape[2]

# Validation split (last 15% of training)
val_split = int(len(X_seq_train) * 0.85)
X_seq_val = X_seq_train[val_split:]
X_flat_val = X_flat_train[val_split:]
y_val = y_train[val_split:]
X_seq_tr = X_seq_train[:val_split]
X_flat_tr = X_flat_train[:val_split]
y_tr = y_train[:val_split]

print(f"Train:      {len(y_tr)} samples")
print(f"Validation: {len(y_val)} samples")
print(f"Test:       {len(y_test)} samples")
print(f"Sequence:   {n_timesteps} timesteps x {n_features} neurons")


# ============================================================
# Model 1: Improved Feedforward (same arch, better data)
# ============================================================
class ImprovedFeedforward(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Model 2: LSTM Decoder
# ============================================================
class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden)


# ============================================================
# Model 3: GRU Decoder (lighter, often similar to LSTM)
# ============================================================
class GRUDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        return self.head(last_hidden)


# ============================================================
# Training function
# ============================================================
def train_model(model, train_loader, val_loader, name, n_epochs=150, patience=25):
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses, val_losses = [], []

    print(f"\nTraining {name}...")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_samples = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
            n_samples += len(X_batch)
        train_losses.append(epoch_loss / n_samples)

        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item() * len(X_batch)
                n_val += len(X_batch)
        val_losses.append(val_loss / n_val)

        scheduler.step()

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{name}_best.pt"))
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or patience_counter == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}: train={train_losses[-1]:.2f} val={val_losses[-1]:.2f} "
                  f"lr={lr:.6f}{' *best*' if patience_counter == 0 else ''}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best: epoch {best_epoch+1})")
            break

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"{name}_best.pt"), weights_only=True))
    return model, train_losses, val_losses, best_epoch


def evaluate(model, test_loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch)
            preds.append(pred.numpy())
            targets.append(y_batch.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_x = 1 - np.sum((y_true[:, 0] - y_pred[:, 0])**2) / np.sum((y_true[:, 0] - y_true[:, 0].mean())**2)
    r2_y = 1 - np.sum((y_true[:, 1] - y_pred[:, 1])**2) / np.sum((y_true[:, 1] - y_true[:, 1].mean())**2)
    return y_pred, mae, rmse, r2_x, r2_y


# ============================================================
# Train all three models
# ============================================================

# DataLoaders for sequence models (LSTM, GRU)
seq_train_ds = TensorDataset(torch.FloatTensor(X_seq_tr), torch.FloatTensor(y_tr))
seq_val_ds = TensorDataset(torch.FloatTensor(X_seq_val), torch.FloatTensor(y_val))
seq_test_ds = TensorDataset(torch.FloatTensor(X_seq_test), torch.FloatTensor(y_test))

seq_train_loader = DataLoader(seq_train_ds, batch_size=256, shuffle=True)
seq_val_loader = DataLoader(seq_val_ds, batch_size=512)
seq_test_loader = DataLoader(seq_test_ds, batch_size=512)

# DataLoaders for feedforward model
flat_train_ds = TensorDataset(torch.FloatTensor(X_flat_tr), torch.FloatTensor(y_tr))
flat_val_ds = TensorDataset(torch.FloatTensor(X_flat_val), torch.FloatTensor(y_val))
flat_test_ds = TensorDataset(torch.FloatTensor(X_flat_test), torch.FloatTensor(y_test))

flat_train_loader = DataLoader(flat_train_ds, batch_size=256, shuffle=True)
flat_val_loader = DataLoader(flat_val_ds, batch_size=512)
flat_test_loader = DataLoader(flat_test_ds, batch_size=512)

print("=" * 70)
print("TRAINING THREE IMPROVED MODELS")
print("=" * 70)

# 1. Improved Feedforward
ff_model = ImprovedFeedforward(X_flat_tr.shape[1])
ff_model, ff_train_loss, ff_val_loss, ff_best = train_model(
    ff_model, flat_train_loader, flat_val_loader, "improved_ff")
ff_pred, ff_mae, ff_rmse, ff_r2x, ff_r2y = evaluate(ff_model, flat_test_loader)

# 2. LSTM
lstm_model = LSTMDecoder(n_features, hidden_dim=256, n_layers=2, dropout=0.3)
lstm_model, lstm_train_loss, lstm_val_loss, lstm_best = train_model(
    lstm_model, seq_train_loader, seq_val_loader, "lstm")
lstm_pred, lstm_mae, lstm_rmse, lstm_r2x, lstm_r2y = evaluate(lstm_model, seq_test_loader)

# 3. GRU
gru_model = GRUDecoder(n_features, hidden_dim=256, n_layers=2, dropout=0.3)
gru_model, gru_train_loss, gru_val_loss, gru_best = train_model(
    gru_model, seq_train_loader, seq_val_loader, "gru")
gru_pred, gru_mae, gru_rmse, gru_r2x, gru_r2y = evaluate(gru_model, seq_test_loader)

# --- Load original results for comparison ---
orig_nn = np.load("results/nn_results.npz")
orig_ridge = np.load("results/ridge_results.npz")

# ============================================================
# Results comparison
# ============================================================
print("\n" + "=" * 70)
print("FULL COMPARISON: Original vs Improved Models")
print("=" * 70)
print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R² X':>8} {'R² Y':>8} {'R² Avg':>8}")
print("-" * 70)

results = [
    ("Original Ridge", float(orig_ridge["mae"]), float(orig_ridge["rmse"]),
     float(orig_ridge["r2_x"]), float(orig_ridge["r2_y"])),
    ("Original NN (6 bins, no smooth)", float(orig_nn["mae"]), float(orig_nn["rmse"]),
     float(orig_nn["r2_x"]), float(orig_nn["r2_y"])),
    ("Improved FF (15 bins+smooth)", ff_mae, ff_rmse, ff_r2x, ff_r2y),
    ("LSTM (15 bins+smooth)", lstm_mae, lstm_rmse, lstm_r2x, lstm_r2y),
    ("GRU (15 bins+smooth)", gru_mae, gru_rmse, gru_r2x, gru_r2y),
]

best_r2 = 0
best_name = ""
for name, mae, rmse, r2x, r2y in results:
    r2_avg = (r2x + r2y) / 2
    print(f"{name:<30} {mae:>8.2f} {rmse:>8.2f} {r2x:>8.3f} {r2y:>8.3f} {r2_avg:>8.3f}")
    if r2_avg > best_r2:
        best_r2 = r2_avg
        best_name = name

orig_r2 = (float(orig_nn["r2_x"]) + float(orig_nn["r2_y"])) / 2
improvement = ((best_r2 - orig_r2) / orig_r2) * 100

print(f"\nBest model: {best_name}")
print(f"R² improvement over original NN: {orig_r2:.3f} -> {best_r2:.3f} ({improvement:+.1f}%)")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

t_test = np.arange(len(y_test)) * 0.02
mask = t_test <= 10

# Row 1: Velocity X time series
ax = axes[0, 0]
ax.plot(t_test[mask], y_test[mask, 0], label="Actual", linewidth=0.8, color="black")
ax.plot(t_test[mask], ff_pred[mask, 0], label="Improved FF", linewidth=0.8, alpha=0.7)
ax.plot(t_test[mask], lstm_pred[mask, 0], label="LSTM", linewidth=0.8, alpha=0.7)
ax.plot(t_test[mask], gru_pred[mask, 0], label="GRU", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity X")
ax.set_title("Velocity X Predictions (first 10s)")
ax.legend(fontsize=8)

# Row 1: Velocity Y time series
ax = axes[0, 1]
ax.plot(t_test[mask], y_test[mask, 1], label="Actual", linewidth=0.8, color="black")
ax.plot(t_test[mask], ff_pred[mask, 1], label="Improved FF", linewidth=0.8, alpha=0.7)
ax.plot(t_test[mask], lstm_pred[mask, 1], label="LSTM", linewidth=0.8, alpha=0.7)
ax.plot(t_test[mask], gru_pred[mask, 1], label="GRU", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity Y")
ax.set_title("Velocity Y Predictions (first 10s)")
ax.legend(fontsize=8)

# Row 1: Bar chart comparison
ax = axes[0, 2]
model_names = ["Ridge", "Orig NN", "Impr FF", "LSTM", "GRU"]
r2_avgs = [(r2x + r2y) / 2 for _, _, _, r2x, r2y in results]
colors = ["#7f8c8d", "#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
bars = ax.bar(model_names, r2_avgs, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Average R²")
ax.set_title("Model Comparison (R²)")
ax.set_ylim(0, max(r2_avgs) * 1.15)
for bar, val in zip(bars, r2_avgs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

# Row 2: Training curves
for idx, (name, tl, vl) in enumerate([
    ("Improved FF", ff_train_loss, ff_val_loss),
    ("LSTM", lstm_train_loss, lstm_val_loss),
    ("GRU", gru_train_loss, gru_val_loss),
]):
    ax = axes[1, idx]
    ax.plot(tl, label="Train", linewidth=0.8)
    ax.plot(vl, label="Val", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{name} Training Curve")
    ax.legend()

plt.suptitle("Improved Neural Decoders: Smoothing + 300ms History + LSTM/GRU",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/improved_comparison.png", dpi=150)
plt.close()
print("\nSaved figures/improved_comparison.png")

# Save results
np.savez(os.path.join(OUTPUT_DIR, "improved_results.npz"),
         ff_pred=ff_pred, lstm_pred=lstm_pred, gru_pred=gru_pred,
         y_test=y_test,
         ff_r2x=ff_r2x, ff_r2y=ff_r2y, ff_mae=ff_mae,
         lstm_r2x=lstm_r2x, lstm_r2y=lstm_r2y, lstm_mae=lstm_mae,
         gru_r2x=gru_r2x, gru_r2y=gru_r2y, gru_mae=gru_mae)
print("Saved results/improved_results.npz")
