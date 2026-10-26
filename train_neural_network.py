import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load data ---
train = np.load("processed_data/train_data.npz")
test = np.load("processed_data/test_data.npz")
X_train, y_train = train["X"], train["y"]
X_test, y_test = test["X"], test["y"]

# Split training into train/validation (last 15% of training for validation)
val_split = int(len(X_train) * 0.85)
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_tr, y_tr = X_train[:val_split], y_train[:val_split]

print(f"Train:      {X_tr.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test:       {X_test.shape[0]} samples")

# --- PyTorch datasets ---
train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512)
test_loader = DataLoader(test_ds, batch_size=512)


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


# --- Training ---
device = torch.device("cpu")
model = NeuralDecoder(X_tr.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.MSELoss()

N_EPOCHS = 100
PATIENCE = 20

train_losses = []
val_losses = []
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

print(f"\nTraining for up to {N_EPOCHS} epochs (early stopping patience={PATIENCE})...")

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(X_batch)
    train_losses.append(epoch_loss / len(train_ds))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            val_loss += criterion(pred, y_batch).item() * len(X_batch)
    val_losses.append(val_loss / len(val_ds))

    scheduler.step(val_losses[-1])

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        best_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "nn_best.pt"))
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0 or patience_counter == 0:
        print(f"  Epoch {epoch+1:3d}: train_loss={train_losses[-1]:.2f}, "
              f"val_loss={val_losses[-1]:.2f}"
              f"{' *best*' if patience_counter == 0 else ''}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (best was epoch {best_epoch+1})")
        break

# --- Load best model and evaluate on test ---
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "nn_best.pt"), weights_only=True))
model.eval()

all_preds = []
all_targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        pred = model(X_batch.to(device))
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y_batch.numpy())

y_pred_nn = np.concatenate(all_preds)
y_test_arr = np.concatenate(all_targets)

mae_nn = mean_absolute_error(y_test_arr, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test_arr, y_pred_nn))
r2_x = 1 - np.sum((y_test_arr[:, 0] - y_pred_nn[:, 0])**2) / np.sum((y_test_arr[:, 0] - y_test_arr[:, 0].mean())**2)
r2_y = 1 - np.sum((y_test_arr[:, 1] - y_pred_nn[:, 1])**2) / np.sum((y_test_arr[:, 1] - y_test_arr[:, 1].mean())**2)

# --- Load Ridge results for comparison ---
ridge_results = np.load("results/ridge_results.npz")
mae_ridge = float(ridge_results["mae"])
rmse_ridge = float(ridge_results["rmse"])
r2_ridge_x = float(ridge_results["r2_x"])
r2_ridge_y = float(ridge_results["r2_y"])
y_pred_ridge = ridge_results["y_pred"]

print("\n" + "=" * 60)
print("COMPARISON: Ridge vs Neural Network")
print("=" * 60)
print(f"{'Metric':<20} {'Ridge':>12} {'NN':>12} {'Winner':>10}")
print("-" * 60)
print(f"{'MAE':<20} {mae_ridge:>12.4f} {mae_nn:>12.4f} {'NN' if mae_nn < mae_ridge else 'Ridge':>10}")
print(f"{'RMSE':<20} {rmse_ridge:>12.4f} {rmse_nn:>12.4f} {'NN' if rmse_nn < rmse_ridge else 'Ridge':>10}")
print(f"{'R² (Vel X)':<20} {r2_ridge_x:>12.4f} {r2_x:>12.4f} {'NN' if r2_x > r2_ridge_x else 'Ridge':>10}")
print(f"{'R² (Vel Y)':<20} {r2_ridge_y:>12.4f} {r2_y:>12.4f} {'NN' if r2_y > r2_ridge_y else 'Ridge':>10}")

nn_wins = sum([mae_nn < mae_ridge, rmse_nn < rmse_ridge, r2_x > r2_ridge_x, r2_y > r2_ridge_y])
if nn_wins >= 3:
    print("\n>> Neural network is the better model.")
    best_model_name = "neural_network"
else:
    print("\n>> Ridge regression is the better (or comparable) model — keeping the simpler model.")
    best_model_name = "ridge"

# --- Plot ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Training curves
ax = axes[0, 0]
ax.plot(train_losses, label="Train loss")
ax.plot(val_losses, label="Val loss")
ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best epoch ({best_epoch+1})")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training & Validation Loss")
ax.legend()

# Time series comparison
t_test = np.arange(len(y_test_arr)) * 0.02
mask = t_test <= 10
ax = axes[0, 1]
ax.plot(t_test[mask], y_test_arr[mask, 0], label="Actual", linewidth=0.8)
ax.plot(t_test[mask], y_pred_ridge[mask, 0], label="Ridge", linewidth=0.8, alpha=0.7)
ax.plot(t_test[mask], y_pred_nn[mask, 0], label="NN", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity X")
ax.set_title("Velocity X: Ridge vs NN (first 10s of test)")
ax.legend()

ax = axes[1, 0]
ax.plot(t_test[mask], y_test_arr[mask, 1], label="Actual", linewidth=0.8)
ax.plot(t_test[mask], y_pred_ridge[mask, 1], label="Ridge", linewidth=0.8, alpha=0.7)
ax.plot(t_test[mask], y_pred_nn[mask, 1], label="NN", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity Y")
ax.set_title("Velocity Y: Ridge vs NN (first 10s of test)")
ax.legend()

# Bar comparison
ax = axes[1, 1]
metrics = ["MAE", "RMSE", "R² X", "R² Y"]
ridge_vals = [mae_ridge, rmse_ridge, r2_ridge_x, r2_ridge_y]
nn_vals = [mae_nn, rmse_nn, r2_x, r2_y]
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, ridge_vals, w, label="Ridge", color="steelblue")
ax.bar(x + w/2, nn_vals, w, label="Neural Network", color="coral")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title("Model Comparison")
ax.legend()

plt.suptitle("Neural Network vs Ridge Regression", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/model_comparison.png", dpi=150)
plt.close()
print("\nSaved figures/model_comparison.png")

np.savez(
    os.path.join(OUTPUT_DIR, "nn_results.npz"),
    y_test=y_test_arr, y_pred=y_pred_nn,
    mae=mae_nn, rmse=rmse_nn, r2_x=r2_x, r2_y=r2_y,
    best_model=best_model_name
)
print(f"Best model selected: {best_model_name}")
