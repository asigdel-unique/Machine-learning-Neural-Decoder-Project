import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load processed data ---
train = np.load("processed_data/train_data.npz")
test = np.load("processed_data/test_data.npz")

X_train, y_train = train["X"], train["y"]
X_test, y_test = test["X"], test["y"]

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")

# --- Baseline: predict training mean velocity for all test samples ---
print("\n" + "=" * 50)
print("BASELINE: Predict mean velocity")
print("=" * 50)

y_mean = y_train.mean(axis=0)
y_pred_baseline = np.tile(y_mean, (X_test.shape[0], 1))

mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))

mae_baseline_x = mean_absolute_error(y_test[:, 0], y_pred_baseline[:, 0])
mae_baseline_y = mean_absolute_error(y_test[:, 1], y_pred_baseline[:, 1])

print(f"  Mean velocity (train): [{y_mean[0]:.2f}, {y_mean[1]:.2f}]")
print(f"  MAE:  {mae_baseline:.4f} (X: {mae_baseline_x:.4f}, Y: {mae_baseline_y:.4f})")
print(f"  RMSE: {rmse_baseline:.4f}")

# --- Linear regression (Ridge) ---
print("\n" + "=" * 50)
print("LINEAR MODEL: Ridge Regression")
print("=" * 50)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_pred_ridge = model.predict(X_test)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

mae_ridge_x = mean_absolute_error(y_test[:, 0], y_pred_ridge[:, 0])
mae_ridge_y = mean_absolute_error(y_test[:, 1], y_pred_ridge[:, 1])

print(f"  MAE:  {mae_ridge:.4f} (X: {mae_ridge_x:.4f}, Y: {mae_ridge_y:.4f})")
print(f"  RMSE: {rmse_ridge:.4f}")

r2_x = 1 - np.sum((y_test[:, 0] - y_pred_ridge[:, 0])**2) / np.sum((y_test[:, 0] - y_test[:, 0].mean())**2)
r2_y = 1 - np.sum((y_test[:, 1] - y_pred_ridge[:, 1])**2) / np.sum((y_test[:, 1] - y_test[:, 1].mean())**2)
print(f"  R²:   X={r2_x:.4f}, Y={r2_y:.4f}")

print(f"\n  Improvement over baseline:")
print(f"    MAE reduction:  {(1 - mae_ridge/mae_baseline)*100:.1f}%")
print(f"    RMSE reduction: {(1 - rmse_ridge/rmse_baseline)*100:.1f}%")

# --- Plot: actual vs predicted ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

t_test = np.arange(len(y_test)) * 0.02  # 20 ms bins
plot_range = (0, 10)  # first 10 seconds of test set
mask = (t_test >= plot_range[0]) & (t_test <= plot_range[1])

for dim, dim_name in enumerate(["Velocity X", "Velocity Y"]):
    ax = axes[0, dim]
    ax.plot(t_test[mask], y_test[mask, dim], label="Actual", linewidth=0.8, alpha=0.8)
    ax.plot(t_test[mask], y_pred_ridge[mask, dim], label="Ridge prediction",
            linewidth=0.8, alpha=0.8)
    ax.axhline(y_mean[dim], color="gray", linestyle="--", alpha=0.5, label="Baseline (mean)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(dim_name)
    ax.set_title(f"{dim_name}: Actual vs Predicted (first 10s of test)")
    ax.legend()

# Scatter: predicted vs actual
for dim, dim_name in enumerate(["Velocity X", "Velocity Y"]):
    ax = axes[1, dim]
    subsample = max(1, len(y_test) // 2000)
    ax.scatter(y_test[::subsample, dim], y_pred_ridge[::subsample, dim],
               s=3, alpha=0.3)
    lims = [min(y_test[:, dim].min(), y_pred_ridge[:, dim].min()),
            max(y_test[:, dim].max(), y_pred_ridge[:, dim].max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    r2 = r2_x if dim == 0 else r2_y
    ax.set_title(f"{dim_name}: Predicted vs Actual (R²={r2:.3f})")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()

plt.suptitle("Ridge Regression Decoder", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/ridge_results.png", dpi=150)
plt.close()
print("\nSaved figures/ridge_results.png")

# --- Save model ---
with open(os.path.join(OUTPUT_DIR, "ridge_model.pkl"), "wb") as f:
    pickle.dump(model, f)

np.savez(
    os.path.join(OUTPUT_DIR, "ridge_results.npz"),
    y_test=y_test, y_pred=y_pred_ridge,
    mae=mae_ridge, rmse=rmse_ridge, r2_x=r2_x, r2_y=r2_y,
    mae_baseline=mae_baseline, rmse_baseline=rmse_baseline
)

print(f"Saved model to {OUTPUT_DIR}/ridge_model.pkl")
print(f"Saved results to {OUTPUT_DIR}/ridge_results.npz")
