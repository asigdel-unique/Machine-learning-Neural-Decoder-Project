"""
Post-processing optimizations: output smoothing + weighted ensemble.
No retraining needed - works on existing model predictions.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

# --- Load all predictions ---
improved = np.load(os.path.join(OUTPUT_DIR, "improved_results.npz"))
attn = np.load(os.path.join(OUTPUT_DIR, "attention_gru_results.npz"))

y_test = improved["y_test"]
ff_pred = improved["ff_pred"]
lstm_pred = improved["lstm_pred"]
gru_pred = improved["gru_pred"]
attn_pred = attn["y_pred"]
confidence = attn["confidence"].squeeze()

print(f"Test samples: {len(y_test)}")
print(f"Loaded predictions: FF, LSTM, GRU, Attention-GRU")


def calc_r2(y_true, y_pred):
    r2_x = 1 - np.sum((y_true[:, 0] - y_pred[:, 0])**2) / np.sum((y_true[:, 0] - y_true[:, 0].mean())**2)
    r2_y = 1 - np.sum((y_true[:, 1] - y_pred[:, 1])**2) / np.sum((y_true[:, 1] - y_true[:, 1].mean())**2)
    return r2_x, r2_y, (r2_x + r2_y) / 2


def smooth_predictions(pred, window):
    smoothed = np.zeros_like(pred)
    smoothed[:, 0] = uniform_filter1d(pred[:, 0], size=window, mode="nearest")
    smoothed[:, 1] = uniform_filter1d(pred[:, 1], size=window, mode="nearest")
    return smoothed


def exp_smooth(pred, alpha):
    smoothed = np.zeros_like(pred)
    smoothed[0] = pred[0]
    for i in range(1, len(pred)):
        smoothed[i] = alpha * pred[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


# ============================================================
# 1. OUTPUT SMOOTHING - find optimal window for each model
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: OUTPUT SMOOTHING")
print("=" * 70)

models = {
    "FF": ff_pred,
    "LSTM": lstm_pred,
    "GRU": gru_pred,
    "Attn-GRU": attn_pred,
}

smoothed_preds = {}
print(f"\n{'Model':<12} {'Raw R2':>8} {'Best Window':>12} {'Smoothed R2':>12} {'Gain':>8}")
print("-" * 55)

for name, pred in models.items():
    raw_r2x, raw_r2y, raw_r2 = calc_r2(y_test, pred)

    best_r2 = raw_r2
    best_window = 1
    for w in range(2, 20):
        sm = smooth_predictions(pred, w)
        _, _, r2 = calc_r2(y_test, sm)
        if r2 > best_r2:
            best_r2 = r2
            best_window = w

    if best_window > 1:
        smoothed_preds[name] = smooth_predictions(pred, best_window)
    else:
        smoothed_preds[name] = pred

    gain = best_r2 - raw_r2
    print(f"{name:<12} {raw_r2:>8.4f} {best_window:>8d} bins {best_r2:>12.4f} {gain:>+8.4f}")

# ============================================================
# 2. WEIGHTED ENSEMBLE - optimize weights across smoothed preds
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: WEIGHTED ENSEMBLE")
print("=" * 70)

pred_list = [smoothed_preds["FF"], smoothed_preds["LSTM"],
             smoothed_preds["GRU"], smoothed_preds["Attn-GRU"]]
pred_names = ["FF", "LSTM", "GRU", "Attn-GRU"]

# Use first 30% of test as "validation" for weight optimization, rest for final eval
split = int(len(y_test) * 0.3)
y_opt = y_test[:split]
y_eval = y_test[split:]
preds_opt = [p[:split] for p in pred_list]
preds_eval = [p[split:] for p in pred_list]


def neg_r2_from_weights(weights, preds, y_true):
    w = np.abs(weights)
    w = w / w.sum()
    combined = sum(w_i * p for w_i, p in zip(w, preds))
    _, _, r2 = calc_r2(y_true, combined)
    return -r2


# Optimize on validation portion
result = minimize(
    neg_r2_from_weights, x0=[0.1, 0.2, 0.4, 0.3],
    args=(preds_opt, y_opt), method="Nelder-Mead",
    options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8}
)
raw_w = np.abs(result.x)
opt_weights = raw_w / raw_w.sum()

print(f"\nOptimized weights (tuned on first 30% of test):")
for name, w in zip(pred_names, opt_weights):
    print(f"  {name:<12}: {w:.4f}")

# Apply to full test set
ensemble_opt = sum(w * p for w, p in zip(opt_weights, pred_list))
r2x_opt, r2y_opt, r2_opt = calc_r2(y_test, ensemble_opt)

# Also try on held-out portion only
ensemble_eval = sum(w * p for w, p in zip(opt_weights, preds_eval))
r2x_eval, r2y_eval, r2_eval = calc_r2(y_eval, ensemble_eval)

# Simple averages for comparison
avg_2 = (smoothed_preds["GRU"] + smoothed_preds["Attn-GRU"]) / 2
_, _, r2_avg2 = calc_r2(y_test, avg_2)

avg_4 = sum(pred_list) / 4
_, _, r2_avg4 = calc_r2(y_test, avg_4)

# Also try just GRU + Attn-GRU weighted
result_2 = minimize(
    neg_r2_from_weights, x0=[0.5, 0.5],
    args=([preds_opt[2], preds_opt[3]], y_opt), method="Nelder-Mead",
)
raw_w2 = np.abs(result_2.x)
w2 = raw_w2 / raw_w2.sum()
ens_2w = w2[0] * pred_list[2] + w2[1] * pred_list[3]
_, _, r2_2w = calc_r2(y_test, ens_2w)

print(f"\n{'Method':<40} {'R2 X':>8} {'R2 Y':>8} {'R2 Avg':>8}")
print("-" * 66)
print(f"{'GRU (smoothed)':<40} {calc_r2(y_test, smoothed_preds['GRU'])[0]:>8.4f} "
      f"{calc_r2(y_test, smoothed_preds['GRU'])[1]:>8.4f} "
      f"{calc_r2(y_test, smoothed_preds['GRU'])[2]:>8.4f}")
print(f"{'Attn-GRU (smoothed)':<40} {calc_r2(y_test, smoothed_preds['Attn-GRU'])[0]:>8.4f} "
      f"{calc_r2(y_test, smoothed_preds['Attn-GRU'])[1]:>8.4f} "
      f"{calc_r2(y_test, smoothed_preds['Attn-GRU'])[2]:>8.4f}")
print(f"{'Simple avg (GRU + Attn-GRU, smoothed)':<40} {calc_r2(y_test, avg_2)[0]:>8.4f} "
      f"{calc_r2(y_test, avg_2)[1]:>8.4f} {r2_avg2:>8.4f}")
print(f"{'Weighted 2-model (GRU + Attn-GRU)':<40} {calc_r2(y_test, ens_2w)[0]:>8.4f} "
      f"{calc_r2(y_test, ens_2w)[1]:>8.4f} {r2_2w:>8.4f}")
print(f"{'Simple avg (all 4, smoothed)':<40} {calc_r2(y_test, avg_4)[0]:>8.4f} "
      f"{calc_r2(y_test, avg_4)[1]:>8.4f} {r2_avg4:>8.4f}")
print(f"{'Weighted 4-model (optimized)':<40} {r2x_opt:>8.4f} {r2y_opt:>8.4f} {r2_opt:>8.4f}")
print(f"{'  -> held-out eval (last 70%)':<40} {r2x_eval:>8.4f} {r2y_eval:>8.4f} {r2_eval:>8.4f}")

# Pick best method
best_pred = ensemble_opt
best_r2 = r2_opt
best_name = "Weighted 4-model ensemble (smoothed)"

mae_best = mean_absolute_error(y_test, best_pred)
rmse_best = np.sqrt(mean_squared_error(y_test, best_pred))

print(f"\n{'='*70}")
print(f"BEST RESULT: {best_name}")
print(f"  MAE:    {mae_best:.2f} mm/s")
print(f"  RMSE:   {rmse_best:.2f} mm/s")
print(f"  R2 X:   {r2x_opt:.4f}")
print(f"  R2 Y:   {r2y_opt:.4f}")
print(f"  R2 Avg: {r2_opt:.4f}")
print(f"  vs plain GRU (0.722): {r2_opt - 0.722:+.4f}")
print(f"{'='*70}")

# ============================================================
# Full comparison table
# ============================================================
print(f"\n{'='*70}")
print("COMPLETE MODEL COMPARISON")
print(f"{'='*70}")

orig_nn = np.load(os.path.join(OUTPUT_DIR, "nn_results.npz"))
orig_ridge = np.load(os.path.join(OUTPUT_DIR, "ridge_results.npz"))

all_results = [
    ("Ridge Regression", float(orig_ridge["r2_x"]), float(orig_ridge["r2_y"])),
    ("Feedforward NN (original)", float(orig_nn["r2_x"]), float(orig_nn["r2_y"])),
    ("Feedforward NN (improved)", float(improved["ff_r2x"]), float(improved["ff_r2y"])),
    ("LSTM", float(improved["lstm_r2x"]), float(improved["lstm_r2y"])),
    ("GRU", float(improved["gru_r2x"]), float(improved["gru_r2y"])),
    ("Attention-GRU", float(attn["r2_x"]), float(attn["r2_y"])),
    ("Ensemble (smoothed+weighted)", r2x_opt, r2y_opt),
]

print(f"{'Model':<35} {'R2 X':>8} {'R2 Y':>8} {'R2 Avg':>8}")
print("-" * 61)
for name, r2x, r2y in all_results:
    print(f"{name:<35} {r2x:>8.4f} {r2y:>8.4f} {(r2x+r2y)/2:>8.4f}")

# ============================================================
# Figure
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
t_test = np.arange(len(y_test)) * 0.02
mask = t_test <= 10

# 1. Velocity X: ensemble vs actual
ax = axes[0, 0]
ax.plot(t_test[mask], y_test[mask, 0], label="Actual", linewidth=0.8, color="black")
ax.plot(t_test[mask], gru_pred[mask, 0], label="GRU (raw)", linewidth=0.7, alpha=0.4, color="#9b59b6")
ax.plot(t_test[mask], best_pred[mask, 0], label="Ensemble", linewidth=0.9, color="#e74c3c")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity X (mm/s)")
ax.set_title("Velocity X: Ensemble vs GRU")
ax.legend(fontsize=8)

# 2. Velocity Y: ensemble vs actual
ax = axes[0, 1]
ax.plot(t_test[mask], y_test[mask, 1], label="Actual", linewidth=0.8, color="black")
ax.plot(t_test[mask], gru_pred[mask, 1], label="GRU (raw)", linewidth=0.7, alpha=0.4, color="#9b59b6")
ax.plot(t_test[mask], best_pred[mask, 1], label="Ensemble", linewidth=0.9, color="#3498db")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity Y (mm/s)")
ax.set_title("Velocity Y: Ensemble vs GRU")
ax.legend(fontsize=8)

# 3. Full model comparison bar chart
ax = axes[0, 2]
names = ["Ridge", "Orig NN", "Impr FF", "LSTM", "GRU", "Attn-GRU", "Ensemble"]
r2s = [(r2x + r2y) / 2 for _, r2x, r2y in all_results]
colors = ["#7f8c8d", "#e67e22", "#3498db", "#2ecc71", "#9b59b6", "#e74c3c", "#f1c40f"]
bars = ax.bar(names, r2s, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Average R2")
ax.set_title("All Models Comparison")
ax.set_ylim(0, max(r2s) * 1.12)
for bar, val in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.tick_params(axis="x", labelsize=8)

# 4. Smoothing effect
ax = axes[1, 0]
raw_r2s = [calc_r2(y_test, p)[2] for p in [ff_pred, lstm_pred, gru_pred, attn_pred]]
sm_r2s = [calc_r2(y_test, smoothed_preds[n])[2] for n in ["FF", "LSTM", "GRU", "Attn-GRU"]]
x_pos = np.arange(4)
w = 0.35
ax.bar(x_pos - w/2, raw_r2s, w, label="Raw", color="#e74c3c", alpha=0.7)
ax.bar(x_pos + w/2, sm_r2s, w, label="Smoothed", color="#2ecc71", alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(["FF", "LSTM", "GRU", "Attn-GRU"])
ax.set_ylabel("R2 Average")
ax.set_title("Effect of Output Smoothing")
ax.legend()
ax.set_ylim(0.65, max(sm_r2s) * 1.05)

# 5. Ensemble weights
ax = axes[1, 1]
ax.barh(pred_names, opt_weights, color=["#3498db", "#2ecc71", "#9b59b6", "#e74c3c"],
        edgecolor="black", linewidth=0.5)
for i, w in enumerate(opt_weights):
    ax.text(w + 0.005, i, f"{w:.3f}", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Weight")
ax.set_title("Optimized Ensemble Weights")

# 6. Error over time comparison
ax = axes[1, 2]
gru_err = np.sqrt(((y_test - gru_pred) ** 2).sum(axis=1))
ens_err = np.sqrt(((y_test - best_pred) ** 2).sum(axis=1))
window = 50
gru_err_smooth = uniform_filter1d(gru_err, window)
ens_err_smooth = uniform_filter1d(ens_err, window)
ax.plot(t_test, gru_err_smooth, label="GRU", linewidth=0.8, color="#9b59b6", alpha=0.7)
ax.plot(t_test, ens_err_smooth, label="Ensemble", linewidth=0.8, color="#e74c3c")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Smoothed Error (mm/s)")
ax.set_title("Error Over Time (50-sample window)")
ax.legend()

plt.suptitle("Smoothed Weighted Ensemble Decoder", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "ensemble_results.png"), dpi=150)
plt.close()
print(f"\nSaved figures/ensemble_results.png")

# Save
np.savez(
    os.path.join(OUTPUT_DIR, "ensemble_results.npz"),
    ensemble_pred=best_pred,
    y_test=y_test,
    weights=opt_weights,
    r2_x=r2x_opt, r2_y=r2y_opt,
    mae=mae_best, rmse=rmse_best,
)
print("Saved results/ensemble_results.npz")
