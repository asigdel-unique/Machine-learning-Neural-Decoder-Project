import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- Load results ---
ridge = np.load("results/ridge_results.npz")
nn = np.load("results/nn_results.npz")

y_test = nn["y_test"]
y_pred_ridge = ridge["y_pred"]
y_pred_nn = nn["y_pred"]
t = np.arange(len(y_test)) * 0.02

# --- Final comparison figure ---
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

# Row 1: Time series comparison (10s window)
mask = t <= 10
for dim, name in enumerate(["Velocity X", "Velocity Y"]):
    ax = fig.add_subplot(gs[0, dim])
    ax.plot(t[mask], y_test[mask, dim], label="Actual", linewidth=1.2, color="#2c3e50")
    ax.plot(t[mask], y_pred_ridge[mask, dim], label="Ridge", linewidth=1, alpha=0.8, color="#3498db")
    ax.plot(t[mask], y_pred_nn[mask, dim], label="NN", linewidth=1, alpha=0.8, color="#e74c3c")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{name} (mm/s)")
    ax.set_title(f"{name}: Model Comparison")
    ax.legend(fontsize=8)

# Row 1 col 3: Bar chart
ax = fig.add_subplot(gs[0, 2])
metrics = ["MAE", "RMSE"]
ridge_vals = [float(ridge["mae"]), float(ridge["rmse"])]
nn_vals = [float(nn["mae"]), float(nn["rmse"])]
baseline_vals = [float(ridge["mae_baseline"]), float(ridge["rmse_baseline"])]

x = np.arange(len(metrics))
w = 0.25
ax.bar(x - w, baseline_vals, w, label="Baseline", color="#95a5a6")
ax.bar(x, ridge_vals, w, label="Ridge", color="#3498db")
ax.bar(x + w, nn_vals, w, label="Neural Net", color="#e74c3c")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_title("Error Comparison")
ax.legend(fontsize=8)

# Row 2: Scatter plots
for dim, name in enumerate(["Velocity X", "Velocity Y"]):
    ax = fig.add_subplot(gs[1, dim])
    sub = max(1, len(y_test) // 2000)
    ax.scatter(y_test[::sub, dim], y_pred_ridge[::sub, dim], s=5, alpha=0.3,
               color="#3498db", label="Ridge")
    ax.scatter(y_test[::sub, dim], y_pred_nn[::sub, dim], s=5, alpha=0.3,
               color="#e74c3c", label="NN")
    lims = [y_test[:, dim].min(), y_test[:, dim].max()]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel(f"Actual {name}")
    ax.set_ylabel(f"Predicted {name}")
    ax.set_title(f"{name}: Predicted vs Actual")
    ax.legend(fontsize=8)

# Row 2 col 3: R² comparison
ax = fig.add_subplot(gs[1, 2])
r2_labels = ["R² Vel X", "R² Vel Y"]
r2_ridge = [float(ridge["r2_x"]), float(ridge["r2_y"])]
r2_nn = [float(nn["r2_x"]), float(nn["r2_y"])]
x = np.arange(2)
ax.bar(x - 0.15, r2_ridge, 0.3, label="Ridge", color="#3498db")
ax.bar(x + 0.15, r2_nn, 0.3, label="Neural Net", color="#e74c3c")
ax.set_xticks(x)
ax.set_xticklabels(r2_labels)
ax.set_ylabel("R²")
ax.set_title("Variance Explained")
ax.set_ylim(0, 0.7)
ax.legend(fontsize=8)

# Row 3: 2D trajectory and summary table
actual_pos = np.cumsum(y_test * 0.02, axis=0)
ridge_pos = np.cumsum(y_pred_ridge * 0.02, axis=0)
nn_pos = np.cumsum(y_pred_nn * 0.02, axis=0)

seg = slice(0, 1500)
ax = fig.add_subplot(gs[2, 0:2])
ax.plot(actual_pos[seg, 0], actual_pos[seg, 1], linewidth=1, color="#2c3e50",
        label="Actual", alpha=0.8)
ax.plot(ridge_pos[seg, 0], ridge_pos[seg, 1], linewidth=1, color="#3498db",
        label="Ridge", alpha=0.6)
ax.plot(nn_pos[seg, 0], nn_pos[seg, 1], linewidth=1, color="#e74c3c",
        label="NN", alpha=0.6)
ax.set_xlabel("X position (integrated)")
ax.set_ylabel("Y position (integrated)")
ax.set_title("2D Trajectory Reconstruction (first 30s)")
ax.legend(fontsize=8)
ax.set_aspect("equal")

# Summary table
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
table_data = [
    ["", "Baseline", "Ridge", "NN"],
    ["MAE", f"{float(ridge['mae_baseline']):.1f}",
     f"{float(ridge['mae']):.1f}", f"{float(nn['mae']):.1f}"],
    ["RMSE", f"{float(ridge['rmse_baseline']):.1f}",
     f"{float(ridge['rmse']):.1f}", f"{float(nn['rmse']):.1f}"],
    ["R² X", "—", f"{float(ridge['r2_x']):.3f}", f"{float(nn['r2_x']):.3f}"],
    ["R² Y", "—", f"{float(ridge['r2_y']):.3f}", f"{float(nn['r2_y']):.3f}"],
]
table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
for j in range(4):
    table[0, j].set_facecolor("#ecf0f1")
    table[0, j].set_text_props(fontweight="bold")
ax.set_title("Results Summary", pad=20)

fig.suptitle("Neural Decoder: Baseline vs Ridge Regression vs Neural Network\n"
             "MC_RTT Dataset — Macaque M1, 96-channel Utah Array",
             fontsize=15, fontweight="bold")
plt.savefig("figures/final_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/final_comparison.png")
