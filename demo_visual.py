"""
Visual Neural Decoder Demo

Shows two cursors moving on screen:
  - BLUE = actual hand movement (ground truth)
  - RED  = predicted from brain signals only

NOTE: OFFLINE replay of pre-recorded data.
NOT a real-time closed-loop brain-computer interface.
Neither cursor controls your real mouse.
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


# --- Load data and model ---
test = np.load("processed_data/test_data.npz")
X_test, y_test = test["X"], test["y"]

model = NeuralDecoder(X_test.shape[1])
model.load_state_dict(torch.load("results/nn_best.pt", weights_only=True))
model.eval()

# --- Precompute all predictions ---
with torch.no_grad():
    y_pred = model(torch.FloatTensor(X_test)).numpy()

# --- Build position trajectories by integrating velocity ---
BIN_S = 0.02
SCALE = 0.3  # scale positions to fit nicely on screen

actual_pos = np.cumsum(y_test * BIN_S * SCALE, axis=0)
pred_pos = np.cumsum(y_pred * BIN_S * SCALE, axis=0)

# Center both trajectories
actual_pos -= actual_pos.mean(axis=0)
pred_pos -= pred_pos.mean(axis=0)

# --- Set up the animation ---
STEP = 3         # advance 3 bins per frame (60ms per frame = ~10x speed)
TRAIL = 100      # show a trail of the last 100 positions
N = len(actual_pos)

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

xlim = max(abs(actual_pos[:, 0]).max(), abs(pred_pos[:, 0]).max()) * 1.1
ylim = max(abs(actual_pos[:, 1]).max(), abs(pred_pos[:, 1]).max()) * 1.1
ax.set_xlim(-xlim, xlim)
ax.set_ylim(-ylim, ylim)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

title = ax.set_title("Neural Decoder Demo — OFFLINE replay, not real-time BCI",
                      color="white", fontsize=13, fontweight="bold", pad=15)

# Trail lines
trail_actual, = ax.plot([], [], color="#4fc3f7", linewidth=1.5, alpha=0.5)
trail_pred, = ax.plot([], [], color="#ef5350", linewidth=1.5, alpha=0.5)

# Cursor dots
dot_actual, = ax.plot([], [], "o", color="#4fc3f7", markersize=18, zorder=5)
dot_pred, = ax.plot([], [], "o", color="#ef5350", markersize=18, zorder=5)

# Labels
ax.plot([], [], "o", color="#4fc3f7", markersize=10, label="Actual hand movement")
ax.plot([], [], "o", color="#ef5350", markersize=10, label="Brain-predicted movement")
legend = ax.legend(loc="upper left", fontsize=11, facecolor="#16213e",
                   edgecolor="gray", labelcolor="white")

time_text = ax.text(0.98, 0.02, "", transform=ax.transAxes, color="white",
                    fontsize=12, ha="right", va="bottom",
                    fontfamily="monospace")
speed_text = ax.text(0.98, 0.07, "", transform=ax.transAxes, color="#aaaaaa",
                     fontsize=10, ha="right", va="bottom",
                     fontfamily="monospace")


def init():
    trail_actual.set_data([], [])
    trail_pred.set_data([], [])
    dot_actual.set_data([], [])
    dot_pred.set_data([], [])
    time_text.set_text("")
    speed_text.set_text("")
    return trail_actual, trail_pred, dot_actual, dot_pred, time_text, speed_text


def update(frame):
    i = frame * STEP
    if i >= N:
        return trail_actual, trail_pred, dot_actual, dot_pred, time_text, speed_text

    start = max(0, i - TRAIL)

    trail_actual.set_data(actual_pos[start:i+1, 0], actual_pos[start:i+1, 1])
    trail_pred.set_data(pred_pos[start:i+1, 0], pred_pos[start:i+1, 1])

    dot_actual.set_data([actual_pos[i, 0]], [actual_pos[i, 1]])
    dot_pred.set_data([pred_pos[i, 0]], [pred_pos[i, 1]])

    t_sec = i * BIN_S
    time_text.set_text(f"t = {t_sec:.1f}s")

    speed = np.sqrt(y_test[i, 0]**2 + y_test[i, 1]**2)
    if speed < 10:
        speed_text.set_text("hand: resting")
        speed_text.set_color("#888888")
    elif speed < 100:
        speed_text.set_text(f"hand: moving ({speed:.0f} mm/s)")
        speed_text.set_color("#aaaaaa")
    else:
        speed_text.set_text(f"hand: FAST ({speed:.0f} mm/s)")
        speed_text.set_color("#ff9800")

    return trail_actual, trail_pred, dot_actual, dot_pred, time_text, speed_text


n_frames = N // STEP
ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init,
                              interval=30, blit=True, repeat=False)

plt.tight_layout()
print("Opening visual demo window...")
print("  BLUE dot = actual hand movement")
print("  RED dot  = predicted from brain signals")
print("  Close the window to stop.")
plt.show()
