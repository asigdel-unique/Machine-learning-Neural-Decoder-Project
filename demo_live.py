"""
Live-feed Neural Decoder Demo

Replays brain recordings one window at a time and prints
predicted vs actual hand movement to the console.

NOTE: This is OFFLINE replay of pre-recorded data,
NOT a real-time closed-loop brain-computer interface.
"""
import numpy as np
import torch
import torch.nn as nn
import time
import sys


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


# --- Load ---
test = np.load("processed_data/test_data.npz")
X_test, y_test = test["X"], test["y"]

model = NeuralDecoder(X_test.shape[1])
model.load_state_dict(torch.load("results/nn_best.pt", weights_only=True))
model.eval()

BIN_MS = 20
PLAYBACK_SPEED = 50  # how many real ms per bin (50 = 2.5x slower than real time)
N = len(X_test)

print("=" * 65)
print("  LIVE-FEED NEURAL DECODER DEMO (offline replay)")
print("  Reading brain signals and predicting hand movement...")
print("  Press Ctrl+C to stop")
print("=" * 65)
print()
print(f"  {'Time':>7}  {'Actual Vel X':>12} {'Actual Vel Y':>12}  |  "
      f"{'Pred Vel X':>10} {'Pred Vel Y':>10}  {'Match':>6}")
print("-" * 80)

try:
    for i in range(N):
        t_sec = i * BIN_MS / 1000

        x = torch.FloatTensor(X_test[i:i + 1])
        with torch.no_grad():
            pred = model(x).numpy()[0]

        actual = y_test[i]
        speed_actual = np.sqrt(actual[0]**2 + actual[1]**2)
        speed_pred = np.sqrt(pred[0]**2 + pred[1]**2)

        if speed_actual < 10:
            status = "  rest"
        elif abs(pred[0] - actual[0]) < 40 and abs(pred[1] - actual[1]) < 40:
            status = "  GOOD"
        else:
            status = "  miss"

        # Simple bar showing movement magnitude
        bar_len = int(min(speed_actual, 300) / 10)
        bar = "#" * bar_len

        print(f"  {t_sec:6.2f}s  {actual[0]:>+10.1f}  {actual[1]:>+10.1f}   | "
              f" {pred[0]:>+10.1f}  {pred[1]:>+10.1f} {status}  {bar}")

        time.sleep(PLAYBACK_SPEED / 1000)

except KeyboardInterrupt:
    print("\n\n  Stopped by user.")

print(f"\n  Replayed {min(i + 1, N)} of {N} windows "
      f"({min(i + 1, N) * BIN_MS / 1000:.1f}s of data)")
print("  This was OFFLINE replay — not a real-time BCI.")
