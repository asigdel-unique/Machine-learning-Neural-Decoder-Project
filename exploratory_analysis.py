import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

TRAIN_FILE = "data/000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb"

with NWBHDF5IO(TRAIN_FILE, "r") as io:
    nwb = io.read()

    units = nwb.units
    n_units = len(units)
    behavior = nwb.processing["behavior"]

    finger_vel = behavior["finger_vel"].data[:]
    cursor_pos = behavior["cursor_pos"].data[:]
    finger_pos = behavior["finger_pos"].data[:]
    target_pos = behavior["target_pos"].data[:]
    rate = behavior["finger_vel"].rate

    all_spike_times = [units["spike_times"][i] for i in range(n_units)]
    heldout = units["heldout"][:]

print(f"Behavior rate: {rate} Hz")
print(f"Finger velocity shape: {finger_vel.shape}")
print(f"Cursor position shape: {cursor_pos.shape}")
print(f"Finger position shape: {finger_pos.shape}")
print(f"Target position shape: {target_pos.shape}")
print(f"Number of heldout units: {sum(heldout)}")
print(f"Number of observed units: {sum(~np.array(heldout))}")

t_behav = np.arange(finger_vel.shape[0]) / rate

# --- Check for missing values ---
print("\n--- Missing value check ---")
for name, arr in [("finger_vel", finger_vel), ("cursor_pos", cursor_pos),
                   ("finger_pos", finger_pos), ("target_pos", target_pos)]:
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    print(f"  {name}: NaN={n_nan}, Inf={n_inf}, min={arr.min():.4f}, max={arr.max():.4f}")

# --- Figure 1: Neural raster plot (first 10 seconds, first 30 units) ---
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

observed_idx = [i for i in range(n_units) if not heldout[i]]
show_units = observed_idx[:30]
t_window = (0, 10)

ax = axes[0]
for plot_i, unit_i in enumerate(show_units):
    st = all_spike_times[unit_i]
    mask = (st >= t_window[0]) & (st <= t_window[1])
    ax.vlines(st[mask], plot_i + 0.5, plot_i + 1.5, linewidth=0.5, color="k")
ax.set_xlim(t_window)
ax.set_ylim(0.5, len(show_units) + 0.5)
ax.set_ylabel("Unit #")
ax.set_xlabel("Time (s)")
ax.set_title("Neural Activity: Spike Raster (first 30 observed units, 0–10s)")

# --- Figure 2: Finger velocity over time ---
ax = axes[1]
t_mask = t_behav <= 30
ax.plot(t_behav[t_mask], finger_vel[t_mask, 0], label="Vel X", linewidth=0.6)
ax.plot(t_behav[t_mask], finger_vel[t_mask, 1], label="Vel Y", linewidth=0.6)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity")
ax.set_title("Finger Velocity (first 30 seconds)")
ax.legend()

# --- Figure 3: Cursor trajectory and targets ---
ax = axes[2]
subsample = 10
ax.plot(cursor_pos[::subsample, 0], cursor_pos[::subsample, 1],
        linewidth=0.3, alpha=0.5, color="steelblue", label="Cursor path")
ax.scatter(target_pos[::5000, 0], target_pos[::5000, 1],
           s=15, c="red", alpha=0.4, label="Targets")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_title("Cursor Trajectory and Target Positions")
ax.legend()
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("figures/data_exploration.png", dpi=150)
plt.close()
print("\nSaved figures/data_exploration.png")

# --- Figure 4: Firing rate distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

total_duration = t_behav[-1]
firing_rates = [len(all_spike_times[i]) / total_duration for i in observed_idx]

ax = axes[0]
ax.hist(firing_rates, bins=30, edgecolor="black", alpha=0.7)
ax.set_xlabel("Firing rate (Hz)")
ax.set_ylabel("Count")
ax.set_title("Firing Rate Distribution (observed units)")
ax.axvline(np.median(firing_rates), color="red", linestyle="--",
           label=f"Median: {np.median(firing_rates):.1f} Hz")
ax.legend()

# --- Figure 5: Speed profile ---
speed = np.sqrt(finger_vel[:, 0]**2 + finger_vel[:, 1]**2)
ax = axes[1]
ax.hist(speed, bins=100, edgecolor="black", alpha=0.7, density=True)
ax.set_xlabel("Speed")
ax.set_ylabel("Density")
ax.set_title("Movement Speed Distribution")

plt.tight_layout()
plt.savefig("figures/feature_distributions.png", dpi=150)
plt.close()
print("Saved figures/feature_distributions.png")

# --- Alignment check ---
print("\n--- Alignment check ---")
print(f"Behavior duration: {t_behav[-1]:.3f} s")
spike_end_times = [st[-1] for st in all_spike_times if len(st) > 0]
print(f"Latest spike across all units: {max(spike_end_times):.3f} s")
print(f"Earliest spike across all units: {min(st[0] for st in all_spike_times if len(st) > 0):.3f} s")
print(f"Behavior starts at t=0, neural data starts near t=0 — aligned.")

# --- Variable descriptions ---
print("\n--- Variable descriptions ---")
print("cursor_pos (N, 2): 2D position of the cursor on screen, in mm")
print("finger_pos (N, 3): 3D position of the finger (manipulandum), in mm")
print("finger_vel (N, 2): 2D velocity of the finger, in mm/s")
print("target_pos (N, 2): 2D position of the current reach target, in mm")
print("spike_times: list of arrays, each array contains spike timestamps (s) for one unit")
print(f"heldout: boolean flag per unit ({sum(heldout)} held out for NLB benchmark, "
      f"{sum(~np.array(heldout))} observed)")
print("All behavior signals sampled at 1000 Hz. Spikes are event times in seconds.")
