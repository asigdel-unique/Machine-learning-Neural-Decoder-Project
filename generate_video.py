"""
Generate a professional animated visualization of motor cortex recording
with GRU decoder predictions overlaid.
Output: figures/neural_recording_10s.gif
"""
import os
import numpy as np
from pynwb import NWBHDF5IO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

TRAIN_FILE = "data/000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb"
DURATION = 10.0
FPS = 25
PLAYBACK_SPEED = 1.0

# --- Load predictions ---
print("Loading model predictions...")
res = np.load("results/improved_results.npz", allow_pickle=True)
gru_pred = res['gru_pred']
y_test = res['y_test']
gru_r2x = float(res['gru_r2x'])
gru_r2y = float(res['gru_r2y'])
r2_avg = (gru_r2x + gru_r2y) / 2

n_test = len(y_test)
dt_bin = 0.02
test_time = np.arange(n_test) * dt_bin

pred_window = int(DURATION / dt_bin)
actual_vx = y_test[:pred_window, 0]
actual_vy = y_test[:pred_window, 1]
pred_vx = gru_pred[:pred_window, 0]
pred_vy = gru_pred[:pred_window, 1]
t_pred = test_time[:pred_window]

# --- Load raw data ---
print("Loading NWB data...")
with NWBHDF5IO(TRAIN_FILE, "r") as io:
    nwb = io.read()
    behavior = nwb.processing["behavior"]
    finger_vel = behavior["finger_vel"].data[:]
    finger_pos = behavior["finger_pos"].data[:]
    rate = behavior["finger_vel"].rate
    units = nwb.units
    heldout = units["heldout"][:]
    n_units = len(units)
    observed_idx = [i for i in range(n_units) if not heldout[i]]
    spike_times_all = [units["spike_times"][i] for i in range(n_units)]

# The test set starts at 80% of the processed data
total_bins = finger_vel.shape[0] // int(rate * dt_bin)
split_idx = int((total_bins - 14) * 0.8)
test_start_time = (split_idx + 14) * dt_bin
test_start_sample = int(test_start_time * rate)

n_behavior = int(DURATION * rate)
raw_vel_x = finger_vel[test_start_sample:test_start_sample + n_behavior, 0]
raw_vel_y = finger_vel[test_start_sample:test_start_sample + n_behavior, 1]
raw_pos_x = finger_pos[test_start_sample:test_start_sample + n_behavior, 0]
raw_pos_y = finger_pos[test_start_sample:test_start_sample + n_behavior, 1]
raw_speed = np.sqrt(raw_vel_x**2 + raw_vel_y**2)
t_raw = np.arange(n_behavior) / rate

ds = max(1, int(rate / 50))
t_raw_ds = t_raw[::ds]
pos_x_ds = raw_pos_x[::ds]
pos_y_ds = raw_pos_y[::ds]

n_raster_neurons = 20
spike_times = []
for j in range(n_raster_neurons):
    st = spike_times_all[observed_idx[j]]
    mask = (st >= test_start_time) & (st < test_start_time + DURATION)
    spike_times.append(st[mask] - test_start_time)

total_frames = int(DURATION * FPS / PLAYBACK_SPEED)
print(f"Generating {total_frames} frames at {FPS} fps...")

# --- Colors ---
C_ACTUAL = '#1a1a1a'
C_PRED = '#d62728'
C_VX = '#1f77b4'
C_VY = '#ff7f0e'
C_PRED_VX = '#d62728'
C_PRED_VY = '#9467bd'
C_TRAJ = '#2ca02c'
C_SPIKE = '#1f77b4'
C_SCAN = '#d62728'
C_GRID = '#cccccc'
C_BORDER = '#888888'

fig = plt.figure(figsize=(16, 10), facecolor='white', dpi=100)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       height_ratios=[1, 1, 0.85],
                       width_ratios=[1.2, 1, 0.8],
                       hspace=0.38, wspace=0.32)
fig.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.06)

ax_vx = fig.add_subplot(gs[0, 0])
ax_vy = fig.add_subplot(gs[1, 0])
ax_traj = fig.add_subplot(gs[0, 1])
ax_pred_traj = fig.add_subplot(gs[1, 1])
ax_speed = fig.add_subplot(gs[0, 2])
ax_r2 = fig.add_subplot(gs[1, 2])
ax_raster = fig.add_subplot(gs[2, :])

for ax in [ax_vx, ax_vy, ax_traj, ax_pred_traj, ax_speed, ax_r2, ax_raster]:
    ax.set_facecolor('white')
    ax.tick_params(direction='in', width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(C_BORDER)

fig.suptitle('Motor Cortex Neural Decoder-GRU Model',
             fontsize=18, fontweight='bold', y=0.97)
subtitle = fig.text(0.5, 0.935,
    f'Subject Indy - 98 neurons - R² = {r2_avg:.3f}',
    ha='center', fontsize=12, color="#080101", style='italic')
time_text = fig.text(0.95, 0.97, 't = 0.00 s', ha='right',
                     fontsize=14, color=C_PRED, fontweight='bold',
                     fontfamily='Times New Roman')

# --- Velocity X panel ---
vel_range = max(np.abs(actual_vx).max(), np.abs(actual_vy).max(),
                np.abs(pred_vx).max(), np.abs(pred_vy).max()) * 1.1
ax_vx.set_xlim(0, DURATION)
ax_vx.set_ylim(-vel_range, vel_range)
ax_vx.set_ylabel('Velocity X (mm/s)')
ax_vx.set_title('Velocity X: Actual vs Predicted')
ax_vx.axhline(0, color=C_GRID, linewidth=0.5, zorder=0)
ax_vx.grid(True, alpha=0.2, linewidth=0.5)
line_ax, = ax_vx.plot([], [], color=C_VX, linewidth=1.8, label='Actual', zorder=2)
line_px, = ax_vx.plot([], [], color=C_PRED_VX, linewidth=1.2, linestyle='--',
                      label='GRU Predicted', alpha=0.85, zorder=3)
ax_vx.legend(loc='upper right', framealpha=0.9, edgecolor=C_GRID)

# --- Velocity Y panel ---
ax_vy.set_xlim(0, DURATION)
ax_vy.set_ylim(-vel_range, vel_range)
ax_vy.set_xlabel('Time (s)')
ax_vy.set_ylabel('Velocity Y (mm/s)')
ax_vy.set_title('Velocity Y: Actual vs Predicted')
ax_vy.axhline(0, color=C_GRID, linewidth=0.5, zorder=0)
ax_vy.grid(True, alpha=0.2, linewidth=0.5)
line_ay, = ax_vy.plot([], [], color=C_VY, linewidth=1.8, label='Actual', zorder=2)
line_py, = ax_vy.plot([], [], color=C_PRED_VY, linewidth=1.2, linestyle='--',
                      label='GRU Predicted', alpha=0.85, zorder=3)
ax_vy.legend(loc='upper right', framealpha=0.9, edgecolor=C_GRID)

# --- Actual trajectory ---
pad = 5
traj_xmin, traj_xmax = pos_x_ds.min() - pad, pos_x_ds.max() + pad
traj_ymin, traj_ymax = pos_y_ds.min() - pad, pos_y_ds.max() + pad
traj_range = max(traj_xmax - traj_xmin, traj_ymax - traj_ymin)
traj_cx = (traj_xmin + traj_xmax) / 2
traj_cy = (traj_ymin + traj_ymax) / 2
ax_traj.set_xlim(traj_cx - traj_range/2, traj_cx + traj_range/2)
ax_traj.set_ylim(traj_cy - traj_range/2, traj_cy + traj_range/2)
ax_traj.set_xlabel('X (mm)')
ax_traj.set_ylabel('Y (mm)')
ax_traj.set_title('Actual Hand Trajectory')
ax_traj.grid(True, alpha=0.15, linewidth=0.5)
line_traj_trail, = ax_traj.plot([], [], color=C_TRAJ, linewidth=0.6, alpha=0.3)
line_traj_recent, = ax_traj.plot([], [], color=C_TRAJ, linewidth=2.5, alpha=0.8)
dot_traj, = ax_traj.plot([], [], 'o', color=C_TRAJ, markersize=7, zorder=5)

# --- Predicted trajectory (from integrating predicted velocity, centered at origin) ---
actual_pos_x = np.cumsum(actual_vx) * dt_bin
actual_pos_y = np.cumsum(actual_vy) * dt_bin
pred_pos_x = np.cumsum(pred_vx) * dt_bin
pred_pos_y = np.cumsum(pred_vy) * dt_bin
actual_pos_x -= actual_pos_x[0]
actual_pos_y -= actual_pos_y[0]
pred_pos_x -= pred_pos_x[0]
pred_pos_y -= pred_pos_y[0]
all_px = np.concatenate([actual_pos_x, pred_pos_x])
all_py = np.concatenate([actual_pos_y, pred_pos_y])

dec_pad = 5
dec_xmin, dec_xmax = all_px.min() - dec_pad, all_px.max() + dec_pad
dec_ymin, dec_ymax = all_py.min() - dec_pad, all_py.max() + dec_pad
dec_range = max(dec_xmax - dec_xmin, dec_ymax - dec_ymin)
dec_cx = (dec_xmin + dec_xmax) / 2
dec_cy = (dec_ymin + dec_ymax) / 2
ax_pred_traj.set_xlim(dec_cx - dec_range/2, dec_cx + dec_range/2)
ax_pred_traj.set_ylim(dec_cy - dec_range/2, dec_cy + dec_range/2)
ax_pred_traj.set_xlabel('X (mm)')
ax_pred_traj.set_ylabel('Y (mm)')
ax_pred_traj.set_title('Decoded Trajectory (Integrated)')
ax_pred_traj.grid(True, alpha=0.15, linewidth=0.5)
line_actual_int, = ax_pred_traj.plot([], [], color=C_VX, linewidth=1.8, label='Actual')
line_pred_int, = ax_pred_traj.plot([], [], color=C_PRED, linewidth=1.5, linestyle='--',
                                   label='Predicted')
dot_actual_int, = ax_pred_traj.plot([], [], 'o', color=C_VX, markersize=6, zorder=5)
dot_pred_int, = ax_pred_traj.plot([], [], 's', color=C_PRED, markersize=5, zorder=5)
ax_pred_traj.legend(loc='upper right', framealpha=0.9, edgecolor=C_GRID, fontsize=8)

# --- Speed panel ---
actual_speed = np.sqrt(actual_vx**2 + actual_vy**2)
pred_speed = np.sqrt(pred_vx**2 + pred_vy**2)
ax_speed.set_xlim(0, DURATION)
ax_speed.set_ylim(0, max(actual_speed.max(), pred_speed.max()) * 1.1)
ax_speed.set_ylabel('Speed (mm/s)')
ax_speed.set_title('Movement Speed')
ax_speed.grid(True, alpha=0.2, linewidth=0.5)
line_aspd, = ax_speed.plot([], [], color=C_VX, linewidth=1.8, label='Actual')
line_pspd, = ax_speed.plot([], [], color=C_PRED, linewidth=1.2, linestyle='--', label='Predicted')
ax_speed.legend(loc='upper right', framealpha=0.9, edgecolor=C_GRID, fontsize=8)

# --- Live R² panel ---
ax_r2.set_xlim(0, DURATION)
ax_r2.set_ylim(-0.1, 1.0)
ax_r2.set_xlabel('Time (s)')
ax_r2.set_ylabel('R² Score')
ax_r2.set_title('Running R² (cumulative)')
ax_r2.grid(True, alpha=0.2, linewidth=0.5)
ax_r2.axhline(0.5, color=C_GRID, linewidth=0.8, linestyle=':')
ax_r2.axhline(gru_r2x, color=C_VX, linewidth=0.5, linestyle=':', alpha=0.4)
ax_r2.axhline(gru_r2y, color=C_VY, linewidth=0.5, linestyle=':', alpha=0.4)
line_r2x, = ax_r2.plot([], [], color=C_VX, linewidth=1.8, label=f'R² X (final: {gru_r2x:.3f})')
line_r2y, = ax_r2.plot([], [], color=C_VY, linewidth=1.8, label=f'R² Y (final: {gru_r2y:.3f})')
ax_r2.legend(loc='lower right', framealpha=0.9, edgecolor=C_GRID, fontsize=8)

# --- Raster panel ---
ax_raster.set_xlim(0, DURATION)
ax_raster.set_ylim(-0.5, n_raster_neurons - 0.5)
ax_raster.set_xlabel('Time (s)')
ax_raster.set_ylabel('Neuron #')
ax_raster.set_title('Spike Raster — 20 Neurons')
ax_raster.set_yticks(range(0, n_raster_neurons, 5))
ax_raster.invert_yaxis()
ax_raster.grid(True, axis='x', alpha=0.15, linewidth=0.5)

raster_segments = []
for n_idx in range(n_raster_neurons):
    for st in spike_times[n_idx]:
        raster_segments.append([(st, n_idx - 0.35), (st, n_idx + 0.35)])

raster_lc = LineCollection([], colors=C_SPIKE, linewidths=0.7, alpha=0.7)
ax_raster.add_collection(raster_lc)
scan_line, = ax_raster.plot([], [], color=C_SCAN, linewidth=1.5, alpha=0.6)

speed_fill = [None]

# Precompute running R² at every prediction index (stable, no jitter)
print("Precomputing running R²...")
r2x_precomputed = np.full(pred_window, np.nan)
r2y_precomputed = np.full(pred_window, np.nan)
for i in range(20, pred_window):
    ss_res_x = np.sum((actual_vx[:i+1] - pred_vx[:i+1]) ** 2)
    ss_tot_x = np.sum((actual_vx[:i+1] - np.mean(actual_vx[:i+1])) ** 2)
    ss_res_y = np.sum((actual_vy[:i+1] - pred_vy[:i+1]) ** 2)
    ss_tot_y = np.sum((actual_vy[:i+1] - np.mean(actual_vy[:i+1])) ** 2)
    r2x_precomputed[i] = max(0, 1 - ss_res_x / ss_tot_x) if ss_tot_x > 0 else 0
    r2y_precomputed[i] = max(0, 1 - ss_res_y / ss_tot_y) if ss_tot_y > 0 else 0

def init():
    for l in [line_ax, line_px, line_ay, line_py, line_traj_trail,
              line_traj_recent, dot_traj, line_actual_int, line_pred_int,
              dot_actual_int, dot_pred_int, line_aspd, line_pspd,
              line_r2x, line_r2y, scan_line]:
        l.set_data([], [])
    raster_lc.set_segments([])
    return []


def update(frame):
    t_now = frame / total_frames * DURATION
    time_text.set_text(f't = {t_now:.2f} s')

    # Prediction index (20ms bins)
    pidx = min(int(t_now / dt_bin), pred_window - 1)

    # Raw trajectory index (50 Hz downsampled)
    ridx = min(int(t_now * 50), len(t_raw_ds) - 1)

    # Velocity X
    line_ax.set_data(t_pred[:pidx+1], actual_vx[:pidx+1])
    line_px.set_data(t_pred[:pidx+1], pred_vx[:pidx+1])

    # Velocity Y
    line_ay.set_data(t_pred[:pidx+1], actual_vy[:pidx+1])
    line_py.set_data(t_pred[:pidx+1], pred_vy[:pidx+1])

    # Actual trajectory
    line_traj_trail.set_data(pos_x_ds[:ridx+1], pos_y_ds[:ridx+1])
    trail_start = max(0, ridx - 25)
    line_traj_recent.set_data(pos_x_ds[trail_start:ridx+1], pos_y_ds[trail_start:ridx+1])
    if ridx > 0:
        dot_traj.set_data([pos_x_ds[ridx]], [pos_y_ds[ridx]])

    # Integrated trajectory
    line_actual_int.set_data(actual_pos_x[:pidx+1], actual_pos_y[:pidx+1])
    line_pred_int.set_data(pred_pos_x[:pidx+1], pred_pos_y[:pidx+1])
    if pidx > 0:
        dot_actual_int.set_data([actual_pos_x[pidx]], [actual_pos_y[pidx]])
        dot_pred_int.set_data([pred_pos_x[pidx]], [pred_pos_y[pidx]])

    # Speed
    actual_spd_slice = np.sqrt(actual_vx[:pidx+1]**2 + actual_vy[:pidx+1]**2)
    pred_spd_slice = np.sqrt(pred_vx[:pidx+1]**2 + pred_vy[:pidx+1]**2)
    line_aspd.set_data(t_pred[:pidx+1], actual_spd_slice)
    line_pspd.set_data(t_pred[:pidx+1], pred_spd_slice)

    # Running R² (precomputed, stable)
    if pidx > 20:
        valid = ~np.isnan(r2x_precomputed[20:pidx+1])
        line_r2x.set_data(t_pred[20:pidx+1][valid], r2x_precomputed[20:pidx+1][valid])
        line_r2y.set_data(t_pred[20:pidx+1][valid], r2y_precomputed[20:pidx+1][valid])

    # Raster
    visible_segs = [seg for seg in raster_segments if seg[0][0] <= t_now]
    raster_lc.set_segments(visible_segs)
    scan_line.set_data([t_now, t_now], [-0.5, n_raster_neurons - 0.5])

    if frame % FPS == 0:
        pct = frame / total_frames * 100
        print(f"  Frame {frame}/{total_frames} ({pct:.0f}%)")

    return []

print("Rendering animation (this takes a few minutes)...")
anim = FuncAnimation(fig, update, init_func=init, frames=total_frames,
                     interval=1000 // FPS, blit=False)

output_path = "figures/neural_recording_10s.gif"
writer = PillowWriter(fps=FPS)
anim.save(output_path, writer=writer)

file_size = round(os.path.getsize(output_path) / (1024 * 1024), 1)
print(f"\nSaved: {output_path}")
print(f"File size: {file_size} MB")
print(f"GRU R²: X={gru_r2x:.3f}, Y={gru_r2y:.3f}, Avg={r2_avg:.3f}")
plt.close()
