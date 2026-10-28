"""
Interpretable Attention-GRU decoder with confidence estimation.
Predicts 2D velocity + confidence score, and exposes attention weights
over timesteps and neurons for interpretability.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


class NeuronAttention(nn.Module):
    def __init__(self, n_neurons, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(n_neurons, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_neurons),
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=-1)
        return x * weights, weights


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, gru_outputs):
        scores = self.attn(gru_outputs).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), gru_outputs).squeeze(1)
        return context, weights


class AttentionGRUDecoder(nn.Module):
    def __init__(self, n_neurons, n_timesteps, hidden_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_timesteps = n_timesteps

        self.neuron_attn = NeuronAttention(n_neurons, hidden_dim=64)
        self.input_norm = nn.LayerNorm(n_neurons)

        self.gru = nn.GRU(
            input_size=n_neurons,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.gru_norm = nn.LayerNorm(hidden_dim)
        self.temporal_attn = TemporalAttention(hidden_dim)

        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, return_attention=False):
        attended_x, neuron_weights = self.neuron_attn(x)
        normed_x = self.input_norm(attended_x)

        gru_out, _ = self.gru(normed_x)
        gru_out = self.gru_norm(gru_out)

        context, temporal_weights = self.temporal_attn(gru_out)

        last_hidden = gru_out[:, -1, :]
        combined = torch.cat([context, last_hidden], dim=-1)

        velocity = self.velocity_head(combined)
        confidence = self.confidence_head(combined)

        if return_attention:
            return velocity, confidence, temporal_weights, neuron_weights
        return velocity, confidence


def confidence_aware_loss(pred_vel, true_vel, confidence, lambda_conf=0.5):
    error_sq = (pred_vel - true_vel).pow(2).sum(dim=-1, keepdim=True)
    vel_loss = error_sq.mean()
    conf_loss = (confidence * error_sq + (1 - confidence) * 50.0).mean()
    return vel_loss + lambda_conf * conf_loss


def train_phase1_velocity(model, train_loader, val_loader, n_epochs=100, patience=25):
    """Phase 1: Train for pure velocity accuracy (MSE only)."""
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses, val_losses = [], []

    print("\n--- Phase 1: Pure velocity training (MSE) ---")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_samples = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred_vel, _ = model(X_batch)
            loss = criterion(pred_vel, y_batch)
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
                pred_vel, _ = model(X_batch)
                val_loss += criterion(pred_vel, y_batch).item() * len(X_batch)
                n_val += len(X_batch)
        val_losses.append(val_loss / n_val)
        scheduler.step()

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "attention_gru_best.pt"))
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}: train={train_losses[-1]:.2f} "
                  f"val={val_losses[-1]:.2f} lr={lr:.6f}"
                  f"{' *best*' if patience_counter == 0 else ''}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best: epoch {best_epoch+1})")
            break

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "attention_gru_best.pt"), weights_only=True))
    return model, train_losses, val_losses, best_epoch


def train_phase2_confidence(model, train_loader, val_loader, n_epochs=30):
    """Phase 2: Fine-tune confidence head only (freeze everything else)."""
    device = torch.device("cpu")
    model = model.to(device)

    for name, param in model.named_parameters():
        if "confidence" not in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.confidence_head.parameters(), lr=5e-4)

    print("\n--- Phase 2: Confidence calibration (backbone frozen) ---")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_samples = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred_vel, confidence = model(X_batch)
            error_sq = (pred_vel.detach() - y_batch).pow(2).sum(dim=-1, keepdim=True)
            conf_loss = (confidence * error_sq + (1 - confidence) * 50.0).mean()
            optimizer.zero_grad()
            conf_loss.backward()
            optimizer.step()
            epoch_loss += conf_loss.item() * len(X_batch)
            n_samples += len(X_batch)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: conf_loss={epoch_loss/n_samples:.2f}")

    for param in model.parameters():
        param.requires_grad = True

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "attention_gru_best.pt"))
    return model


def evaluate_attention_model(model, test_loader):
    model.eval()
    all_vel, all_conf, all_true = [], [], []
    all_temporal_attn, all_neuron_attn = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            vel, conf, t_attn, n_attn = model(X_batch, return_attention=True)
            all_vel.append(vel.numpy())
            all_conf.append(conf.numpy())
            all_true.append(y_batch.numpy())
            all_temporal_attn.append(t_attn.numpy())
            all_neuron_attn.append(n_attn[:, -1, :].numpy())

    y_pred = np.concatenate(all_vel)
    y_true = np.concatenate(all_true)
    confidence = np.concatenate(all_conf)
    temporal_attn = np.concatenate(all_temporal_attn)
    neuron_attn = np.concatenate(all_neuron_attn)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_x = 1 - np.sum((y_true[:, 0] - y_pred[:, 0]) ** 2) / np.sum(
        (y_true[:, 0] - y_true[:, 0].mean()) ** 2
    )
    r2_y = 1 - np.sum((y_true[:, 1] - y_pred[:, 1]) ** 2) / np.sum(
        (y_true[:, 1] - y_true[:, 1].mean()) ** 2
    )

    return y_pred, confidence, temporal_attn, neuron_attn, mae, rmse, r2_x, r2_y


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    train_data = np.load(os.path.join(BASE_DIR, "processed_data", "improved_train.npz"))
    test_data = np.load(os.path.join(BASE_DIR, "processed_data", "improved_test.npz"))

    X_seq_train, y_train = train_data["X_seq"], train_data["y"]
    X_seq_test, y_test = test_data["X_seq"], test_data["y"]

    n_timesteps = X_seq_train.shape[1]
    n_neurons = X_seq_train.shape[2]

    val_split = int(len(X_seq_train) * 0.85)
    X_tr = X_seq_train[:val_split]
    y_tr = y_train[:val_split]
    X_val = X_seq_train[val_split:]
    y_val = y_train[val_split:]

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
    import gc; gc.collect()

    print("=" * 70)
    print("ATTENTION-GRU WITH CONFIDENCE ESTIMATION")
    print("=" * 70)

    model = AttentionGRUDecoder(
        n_neurons=n_neurons,
        n_timesteps=n_timesteps,
        hidden_dim=256,
        n_layers=2,
        dropout=0.3,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    ckpt_path = os.path.join(OUTPUT_DIR, "attention_gru_best.pt")
    if os.path.exists(ckpt_path):
        print("Loading existing checkpoint (skip training)...")
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        train_losses, val_losses, best_epoch = [], [], 0
    else:
        model, train_losses, val_losses, best_epoch = train_phase1_velocity(
            model, train_loader, val_loader, n_epochs=100, patience=25
        )
        model = train_phase2_confidence(model, train_loader, val_loader, n_epochs=30)

    y_pred, confidence, temporal_attn, neuron_attn, mae, rmse, r2_x, r2_y = (
        evaluate_attention_model(model, test_loader)
    )
    r2_avg = (r2_x + r2_y) / 2

    print("\n" + "=" * 70)
    print("ATTENTION-GRU RESULTS")
    print("=" * 70)
    print(f"  MAE:    {mae:.2f} mm/s")
    print(f"  RMSE:   {rmse:.2f} mm/s")
    print(f"  R2 X:   {r2_x:.4f}")
    print(f"  R2 Y:   {r2_y:.4f}")
    print(f"  R2 Avg: {r2_avg:.4f}")

    prev_best = 0.722
    improvement = r2_avg - prev_best
    print(f"\n  Previous best (GRU): {prev_best:.3f}")
    print(f"  Attention-GRU:       {r2_avg:.3f} ({improvement:+.3f})")

    # --- Confidence analysis ---
    confidence_flat = confidence.squeeze()
    errors = np.sqrt(((y_pred - y_test) ** 2).sum(axis=1))

    p75 = np.percentile(confidence_flat, 75)
    p25 = np.percentile(confidence_flat, 25)
    high_conf = confidence_flat >= p75
    low_conf = confidence_flat <= p25

    print(f"\n  Confidence range: [{confidence_flat.min():.3f}, {confidence_flat.max():.3f}]")
    print(f"  High conf count: {high_conf.sum()}, Low conf count: {low_conf.sum()}")

    def safe_r2(y_true, y_pred_sub):
        ss_res = np.sum((y_true - y_pred_sub) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if high_conf.sum() > 10:
        r2_high = (safe_r2(y_test[high_conf, 0], y_pred[high_conf, 0]) +
                   safe_r2(y_test[high_conf, 1], y_pred[high_conf, 1])) / 2
        print(f"\n  Confidence-based filtering:")
        print(f"    High confidence (top 25%): R2 = {r2_high:.4f}")
        print(f"    Mean error (high conf): {errors[high_conf].mean():.2f} mm/s")
    if low_conf.sum() > 10:
        r2_low = (safe_r2(y_test[low_conf, 0], y_pred[low_conf, 0]) +
                  safe_r2(y_test[low_conf, 1], y_pred[low_conf, 1])) / 2
        print(f"    Low confidence (bottom 25%): R2 = {r2_low:.4f}")
        print(f"    Mean error (low conf):  {errors[low_conf].mean():.2f} mm/s")

    # --- Top neurons ---
    mean_neuron_attn = neuron_attn.mean(axis=0)
    top_neurons = np.argsort(mean_neuron_attn)[::-1][:10]
    print("\n  Top 10 most attended neurons:")
    for rank, nid in enumerate(top_neurons):
        print(f"    {rank+1}. Neuron {nid:2d} (attention weight: {mean_neuron_attn[nid]:.4f})")

    # --- Ensemble: average GRU + Attention-GRU ---
    prev_results_ens = np.load(os.path.join(OUTPUT_DIR, "improved_results.npz"))
    gru_pred_prev = prev_results_ens["gru_pred"]
    ensemble_pred = (y_pred + gru_pred_prev) / 2

    ens_mae = mean_absolute_error(y_test, ensemble_pred)
    ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ens_r2_x = 1 - np.sum((y_test[:, 0] - ensemble_pred[:, 0]) ** 2) / np.sum(
        (y_test[:, 0] - y_test[:, 0].mean()) ** 2)
    ens_r2_y = 1 - np.sum((y_test[:, 1] - ensemble_pred[:, 1]) ** 2) / np.sum(
        (y_test[:, 1] - y_test[:, 1].mean()) ** 2)
    ens_r2_avg = (ens_r2_x + ens_r2_y) / 2

    print(f"\n  Ensemble (GRU + Attn-GRU average):")
    print(f"    MAE:    {ens_mae:.2f} mm/s")
    print(f"    RMSE:   {ens_rmse:.2f} mm/s")
    print(f"    R2 X:   {ens_r2_x:.4f}")
    print(f"    R2 Y:   {ens_r2_y:.4f}")
    print(f"    R2 Avg: {ens_r2_avg:.4f} (GRU was 0.722)")

    # ============================================================
    # Visualization (6-panel figure)
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    t_test = np.arange(len(y_test)) * 0.02
    mask = t_test <= 10

    # 1. Velocity X predictions
    ax = axes[0, 0]
    ax.plot(t_test[mask], y_test[mask, 0], label="Actual", linewidth=0.8, color="black")
    ax.plot(t_test[mask], y_pred[mask, 0], label="Attn-GRU", linewidth=0.8, color="#e74c3c")
    ax.fill_between(
        t_test[mask],
        y_pred[mask, 0] - (1 - confidence_flat[mask]) * 40,
        y_pred[mask, 0] + (1 - confidence_flat[mask]) * 40,
        alpha=0.15,
        color="#e74c3c",
        label="Uncertainty",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity X (mm/s)")
    ax.set_title("Velocity X: Prediction + Confidence")
    ax.legend(fontsize=8)

    # 2. Velocity Y predictions
    ax = axes[0, 1]
    ax.plot(t_test[mask], y_test[mask, 1], label="Actual", linewidth=0.8, color="black")
    ax.plot(t_test[mask], y_pred[mask, 1], label="Attn-GRU", linewidth=0.8, color="#3498db")
    ax.fill_between(
        t_test[mask],
        y_pred[mask, 1] - (1 - confidence_flat[mask]) * 40,
        y_pred[mask, 1] + (1 - confidence_flat[mask]) * 40,
        alpha=0.15,
        color="#3498db",
        label="Uncertainty",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity Y (mm/s)")
    ax.set_title("Velocity Y: Prediction + Confidence")
    ax.legend(fontsize=8)

    # 3. Confidence vs Error scatter
    ax = axes[0, 2]
    scatter = ax.scatter(
        confidence_flat[::5],
        errors[::5],
        c=errors[::5],
        cmap="RdYlGn_r",
        alpha=0.4,
        s=8,
        edgecolors="none",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Prediction Error (mm/s)")
    ax.set_title("Confidence vs Error")
    plt.colorbar(scatter, ax=ax, label="Error")

    # 4. Temporal attention heatmap
    ax = axes[1, 0]
    mean_temporal = temporal_attn[mask].mean(axis=0)
    bins_ms = np.arange(n_timesteps) * 20
    ax.bar(bins_ms, mean_temporal, width=16, color="#2ecc71", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Time lag (ms)")
    ax.set_ylabel("Attention Weight")
    ax.set_title("Temporal Attention (which timesteps matter)")
    ax.set_xticks(bins_ms)
    ax.set_xticklabels([f"-{(n_timesteps-1-i)*20}" for i in range(n_timesteps)], fontsize=7)

    # 5. Neuron importance
    ax = axes[1, 1]
    sorted_idx = np.argsort(mean_neuron_attn)[::-1][:30]
    ax.barh(
        range(30),
        mean_neuron_attn[sorted_idx],
        color="#9b59b6",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_yticks(range(30))
    ax.set_yticklabels([f"N{i}" for i in sorted_idx], fontsize=7)
    ax.set_xlabel("Mean Attention Weight")
    ax.set_title("Top 30 Neurons by Importance")
    ax.invert_yaxis()

    # 6. Comparison bar chart
    ax = axes[1, 2]
    prev_results = np.load(os.path.join(OUTPUT_DIR, "improved_results.npz"))
    prev_gru_r2 = (float(prev_results["gru_r2x"]) + float(prev_results["gru_r2y"])) / 2
    prev_lstm_r2 = (float(prev_results["lstm_r2x"]) + float(prev_results["lstm_r2y"])) / 2
    prev_ff_r2 = (float(prev_results["ff_r2x"]) + float(prev_results["ff_r2y"])) / 2

    model_names = ["Impr FF", "LSTM", "GRU", "Attn-GRU", "Ensemble"]
    r2_vals = [prev_ff_r2, prev_lstm_r2, prev_gru_r2, r2_avg, ens_r2_avg]
    colors = ["#3498db", "#2ecc71", "#9b59b6", "#e74c3c", "#f39c12"]
    bars = ax.bar(model_names, r2_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Average R2")
    ax.set_title("Model Comparison")
    ax.set_ylim(0.6, max(r2_vals) * 1.08)
    for bar, val in zip(bars, r2_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.suptitle(
        "Interpretable Attention-GRU with Confidence Estimation",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "attention_gru_results.png"), dpi=150)
    plt.close()
    print("\nSaved figures/attention_gru_results.png")

    # Save all results
    np.savez(
        os.path.join(OUTPUT_DIR, "attention_gru_results.npz"),
        y_pred=y_pred,
        y_test=y_test,
        confidence=confidence,
        temporal_attn=temporal_attn,
        neuron_attn=neuron_attn,
        r2_x=r2_x,
        r2_y=r2_y,
        mae=mae,
        rmse=rmse,
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        top_neurons=top_neurons,
        ensemble_pred=ensemble_pred,
        ens_r2_x=ens_r2_x, ens_r2_y=ens_r2_y, ens_mae=ens_mae,
    )
    print("Saved results/attention_gru_results.npz")
