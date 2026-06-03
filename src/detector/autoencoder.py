from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .constants import AE_LATENT_DIM, AE_PERCENTILE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Autoencoder(nn.Module):
    """Feedforward autoencoder: n_features → n_features*2 → latent → n_features*2 → n_features.

    Architecture from test_notebooks/anomaly_detection.ipynb — unchanged.
    Trained on normal rows only. Anomaly score = per-row MSE reconstruction error.
    """

    def __init__(self, n_features: int, latent_dim: int = AE_LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, n_features * 2),
            nn.ReLU(),
            nn.BatchNorm1d(n_features * 2),
            nn.Linear(n_features * 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_features * 2),
            nn.ReLU(),
            nn.BatchNorm1d(n_features * 2),
            nn.Linear(n_features * 2, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    X_normal: np.ndarray,
    n_features: int,
    latent_dim: int = AE_LATENT_DIM,
    n_epochs: int = 150,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 15,
    device: torch.device = DEVICE,
    verbose: bool = True,
) -> tuple:
    """Train autoencoder on normal rows. Returns (model, train_losses, val_losses)."""
    model     = Autoencoder(n_features, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    n_val = int(len(X_normal) * 0.1)
    X_t   = torch.FloatTensor(X_normal[n_val:]).to(device)
    X_v   = torch.FloatTensor(X_normal[:n_val]).to(device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)

    train_losses, val_losses         = [], []
    best_val, no_improve, best_state = float('inf'), 0, None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        train_loss = epoch_loss / len(X_t)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), X_v).item()
        model.train()

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            if verbose:
                print(f'  Early stop at epoch {epoch+1} — val_loss={best_val:.6f}')
            break
        if verbose and (epoch + 1) % 25 == 0:
            print(f'  Epoch {epoch+1:3d} | train={train_loss:.6f} | val={val_loss:.6f}')

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


def score_autoencoder(
    model: Autoencoder,
    X_all: np.ndarray,
    X_normal: np.ndarray,
    percentile: int = AE_PERCENTILE,
    device: torch.device = DEVICE,
) -> tuple:
    """Compute reconstruction errors and anomaly scores for all rows.

    Threshold is derived from p{percentile} of normal-row errors so no labels needed.

    Returns:
        ae_errors:    per-row MSE, shape (N,)
        ae_per_sensor: per-sensor squared errors, shape (N, n_features)
        ae_scores:    normalised [0, 1], higher = more anomalous
        ae_flags:     bool, True where error > threshold
    """
    model.eval()
    with torch.no_grad():
        X_recon        = model(torch.FloatTensor(X_all).to(device)).cpu().numpy()
        X_normal_recon = model(torch.FloatTensor(X_normal).to(device)).cpu().numpy()

    ae_per_sensor  = (X_all - X_recon) ** 2
    ae_errors      = ae_per_sensor.mean(axis=1)
    normal_errors  = ((X_normal - X_normal_recon) ** 2).mean(axis=1)
    ae_threshold   = float(np.percentile(normal_errors, percentile))

    ae_scores = np.clip(ae_errors / (ae_threshold * 2), 0, 1)
    ae_flags  = ae_errors > ae_threshold
    return ae_errors, ae_per_sensor, ae_scores, ae_flags


def save_autoencoder(model: Autoencoder, path: Path) -> None:
    torch.save(model.state_dict(), path)


def load_autoencoder(
    path: Path,
    n_features: int = 5,
    latent_dim: int = AE_LATENT_DIM,
    device: torch.device = DEVICE,
) -> Autoencoder:
    model = Autoencoder(n_features, latent_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
