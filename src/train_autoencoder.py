import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from config import (MODELS_DIR, METRICS_DIR, FIGURES_DIR, RANDOM_SEED, PATCH_FRAMES,
                    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, LATENT_DIM)
from data_utils import load_processed
from model import CNNAutoencoder, normalize, patch_starts


class PatchDataset(Dataset):
    def __init__(self, specs, indices, mean, std):
        self.specs, self.mean, self.std = specs, mean, std
        self.items = [(i, s) for i in indices for s in patch_starts(specs.shape[2])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, s = self.items[k]
        patch = normalize(self.specs[i][:, s:s + PATCH_FRAMES], self.mean, self.std)
        return torch.from_numpy(patch[None])


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for x in loader:
            loss = loss_fn(model(x), x)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / n


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    for d in (MODELS_DIR, METRICS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df, specs, mean, std = load_processed()
    specs = np.array(specs)  # load into RAM (~450 MB) for speed
    train_idx = df.index[df.split == "train"].values
    val_idx = df.index[df.split == "val"].values

    train_loader = DataLoader(PatchDataset(specs, train_idx, mean, std),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(PatchDataset(specs, val_idx, mean, std),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train patches: {len(train_loader.dataset)} | Val patches: {len(val_loader.dataset)}")

    model = CNNAutoencoder()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()

    best_val, bad_epochs, history = float("inf"), 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        msg = f"Epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f} | {time.time() - t0:.0f}s"
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            torch.save({"state_dict": model.state_dict(), "latent_dim": LATENT_DIM},
                       MODELS_DIR / "cnn_autoencoder.pt")
            msg += "  <- best, saved"
        else:
            bad_epochs += 1
        print(msg)
        if bad_epochs >= PATIENCE:
            print(f"Early stopping: no val improvement for {PATIENCE} epochs.")
            break

    hist = pd.DataFrame(history)
    hist.to_csv(METRICS_DIR / "cnn_ae_training_history.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(hist.epoch, hist.train_loss, label="Train loss")
    plt.plot(hist.epoch, hist.val_loss, label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("CNN autoencoder training")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cnn_ae_loss_curve.png", dpi=150)
    print(f"Best val loss: {best_val:.4f}. Model and loss curve saved.")


if __name__ == "__main__":
    main()