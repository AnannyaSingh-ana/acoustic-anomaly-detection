import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from config import (MODELS_DIR, METRICS_DIR, FIGURES_DIR, RANDOM_SEED, PATCH_FRAMES,
                    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
                    CLF_MAX_EPOCHS, MACHINE_IDS)
from data_utils import load_processed
from model import normalize, patch_starts
from id_classifier import IDClassifier


class LabeledPatchDataset(Dataset):
    def __init__(self, specs, indices, labels, mean, std):
        self.data = np.stack([normalize(specs[i], mean, std) for i in indices])
        self.labels = np.asarray(labels)
        self.items = [(j, s) for j in range(len(indices))
                      for s in patch_starts(self.data.shape[2])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        j, s = self.items[k]
        x = torch.from_numpy(self.data[j, None, :, s:s + PATCH_FRAMES].copy())
        return x, int(self.labels[j])


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(x)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(x)
    return total_loss / n, correct / n


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    for d in (MODELS_DIR, METRICS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df, specs, mean, std = load_processed()
    id_to_idx = {m: i for i, m in enumerate(MACHINE_IDS)}
    loaders = {}
    for split, shuffle in [("train", True), ("val", False)]:
        idx = df.index[df.split == split].values
        labels = df.loc[idx, "machine_id"].map(id_to_idx).values
        loaders[split] = DataLoader(LabeledPatchDataset(specs, idx, labels, mean, std),
                                    batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)
    print(f"Train patches: {len(loaders['train'].dataset)} | "
          f"Val patches: {len(loaders['val'].dataset)}")

    model = IDClassifier()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val, bad_epochs, history = float("inf"), 0, []
    for epoch in range(1, CLF_MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], loss_fn, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], loss_fn)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "train_acc": tr_acc, "val_acc": va_acc})
        msg = (f"Epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
               f"val loss {va_loss:.4f} acc {va_acc:.3f} | {time.time() - t0:.0f}s")
        if va_loss < best_val:
            best_val, bad_epochs = va_loss, 0
            torch.save({"state_dict": model.state_dict()}, MODELS_DIR / "id_classifier.pt")
            msg += "  <- best, saved"
        else:
            bad_epochs += 1
        print(msg)
        if bad_epochs >= PATIENCE:
            print(f"Early stopping: no val improvement for {PATIENCE} epochs.")
            break

    hist = pd.DataFrame(history)
    hist.to_csv(METRICS_DIR / "id_clf_training_history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(hist.epoch, hist.train_loss, label="Train"); ax[0].plot(hist.epoch, hist.val_loss, label="Validation")
    ax[0].set_title("Loss (cross-entropy)"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(hist.epoch, hist.train_acc, label="Train"); ax[1].plot(hist.epoch, hist.val_acc, label="Validation")
    ax[1].set_title("Machine-ID accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "id_clf_training_curves.png", dpi=150)
    print(f"Best val loss: {best_val:.4f}. Model and curves saved.")


if __name__ == "__main__":
    main()