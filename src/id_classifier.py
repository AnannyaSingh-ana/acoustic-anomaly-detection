import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MACHINE_IDS, DROPOUT
from model import clip_to_patches


class IDClassifier(nn.Module):
    """Small CNN that predicts which machine ID a spectrogram patch came from."""

    def __init__(self, n_classes=len(MACHINE_IDS), dropout=DROPOUT):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 16), block(16, 32),
                                      block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):                    # x: (batch, 1, 128, 64)
        h = self.features(x).mean(dim=(2, 3))  # global average pooling -> (batch, 128)
        return self.head(h)                  # logits for each machine ID


@torch.no_grad()
def clip_id_scores(model, specs, indices, id_labels, mean, std, batch_clips=32):
    """Anomaly score per clip = mean over patches of -log p(correct machine ID)."""
    model.eval()
    scores = []
    for b in range(0, len(indices), batch_clips):
        idx = indices[b:b + batch_clips]
        chunk = [clip_to_patches(specs[i], mean, std) for i in idx]
        n_patches = chunk[0].shape[0]
        x = torch.from_numpy(np.concatenate(chunk))
        logp = F.log_softmax(model(x), dim=1).view(len(idx), n_patches, -1)
        y = torch.as_tensor(np.array(id_labels[b:b + batch_clips]), dtype=torch.long)
        logp_true = logp.gather(2, y.view(-1, 1, 1).expand(-1, n_patches, 1)).squeeze(2)
        scores.append((-logp_true).mean(dim=1).numpy())
    return np.concatenate(scores)