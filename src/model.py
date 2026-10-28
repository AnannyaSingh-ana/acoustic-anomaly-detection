import numpy as np
import torch
import torch.nn as nn
from config import N_MELS, PATCH_FRAMES, PATCH_HOP, LATENT_DIM


class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        def down(i, o):  # halves height and width
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU())

        def up(i, o):    # doubles height and width
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, stride=2, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU())

        self.h, self.w = N_MELS // 8, PATCH_FRAMES // 8          # 16 x 8
        self.encoder = nn.Sequential(down(1, 16), down(16, 32), down(32, 64))
        self.to_latent = nn.Linear(64 * self.h * self.w, latent_dim)
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, 64 * self.h * self.w), nn.ReLU())
        self.decoder = nn.Sequential(up(64, 32), up(32, 16),
                                     nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1))

    def forward(self, x):                       # x: (batch, 1, 128, 64)
        z = self.to_latent(self.encoder(x).flatten(1))
        h = self.from_latent(z).view(-1, 64, self.h, self.w)
        return self.decoder(h)


def patch_starts(n_frames):
    return list(range(0, n_frames - PATCH_FRAMES + 1, PATCH_HOP))


def normalize(spec, mean, std):
    """Per-Mel-band standardization using TRAIN statistics."""
    return (np.asarray(spec, dtype=np.float32) - mean[:, None]) / std[:, None]


def clip_to_patches(spec, mean, std):
    x = normalize(spec, mean, std)
    return np.stack([x[:, s:s + PATCH_FRAMES] for s in patch_starts(x.shape[1])])[:, None]


@torch.no_grad()
def clip_scores(model, specs, indices, mean, std, batch_clips=32):
    """Anomaly score per clip = mean reconstruction error over its patches."""
    model.eval()
    scores = []
    for b in range(0, len(indices), batch_clips):
        chunk = [clip_to_patches(specs[i], mean, std) for i in indices[b:b + batch_clips]]
        n_patches = chunk[0].shape[0]
        x = torch.from_numpy(np.concatenate(chunk))
        err = ((model(x) - x) ** 2).mean(dim=(1, 2, 3))
        scores.append(err.view(len(chunk), n_patches).mean(dim=1).numpy())
    return np.concatenate(scores)