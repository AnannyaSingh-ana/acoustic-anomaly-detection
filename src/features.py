import numpy as np


def clip_features(spec):
    """Summarise one log-Mel spectrogram (128 x T) as 256 numbers:
    the mean and the std of each Mel band over time."""
    spec = np.asarray(spec, dtype=np.float32)
    return np.concatenate([spec.mean(axis=1), spec.std(axis=1)])


def features_for(specs, indices):
    return np.stack([clip_features(specs[i]) for i in indices])