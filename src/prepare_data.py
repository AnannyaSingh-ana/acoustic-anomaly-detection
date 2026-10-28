import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import (TRAIN_DIR, TEST_DIR, PROCESSED_DIR, N_MELS, N_FRAMES,
                    VAL_FRACTION, RANDOM_SEED)
from audio_utils import load_audio, log_mel_spectrogram


def parse_filename(name):
    """'anomaly_id_02_00000012.wav' -> (label=1, machine_id='02')"""
    m = re.match(r"(normal|anomaly)_id_(\d+)_", name)
    if m is None:
        raise ValueError(f"Unexpected filename: {name}")
    return int(m.group(1) == "anomaly"), m.group(2)


def build_metadata():
    """One row per file: path, machine ID, label (0 normal / 1 anomaly), split."""
    rows = []
    for split, folder in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        for path in sorted(folder.glob("*.wav")):
            label, machine_id = parse_filename(path.name)
            rows.append({"file": path.name, "path": str(path),
                         "machine_id": machine_id, "label": label, "split": split})
    df = pd.DataFrame(rows)

    # Safety check: training data must contain only normal sounds
    assert df.loc[df.split == "train", "label"].sum() == 0, "Anomaly found in train!"

    # Move 10% of train into validation, keeping machine IDs balanced
    train_idx = df.index[df.split == "train"]
    _, val_idx = train_test_split(train_idx, test_size=VAL_FRACTION,
                                  random_state=RANDOM_SEED,
                                  stratify=df.loc[train_idx, "machine_id"])
    df.loc[val_idx, "split"] = "val"
    return df


def fix_length(spec):
    """Crop or pad so every spectrogram has exactly N_FRAMES columns."""
    if spec.shape[1] >= N_FRAMES:
        return spec[:, :N_FRAMES]
    return np.pad(spec, ((0, 0), (0, N_FRAMES - spec.shape[1])), mode="edge")


def compute_spectrograms(df):
    specs = np.zeros((len(df), N_MELS, N_FRAMES), dtype=np.float16)  # float16 saves RAM
    for i, path in enumerate(tqdm(df["path"], desc="Computing spectrograms")):
        y, sr = load_audio(path)
        specs[i] = fix_length(log_mel_spectrogram(y, sr))
    return specs


def compute_norm_stats(specs, mask):
    """Mean and std per Mel band, using ONLY the given rows (train split)."""
    total = np.zeros(N_MELS)
    total_sq = np.zeros(N_MELS)
    count = 0
    for i in np.where(mask)[0]:
        s = specs[i].astype(np.float64)
        total += s.sum(axis=1)
        total_sq += (s ** 2).sum(axis=1)
        count += s.shape[1]
    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = build_metadata()
    print("\nFiles per split / machine ID / label (0=normal, 1=anomaly):")
    print(pd.crosstab([df.split, df.label], df.machine_id, margins=True))

    specs = compute_spectrograms(df)
    mean, std = compute_norm_stats(specs, (df.split == "train").values)

    df.to_csv(PROCESSED_DIR / "metadata.csv", index=False)
    np.save(PROCESSED_DIR / "spectrograms.npy", specs)
    np.savez(PROCESSED_DIR / "norm_stats.npz", mean=mean, std=std)

    print(f"\nSaved spectrograms: {specs.shape} -> {PROCESSED_DIR}")
    print(f"Norm stats (train only): mean range {mean.min():.1f} to {mean.max():.1f} dB, "
          f"std range {std.min():.1f} to {std.max():.1f} dB")


if __name__ == "__main__":
    main()