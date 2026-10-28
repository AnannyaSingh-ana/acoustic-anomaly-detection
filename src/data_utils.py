import numpy as np
import pandas as pd
from config import PROCESSED_DIR


def load_processed():
    df = pd.read_csv(PROCESSED_DIR / "metadata.csv", dtype={"machine_id": str})
    specs = np.load(PROCESSED_DIR / "spectrograms.npy", mmap_mode="r")
    stats = np.load(PROCESSED_DIR / "norm_stats.npz")
    return df, specs, stats["mean"], stats["std"]