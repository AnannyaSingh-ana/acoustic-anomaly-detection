import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from config import MODELS_DIR, METRICS_DIR, FIGURES_DIR, THRESHOLD_PERCENTILE
from data_utils import load_processed
from model import CNNAutoencoder, clip_scores
from metrics_utils import evaluate, print_report


def main():
    df, specs, mean, std = load_processed()
    specs = np.array(specs)
    ckpt = torch.load(MODELS_DIR / "cnn_autoencoder.pt")
    model = CNNAutoencoder(ckpt["latent_dim"])
    model.load_state_dict(ckpt["state_dict"])

    idx = {s: df.index[df.split == s].values for s in ["train", "val", "test"]}
    scores = {s: clip_scores(model, specs, idx[s], mean, std) for s in idx}
    test_df = df.loc[idx["test"]]
    t_lab = test_df["label"].values

    print("Mean anomaly score (reconstruction error):")
    print(f"  train (normal)  : {scores['train'].mean():.4f}")
    print(f"  val   (normal)  : {scores['val'].mean():.4f}")
    print(f"  test  (normal)  : {scores['test'][t_lab == 0].mean():.4f}")
    print(f"  test  (anomaly) : {scores['test'][t_lab == 1].mean():.4f}")

    threshold = np.percentile(scores["val"], THRESHOLD_PERCENTILE)
    print(f"\nThreshold ({THRESHOLD_PERCENTILE}th percentile of val): {threshold:.4f}")

    summary, per_id = evaluate(t_lab, scores["test"], test_df["machine_id"], threshold)
    print_report("CNN Autoencoder", summary, per_id)

    # Save results
    np.save(MODELS_DIR / "cnn_ae_threshold.npy", threshold)
    test_df[["file", "machine_id", "label"]].assign(
        score=scores["test"], pred=(scores["test"] > threshold).astype(int)
    ).to_csv(METRICS_DIR / "cnn_ae_test_scores.csv", index=False)
    pd.DataFrame([{"model": "cnn_autoencoder", **summary}]).to_csv(
        METRICS_DIR / "cnn_ae_metrics.csv", index=False)
    per_id.to_csv(METRICS_DIR / "cnn_ae_per_id_auc.csv", index=False)

    # Histogram: normal vs anomaly scores on test
    plt.figure(figsize=(8, 5))
    plt.hist(scores["test"][t_lab == 0], bins=50, alpha=0.6, density=True, label="Normal (test)")
    plt.hist(scores["test"][t_lab == 1], bins=50, alpha=0.6, density=True, label="Anomaly (test)")
    plt.axvline(threshold, color="k", linestyle="--", label="Threshold")
    plt.xlabel("Anomaly score"); plt.ylabel("Density"); plt.legend()
    plt.title("CNN autoencoder: test score distribution"); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cnn_ae_score_hist.png", dpi=150)

    # Side-by-side comparison with the baseline
    iforest_file = METRICS_DIR / "iforest_metrics.csv"
    if iforest_file.exists():
        comp = pd.concat([pd.read_csv(iforest_file), pd.read_csv(METRICS_DIR / "cnn_ae_metrics.csv")])
        cols = ["model", "roc_auc", "precision", "recall", "f1", "false_alarm_rate"]
        print("\n===== COMPARISON =====")
        print(comp[cols].to_string(index=False, float_format="%.3f"))
        comp.to_csv(METRICS_DIR / "comparison.csv", index=False)


if __name__ == "__main__":
    main()