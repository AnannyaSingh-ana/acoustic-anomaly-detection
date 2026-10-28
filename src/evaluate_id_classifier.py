import joblib
import numpy as np
import pandas as pd
import torch
from config import MODELS_DIR, METRICS_DIR, THRESHOLD_PERCENTILE, MACHINE_IDS
from data_utils import load_processed
from id_classifier import IDClassifier, clip_id_scores
from metrics_utils import evaluate, print_report


def main():
    df, specs, mean, std = load_processed()
    specs = np.array(specs)
    model = IDClassifier()
    model.load_state_dict(torch.load(MODELS_DIR / "id_classifier.pt")["state_dict"])

    id_to_idx = {m: i for i, m in enumerate(MACHINE_IDS)}
    idx = {s: df.index[df.split == s].values for s in ["val", "test"]}
    labels = {s: df.loc[idx[s], "machine_id"].map(id_to_idx).values for s in idx}
    scores = {s: clip_id_scores(model, specs, idx[s], labels[s], mean, std) for s in idx}

    threshold = np.percentile(scores["val"], THRESHOLD_PERCENTILE)
    np.save(MODELS_DIR / "id_clf_threshold.npy", threshold)
    test_df = df.loc[idx["test"]]
    test_df[["file", "machine_id", "label"]].assign(
        score=scores["test"], pred=(scores["test"] > threshold).astype(int)
    ).to_csv(METRICS_DIR / "id_clf_test_scores.csv", index=False)
    print(f"Threshold ({THRESHOLD_PERCENTILE}th percentile of val): {threshold:.4f}")

    # ---- Compare all models on the DEV half only ----
    test_parts = df[df.split == "test"][["file", "test_part"]]
    models = {
        "Isolation Forest": ("iforest_test_scores.csv",
                             joblib.load(MODELS_DIR / "iforest.joblib")["threshold"]),
        "CNN Autoencoder": ("cnn_ae_test_scores.csv",
                            float(np.load(MODELS_DIR / "cnn_ae_threshold.npy"))),
        "Machine-ID Classifier": ("id_clf_test_scores.csv", threshold),
    }
    rows = []
    for name, (csv_name, thr) in models.items():
        s = pd.read_csv(METRICS_DIR / csv_name, dtype={"machine_id": str}).merge(test_parts, on="file")
        dev = s[s.test_part == "dev"]
        summary, per_id = evaluate(dev["label"], dev["score"], dev["machine_id"], thr)
        print_report(f"{name} (DEV half)", summary, per_id)
        rows.append({"model": name, **summary,
                     **{f"auc_id_{r.machine_id}": r.roc_auc for r in per_id.itertuples()}})

    comp = pd.DataFrame(rows)
    comp.to_csv(METRICS_DIR / "dev_comparison.csv", index=False)
    cols = ["model", "roc_auc"] + [f"auc_id_{m}" for m in MACHINE_IDS] + ["recall", "false_alarm_rate"]
    print("\n===== DEV-HALF COMPARISON =====")
    print(comp[cols].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()