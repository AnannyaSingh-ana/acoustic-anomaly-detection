import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import MODELS_DIR, METRICS_DIR, RANDOM_SEED, THRESHOLD_PERCENTILE
from data_utils import load_processed
from features import features_for
from metrics_utils import evaluate, print_report


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df, specs, _, _ = load_processed()
    idx = {s: df.index[df.split == s].values for s in ["train", "val", "test"]}

    # 1. Features
    X = {s: features_for(specs, idx[s]) for s in idx}
    print({s: X[s].shape for s in X})

    # 2. Fit scaler + Isolation Forest on TRAIN only
    scaler = StandardScaler().fit(X["train"])
    model = IsolationForest(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(scaler.transform(X["train"]))

    # 3. Anomaly scores (higher = more anomalous)
    def score(features):
        return -model.score_samples(scaler.transform(features))
    scores = {s: score(X[s]) for s in X}

    # 4. Overfitting check: train vs val should be similar
    test_df = df.loc[idx["test"]]
    t_lab = test_df["label"].values
    print("\nMean anomaly score:")
    print(f"  train (normal)      : {scores['train'].mean():.4f}")
    print(f"  val   (normal)      : {scores['val'].mean():.4f}")
    print(f"  test  (normal)      : {scores['test'][t_lab == 0].mean():.4f}")
    print(f"  test  (anomaly)     : {scores['test'][t_lab == 1].mean():.4f}")

    # 5. Threshold from VALIDATION only
    threshold = np.percentile(scores["val"], THRESHOLD_PERCENTILE)
    print(f"\nThreshold ({THRESHOLD_PERCENTILE}th percentile of val): {threshold:.4f}")

    # 6. Final evaluation on TEST
    summary, per_id = evaluate(t_lab, scores["test"], test_df["machine_id"], threshold)
    print_report("Isolation Forest", summary, per_id)

    # 7. Save model and results
    joblib.dump({"scaler": scaler, "model": model, "threshold": threshold},
                MODELS_DIR / "iforest.joblib")
    test_df[["file", "machine_id", "label"]].assign(
        score=scores["test"], pred=(scores["test"] > threshold).astype(int)
    ).to_csv(METRICS_DIR / "iforest_test_scores.csv", index=False)
    pd.DataFrame([{"model": "isolation_forest", **summary}]).to_csv(
        METRICS_DIR / "iforest_metrics.csv", index=False)
    per_id.to_csv(METRICS_DIR / "iforest_per_id_auc.csv", index=False)
    print(f"\nSaved model and results to {MODELS_DIR} and {METRICS_DIR}")


if __name__ == "__main__":
    main()