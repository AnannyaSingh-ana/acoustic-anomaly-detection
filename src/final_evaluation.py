import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from config import MODELS_DIR, METRICS_DIR, FIGURES_DIR, THRESHOLD_PERCENTILE, MACHINE_IDS
from data_utils import load_processed
from id_classifier import IDClassifier, clip_id_scores
from metrics_utils import evaluate, print_report

METRIC_COLS = (["roc_auc"] + [f"auc_id_{m}" for m in MACHINE_IDS]
               + ["precision", "recall", "f1", "false_alarm_rate"])


def result_row(name, labels, scores, machine_ids, threshold):
    summary, per_id = evaluate(labels, scores, machine_ids, threshold)
    print_report(f"{name} (FINAL half)", summary, per_id)
    return {"model": name, **summary,
            **{f"auc_id_{r.machine_id}": r.roc_auc for r in per_id.itertuples()}}


def main():
    df, specs, mean, std = load_processed()
    test = df[df.split == "test"]
    final = test[test.test_part == "final"]
    print(f"FINAL half: {len(final)} clips "
          f"({(final.label == 0).sum()} normal, {(final.label == 1).sum()} anomaly)")

    rows, roc_data = [], {}

    # 1-2. Isolation Forest and CNN Autoencoder (scores were saved for the whole test set)
    for name, csv_name, thr in [
        ("Isolation Forest", "iforest_test_scores.csv",
         joblib.load(MODELS_DIR / "iforest.joblib")["threshold"]),
        ("CNN Autoencoder", "cnn_ae_test_scores.csv",
         float(np.load(MODELS_DIR / "cnn_ae_threshold.npy"))),
    ]:
        s = pd.read_csv(METRICS_DIR / csv_name, dtype={"machine_id": str})
        s = s[s.file.isin(final.file)]
        rows.append(result_row(name, s.label, s.score, s.machine_id, thr))
        roc_data[name] = (s.label.values, s.score.values)

    # 3. Machine-ID classifier: every trained seed
    specs = np.array(specs)
    id_to_idx = {m: i for i, m in enumerate(MACHINE_IDS)}
    val_idx = df.index[df.split == "val"].values
    fin_idx = final.index.values
    val_lab = df.loc[val_idx, "machine_id"].map(id_to_idx).values
    fin_lab = df.loc[fin_idx, "machine_id"].map(id_to_idx).values

    seed_rows = []
    for f in sorted(MODELS_DIR.glob("id_classifier*.pt")):
        model = IDClassifier()
        model.load_state_dict(torch.load(f)["state_dict"])
        thr = np.percentile(clip_id_scores(model, specs, val_idx, val_lab, mean, std),
                            THRESHOLD_PERCENTILE)
        sc = clip_id_scores(model, specs, fin_idx, fin_lab, mean, std)
        seed_rows.append(result_row(f"ID Classifier [{f.stem}]", final.label, sc,
                                    final.machine_id, thr))
        if f.stem == "id_classifier":  # main model (seed 42), fixed in advance
            roc_data["Machine-ID Classifier"] = (final.label.values, sc)

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(METRICS_DIR / "final_id_clf_seeds.csv", index=False)
    means = seed_df[METRIC_COLS].mean()
    stds = seed_df[METRIC_COLS].std() if len(seed_df) > 1 else means * 0
    rows.append({"model": f"Machine-ID Classifier (mean of {len(seed_df)} seeds)", **means.to_dict()})

    print("\n===== ID CLASSIFIER: EACH SEED (FINAL half) =====")
    print(seed_df[["model"] + METRIC_COLS].to_string(index=False, float_format="%.3f"))
    print("\nStd across seeds:")
    print(stds.to_string(float_format="%.3f"))

    final_df = pd.DataFrame(rows)
    final_df.to_csv(METRICS_DIR / "final_results.csv", index=False)
    print("\n===== FINAL RESULTS (untouched FINAL half) =====")
    print(final_df[["model"] + METRIC_COLS].to_string(index=False, float_format="%.3f"))

    # Figure 1: ROC curves
    plt.figure(figsize=(6.5, 6))
    for name, (y, s) in roc_data.items():
        fpr, tpr, _ = roc_curve(y, s)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC 0.5)")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves: final test half"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "final_roc_curves.png", dpi=150)

    # Figure 2: ROC-AUC per fan
    plt.figure(figsize=(9, 5))
    x, w = np.arange(len(MACHINE_IDS)), 0.27
    bars = [("Isolation Forest", rows[0], None),
            ("CNN Autoencoder", rows[1], None),
            ("Machine-ID Classifier", rows[2],
             [stds[f"auc_id_{m}"] for m in MACHINE_IDS] if len(seed_df) > 1 else None)]
    for k, (name, r, err) in enumerate(bars):
        plt.bar(x + (k - 1) * w, [r[f"auc_id_{m}"] for m in MACHINE_IDS], w,
                yerr=err, capsize=4, label=name)
    plt.axhline(0.5, ls="--", color="gray", label="Random")
    plt.xticks(x, [f"Fan {m}" for m in MACHINE_IDS]); plt.ylim(0, 1.05)
    plt.ylabel("ROC-AUC"); plt.title("ROC-AUC per fan: final test half")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_per_id_auc.png", dpi=150)
    print("\nSaved final_results.csv, final_id_clf_seeds.csv and 2 figures.")


if __name__ == "__main__":
    main()