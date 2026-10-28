import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix)


def evaluate(labels, scores, machine_ids, threshold):
    labels, scores, machine_ids = map(np.asarray, (labels, scores, machine_ids))
    preds = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    summary = {
        "roc_auc": roc_auc_score(labels, scores),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "false_alarm_rate": fp / (fp + tn),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    per_id = pd.DataFrame([
        {"machine_id": m,
         "roc_auc": roc_auc_score(labels[machine_ids == m], scores[machine_ids == m])}
        for m in sorted(set(machine_ids))
    ])
    return summary, per_id


def print_report(name, summary, per_id):
    print(f"\n===== {name}: TEST RESULTS =====")
    print(f"ROC-AUC          : {summary['roc_auc']:.3f}   (0.5 = random guessing)")
    print(f"Precision        : {summary['precision']:.3f}")
    print(f"Recall           : {summary['recall']:.3f}")
    print(f"F1               : {summary['f1']:.3f}")
    print(f"False alarm rate : {summary['false_alarm_rate']:.3f}   (normal clips wrongly flagged)")
    print(f"Confusion: TP={summary['tp']} FP={summary['fp']} TN={summary['tn']} FN={summary['fn']}")
    print("\nROC-AUC per machine ID:")
    print(per_id.to_string(index=False, float_format="%.3f"))