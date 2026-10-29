import io
import re
import sys
from pathlib import Path

import joblib
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

# Let the dashboard import our code from the src/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import (PROCESSED_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR,
                    SAMPLE_RATE, HOP_LENGTH, N_FRAMES, MACHINE_IDS)
from audio_utils import load_audio, log_mel_spectrogram
from prepare_data import fix_length
from features import clip_features
from model import CNNAutoencoder, clip_to_patches, patch_starts
from id_classifier import IDClassifier

st.set_page_config(page_title="Acoustic Anomaly Detection", page_icon="🔊", layout="wide")

MODEL_NAMES = ["Machine-ID Classifier (best)", "CNN Autoencoder", "Isolation Forest (baseline)"]


# ---------- Loading (cached so it only happens once) ----------
@st.cache_resource
def load_models():
    stats = np.load(PROCESSED_DIR / "norm_stats.npz")
    clf = IDClassifier()
    clf.load_state_dict(torch.load(MODELS_DIR / "id_classifier.pt")["state_dict"])
    clf.eval()
    ckpt = torch.load(MODELS_DIR / "cnn_autoencoder.pt")
    ae = CNNAutoencoder(ckpt["latent_dim"])
    ae.load_state_dict(ckpt["state_dict"])
    ae.eval()
    iforest = joblib.load(MODELS_DIR / "iforest.joblib")
    thresholds = {
        MODEL_NAMES[0]: float(np.load(MODELS_DIR / "id_clf_threshold.npy")),
        MODEL_NAMES[1]: float(np.load(MODELS_DIR / "cnn_ae_threshold.npy")),
        MODEL_NAMES[2]: float(iforest["threshold"]),
    }
    return stats["mean"], stats["std"], clf, ae, iforest, thresholds


@st.cache_data
def load_test_metadata():
    df = pd.read_csv(PROCESSED_DIR / "metadata.csv", dtype={"machine_id": str})
    return df[df.split == "test"].reset_index(drop=True)


# ---------- Scoring ----------
@torch.no_grad()
def score_clip(spec, model_name, machine_id, mean, std, clf, ae, iforest):
    details = {}
    if model_name == MODEL_NAMES[0]:
        x = torch.from_numpy(clip_to_patches(spec, mean, std))
        logp = F.log_softmax(clf(x), dim=1)
        k = MACHINE_IDS.index(machine_id)
        score = float(-logp[:, k].mean())
        details["probs"] = logp.exp().mean(dim=0).numpy()
    elif model_name == MODEL_NAMES[1]:
        x = torch.from_numpy(clip_to_patches(spec, mean, std))
        errors = ((ae(x) - x) ** 2).mean(dim=(1, 2, 3)).numpy()
        score = float(errors.mean())
        details["patch_errors"] = errors
    else:
        f = iforest["scaler"].transform(clip_features(spec)[None])
        score = float(-iforest["model"].score_samples(f)[0])
    return score, details


# ---------- Plots ----------
def plot_spectrogram(spec):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    img = librosa.display.specshow(spec, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                   x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Log-Mel spectrogram")
    fig.tight_layout()
    return fig


def plot_id_probs(probs, machine_id):
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["tab:green" if m == machine_id else "tab:gray" for m in MACHINE_IDS]
    ax.bar([f"Fan {m}" for m in MACHINE_IDS], probs, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean probability")
    ax.set_title(f"How sure is the model that this is Fan {machine_id}?")
    fig.tight_layout()
    return fig


def plot_patch_errors(errors, threshold):
    t = [s * HOP_LENGTH / SAMPLE_RATE for s in patch_starts(N_FRAMES)]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, errors, marker="o", label="Patch error")
    ax.axhline(threshold, ls="--", color="red", label="Clip threshold")
    ax.set_xlabel("Patch start time (s)")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Reconstruction error over time")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------- Tab 1: analyse a recording ----------
def analyse_tab(model_name, source, models, test_df):
    mean, std, clf, ae, iforest, thresholds = models
    true_label = None

    if source == "Pick a test clip":
        c1, c2 = st.columns(2)
        fan = c1.selectbox("Fan", MACHINE_IDS, format_func=lambda m: f"Fan {m}")
        kind = c2.selectbox("Type", ["All", "Normal", "Anomaly"])
        subset = test_df[test_df.machine_id == fan]
        if kind != "All":
            subset = subset[subset.label == (1 if kind == "Anomaly" else 0)]
        name = st.selectbox("Recording", subset.file.tolist())
        row = subset[subset.file == name].iloc[0]
        audio_bytes = Path(row.path).read_bytes()
        machine_id, true_label = fan, int(row.label)
    else:
        upload = st.file_uploader("Upload a WAV recording", type=["wav"])
        if upload is None:
            st.info("Upload a .wav file to analyse it.")
            return
        name, audio_bytes = upload.name, upload.getvalue()
        m = re.search(r"id_(\d+)", name)
        guess = m.group(1) if m and m.group(1) in MACHINE_IDS else MACHINE_IDS[0]
        machine_id = st.selectbox("Which fan was this recorded from?", MACHINE_IDS,
                                  index=MACHINE_IDS.index(guess),
                                  format_func=lambda m: f"Fan {m}")

    try:
        y, sr = load_audio(io.BytesIO(audio_bytes))
    except Exception as e:
        st.error(f"Could not read this audio file: {e}")
        return
    duration = len(y) / sr
    if abs(duration - 10) > 0.5:
        st.warning(f"This clip is {duration:.1f} s long. The models were trained on 10 s "
                   "clips, so it was cropped or padded to 10 s.")

    spec = fix_length(log_mel_spectrogram(y, sr))
    score, details = score_clip(spec, model_name, machine_id, mean, std, clf, ae, iforest)
    threshold = thresholds[model_name]
    is_anomaly = score > threshold

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(f"**{name}**")
        st.audio(audio_bytes, format="audio/wav")
        if is_anomaly:
            st.error("⚠️ ANOMALY: this sound differs from normal operation")
        else:
            st.success("✅ NORMAL: this sound matches normal operation")
        m1, m2 = st.columns(2)
        m1.metric("Anomaly score", f"{score:.4f}")
        m2.metric("Threshold", f"{threshold:.4f}")
        if true_label is not None:
            truth = "Anomaly" if true_label == 1 else "Normal"
            correct = (true_label == 1) == is_anomaly
            st.info(f"Ground truth: **{truth}** → prediction is "
                    f"**{'correct' if correct else 'incorrect'}**")
        st.caption("An anomaly flag means the sound is unusual. It does not identify "
                   "which mechanical fault is present.")
    with right:
        fig = plot_spectrogram(spec)
        st.pyplot(fig)
        plt.close(fig)
        if "probs" in details:
            fig = plot_id_probs(details["probs"], machine_id)
            st.pyplot(fig)
            plt.close(fig)
        if "patch_errors" in details:
            fig = plot_patch_errors(details["patch_errors"], threshold)
            st.pyplot(fig)
            plt.close(fig)


# ---------- Tab 2: results ----------
def results_tab():
    st.subheader("Final results on the held-out test half (938 clips)")
    path = METRICS_DIR / "final_results.csv"
    if path.exists():
        df = pd.read_csv(path)
        nice = {"model": "Model", "roc_auc": "ROC-AUC",
                **{f"auc_id_{m}": f"AUC Fan {m}" for m in MACHINE_IDS},
                "precision": "Precision", "recall": "Recall", "f1": "F1",
                "false_alarm_rate": "False alarm rate"}
        st.dataframe(df[list(nice)].rename(columns=nice).round(3), hide_index=True)
    else:
        st.warning("final_results.csv not found. Run src/final_evaluation.py first.")

    c1, c2 = st.columns(2)
    for col, img in [(c1, "final_roc_curves.png"), (c2, "final_per_id_auc.png")]:
        if (FIGURES_DIR / img).exists():
            col.image(str(FIGURES_DIR / img))

    st.markdown("""
**Notes**
- All models were trained only on normal sounds. Thresholds were set at the 95th percentile
  of validation scores, so roughly 5% of normal clips are expected to be flagged.
- The Machine-ID Classifier result is the mean of 3 random seeds (std ≈ 0.01 ROC-AUC).
- Fan 00 is near chance for all methods. This is a known limitation of the current approach.
""")


# ---------- Main ----------
def main():
    st.title("🔊 Acoustic Anomaly Detection for Industrial Fans")
    st.caption("Deep learning models trained only on normal sounds from the "
               "DCASE 2020 Task 2 fan dataset.")

    try:
        models = load_models()
    except FileNotFoundError as e:
        st.error(f"A model file is missing: {e}. Run the training scripts first.")
        return
    test_df = load_test_metadata()

    with st.sidebar:
        st.header("Settings")
        model_name = st.radio("Model", MODEL_NAMES)
        source = st.radio("Audio source", ["Pick a test clip", "Upload a WAV file"])

    tab1, tab2 = st.tabs(["🎧 Analyse a recording", "📊 Model results"])
    with tab1:
        analyse_tab(model_name, source, models, test_df)
    with tab2:
        results_tab()


main()