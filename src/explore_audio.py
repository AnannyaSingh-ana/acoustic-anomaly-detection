import matplotlib.pyplot as plt
import librosa.display
from config import TRAIN_DIR, TEST_DIR, FIGURES_DIR, HOP_LENGTH
from audio_utils import load_audio, log_mel_spectrogram


def main():
    # Count files
    train_files = list(TRAIN_DIR.glob("*.wav"))
    test_normal = sorted(TEST_DIR.glob("normal_*.wav"))
    test_anomaly = sorted(TEST_DIR.glob("anomaly_*.wav"))
    print(f"Train files (all normal): {len(train_files)}")
    print(f"Test normal: {len(test_normal)} | Test anomaly: {len(test_anomaly)}")

    # Pick one normal and one anomalous clip from the test set
    files = {"Normal": test_normal[0], "Anomaly": test_anomaly[0]}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for col, (label, path) in enumerate(files.items()):
        y, sr = load_audio(path)
        spec = log_mel_spectrogram(y, sr)
        print(f"{label}: {path.name} | {len(y)/sr:.2f} s | {sr} Hz | spectrogram {spec.shape}")

        librosa.display.waveshow(y, sr=sr, ax=axes[0, col])
        axes[0, col].set_title(f"{label} waveform\n{path.name}")

        img = librosa.display.specshow(spec, sr=sr, hop_length=HOP_LENGTH,
                                       x_axis="time", y_axis="mel", ax=axes[1, col])
        axes[1, col].set_title(f"{label} log-Mel spectrogram")
        fig.colorbar(img, ax=axes[1, col], format="%+2.0f dB")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "stage1_normal_vs_anomaly.png"
    plt.savefig(out, dpi=150)
    print(f"Saved figure to {out}")
    plt.show()


if __name__ == "__main__":
    main()