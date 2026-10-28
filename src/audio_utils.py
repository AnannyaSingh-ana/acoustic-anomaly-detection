import numpy as np
import librosa
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS


def load_audio(path):
    """Load a WAV file as a mono signal at 16 kHz."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y, sr


def log_mel_spectrogram(y, sr=SAMPLE_RATE):
    """Convert audio into a log-Mel spectrogram (in decibels)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    return librosa.power_to_db(mel, ref=1.0)