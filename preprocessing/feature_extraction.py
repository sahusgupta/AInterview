import librosa
import numpy as np
import yaml

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

N_MFCC = config["preprocessing"]["mfcc"]["n_mfcc"]
HOP_LENGTH = config["preprocessing"]["mfcc"]["hop_length"]


def extract_mfcc(y, sr):
    """Extracts MFCC features from an audio signal."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    return mfccs


def extract_spectral_features(y, sr):
    """Extracts spectral features such as centroid, bandwidth, and roll-off."""
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return spectral_centroid, spectral_bandwidth, spectral_rolloff


def extract_features(y, sr):
    """Extracts all relevant audio features for analysis."""
    mfccs = extract_mfcc(y, sr)
    spectral_centroid, spectral_bandwidth, spectral_rolloff = extract_spectral_features(y, sr)
    return {
        "mfccs": mfccs,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
    }

if __name__ == "__main__":
    sample_audio_path = "./data/input_audio/sample.wav"
    y, sr = librosa.load(sample_audio_path, sr=16000)
    features = extract_features(y, sr)
    print("Extracted Features:", features)
