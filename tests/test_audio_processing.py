import pytest
import numpy as np
import librosa
from preprocessing.preprocess_audio import load_and_preprocess
from preprocessing.vad import apply_vad
from preprocessing.feature_extraction import extract_features

def test_audio_pipeline():
    """Test the full audio processing pipeline."""
    y = np.random.rand(16000 * 5)  # Generate 5 seconds of dummy audio
    sr = 16000
    
    # Step 1: Preprocess Audio (Pass NumPy array instead of file path)
    processed_audio, sr = load_and_preprocess(y)
    assert isinstance(processed_audio, np.ndarray)
    assert sr == 16000
    
    # Step 2: Apply Voice Activity Detection (VAD)
    y_vad = apply_vad(processed_audio)
    assert isinstance(y_vad, np.ndarray)
    assert len(y_vad) > 0  # Ensure VAD retains useful audio
    
    # Step 3: Extract Features
    features = extract_features(y_vad, sr)
    assert isinstance(features, dict)
    assert "mfccs" in features
    assert "spectral_centroid" in features
    assert "spectral_bandwidth" in features
    assert "spectral_rolloff" in features
    
    print("Audio processing pipeline test passed.")
