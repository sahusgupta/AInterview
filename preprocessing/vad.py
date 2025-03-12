import librosa
import numpy as np

def apply_vad(y, top_db=20):
    """Applies Voice Activity Detection (VAD) to remove silence."""
    intervals = librosa.effects.split(y, top_db=top_db)
    y_processed = np.concatenate([y[start:end] for start, end in intervals])
    return y_processed

if __name__ == "__main__":
    sample_audio_path = "./data/input_audio/sample.wav"
    y, sr = librosa.load(sample_audio_path, sr=16000)
    y_vad = apply_vad(y)
    librosa.output.write_wav("./data/preprocessed_audio/sample_vad.wav", y_vad, sr)
    print("Voice Activity Detection applied.")