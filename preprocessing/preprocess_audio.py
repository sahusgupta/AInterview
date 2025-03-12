import os
import librosa
import numpy as np
import yaml
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

INPUT_DIR = config["audio"]["input_dir"]
OUTPUT_DIR = config["audio"]["output_dir"]
SAMPLE_RATE = config["audio"]["sample_rate"]
N_MFCC = config["preprocessing"]["mfcc"]["n_mfcc"]
HOP_LENGTH = config["preprocessing"]["mfcc"]["hop_length"]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess(audio_input):
    """Load audio, apply noise reduction, segment silence, and extract MFCCs."""
    
    if isinstance(audio_input, str):  # If it's a file path
        if not audio_input.endswith(".wav"):
            audio = AudioSegment.from_file(audio_input)
            audio_input = audio_input.replace(os.path.splitext(audio_input)[-1], ".wav")
            audio.export(audio_input, format="wav")
        
        # Load audio file
        y, sr = librosa.load(audio_input, sr=SAMPLE_RATE)

    elif isinstance(audio_input, np.ndarray):  # If raw NumPy array is passed
        y = audio_input
        sr = SAMPLE_RATE
    else:
        raise ValueError("Invalid audio input type. Expected file path or NumPy array.")

    # Apply noise reduction if enabled
    if config["preprocessing"]["noise_reduction"]:
        y = nr.reduce_noise(y=y, sr=sr)

    return y, sr


if __name__ == "__main__":
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            file_path = os.path.join(INPUT_DIR, filename)
            processed_audio, mfccs = load_and_preprocess(file_path)
            np.save(os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_mfcc.npy"), mfccs)
            print(f"Processed: {filename}")
