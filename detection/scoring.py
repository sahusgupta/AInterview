import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
def compute_likelihood(transcript: str, audio_features : dict):
    suspicious_keywords = ["furthermore", "indeed", "hello, my name is AI"]
    text_score = sum(transcript.lower().count(k) for k in suspicious_keywords)

    # Audio side: maybe we check MFCC variance
    mfccs = audio_features.get("mfccs")
    audio_score = 0.0
    if mfccs is not None:
        var_mfcc = np.var(mfccs)
        # Arbitrary scaling to get 0–1
        audio_score = min(var_mfcc / 5000, 1.0)

    # Weighted sum as a placeholder
    total_score = (0.6 * audio_score) + (0.4 * text_score)
    return float(min(total_score, 1.0))