# RealTalk

## Overview
RealTalk is an advanced AI-powered system designed to analyze interview recordings and determine the likelihood that a candidate used AI-generated responses. It integrates cutting-edge speech processing, natural language analysis, and machine learning models to assess speech authenticity, ensuring interview integrity and human-driven communication.

## Features
### 🔹 Audio Preprocessing
- Noise reduction and silence removal (Voice Activity Detection - VAD)
- MFCC feature extraction for deep speech analysis

### 🔹 Speech-to-Text Transcription
- OpenAI Whisper or Google Speech-to-Text API for transcription
- Handles multiple languages and accents

### 🔹 NLP-Based AI Detection
- Analysis of unnatural phrasing and sentence structuring
- Detection of robotic or AI-generated speech patterns

### 🔹 AI-Likelihood Scoring
- Weighted scoring model integrating both audio and text-based features
- Threshold-based decision-making for AI identification

### 🔹 Intuitive Web Interface
- Simple UI for uploading and analyzing interview recordings
- Displays AI-likelihood results with detailed breakdowns

## Installation
### Requirements
Ensure you have Python 3.8+ installed.
Install dependencies using:
```sh
pip install -r requirements.txt
```

## Usage
To analyze an interview recording without using the interface, run:
```sh
python main.py --input data/input_audio/sample.wav
```

## Project Structure
```
RealTalk/
│── config.yaml                    # Configuration settings
│── requirements.txt                # Dependencies
│── main.py                         # Main execution script
│
├── data/                           # Data storage
│   ├── input_audio/                # Raw interview recordings
│   ├── preprocessed_audio/         # Processed and cleaned audio
│   ├── transcriptions/             # Transcribed text data
│   ├── analysis_results/           # AI-likelihood detection results
│
├── preprocessing/                  # Audio preprocessing scripts
├── transcription/                  # Speech-to-text modules
├── nlp_analysis/                    # Text-based AI detection models
├── detection/                       # AI-likelihood scoring system
├── interface/                       # Web interface components
├── utils/                           # Helper functions and utilities
├── tests/                           # Unit and integration tests
└── README.md                        # Documentation
```

## Configuration
The `config.yaml` file allows customization of system settings. Example:
```yaml
audio:
  input_dir: "./data/input_audio"
  output_dir: "./data/preprocessed_audio"
  sample_rate: 16000

preprocessing:
  noise_reduction: true
  vad: true

transcription:
  api: "whisper"
  model: "base"

nlp:
  model: "en_core_web_sm"

detection:
  weight_audio: 0.6
  weight_text: 0.4
  threshold: 0.5
```

## Testing
Run unit tests using:
```sh
pytest tests/
```

## Future Roadmap
- 📈 Train a deep-learning-based AI detection model
- 🎙️ Expand support for additional transcription services
- 📊 Develop an interactive dashboard for insights visualization

## License
**Copyright © 2025 [Your Name]. All rights reserved.**

This software is proprietary and may not be copied, modified, distributed, or used for commercial purposes without explicit written permission from the owner. Unauthorized use of this software is strictly prohibited.

For licensing inquiries, contact: [your_email@example.com]
