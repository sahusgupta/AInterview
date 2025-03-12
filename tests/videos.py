# Make sure you have the required libraries installed:
# pip install pyttsx3 moviepy

import pyttsx3
from moviepy import VideoFileClip, AudioFileClip

# Define the placeholder scripts for the two candidates
script_candidate1 = (
    "Hello, my name is Candidate One. I am excited to be here today and share my experience and vision. "
    "I have a background in technology and innovation, and I look forward to discussing how my skills and passion "
    "align with this opportunity. Thank you for your time and consideration."
)

script_candidate2 = (
    "Hi, I am Candidate Two. I appreciate the opportunity to present my qualifications and share my journey in the field "
    "of innovation. I bring years of experience, a commitment to excellence, and a strong drive to contribute to your team. "
    "Thank you for this opportunity."
)

# Initialize the pyttsx3 engine for offline text-to-speech conversion
engine = pyttsx3.init()

# Generate audio for Candidate One and save as a WAV file
candidate1_audio_path = "candidate1_audio.wav"
engine.save_to_file(script_candidate1, candidate1_audio_path)
engine.runAndWait()
print(f"Generated {candidate1_audio_path}")

# Generate audio for Candidate Two and save as a WAV file
candidate2_audio_path = "candidate2_audio.wav"
engine.save_to_file(script_candidate2, candidate2_audio_path)
engine.runAndWait()
print(f"Generated {candidate2_audio_path}")

# Define paths to your video files (update these paths if necessary)
video1_path = "20250312_0050_Interview with AI Twist_simple_compose_01jp4efdj1fr8ba7nz1378bb8p.mp4"
video2_path = "20250312_0050_Interview with AI Twist_simple_compose_01jp4efdhwfnzrcccxc6y95v1h.mp4"

# Process the first video: load the video and the generated audio
try:
    video1 = VideoFileClip(video1_path)
    audio1 = AudioFileClip(candidate1_audio_path)
    # Instead of using set_audio(), assign the audio directly
    video1.audio = audio1
    output_video1 = "video1_with_voiceover.mp4"
    video1.write_videofile(output_video1, codec="libx264", audio_codec="aac")
    print(f"Created {output_video1}")
except Exception as e:
    print(f"Error processing video 1: {e}")

# Process the second video: load the video and the corresponding generated audio
try:
    video2 = VideoFileClip(video2_path)
    audio2 = AudioFileClip(candidate2_audio_path)
    video2.audio = audio2
    output_video2 = "video2_with_voiceover.mp4"
    video2.write_videofile(output_video2, codec="libx264", audio_codec="aac")
    print(f"Created {output_video2}")
except Exception as e:
    print(f"Error processing video 2: {e}")
