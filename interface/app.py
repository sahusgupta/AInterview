import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
from moviepy import VideoFileClip
import tempfile

from preprocessing.preprocess_audio import load_and_preprocess
from preprocessing.vad import apply_vad
from preprocessing.feature_extraction import extract_features
from transcription.whisper_transcribe import attempt_transcribe
from detection.scoring import compute_likelihood

app = Flask(__name__)
app.secret_key = "SOME_SUPER_SECRET_KEY"
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  
ALLOWED_EXTENSIONS = {'mp4', 'wav', 'mp3'}

# In-memory user store: {username: {"password": ..., "recordings": [...]}}
users = {}

##############################################
# Utility & Auth Functions
##############################################
def is_logged_in():
    return session.get("username", False)

def get_current_user():
    return session.get("username")

##############################################
# Routes
##############################################

app.config["GLADIA_API_KEY"] = os.getenv("GLADIA_API_KEY", "c3bd73ad-2ee7-4663-9f82-564e84516bd6")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def landing():

    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            flash("Please fill out all fields.", "error")
            return redirect(url_for("register"))    
        
        if users.get(username, False):
            flash("Username already taken!", "error")
            return redirect(url_for("register"))

        users[username] = {"password": password, "recordings": []}
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = users.get(username)
        if not user or user["password"] != password:
            flash("Invalid username or password!", "error")
            return redirect(url_for("login"))

        session["username"] = username
        flash("Logged in successfully!", "success")
        return redirect(url_for("dashboard"))
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    users.clear()
    flash("Logged out!", "info")
    return redirect("/")

@app.route("/dashboard", methods=["GET"])
def dashboard():
    if not is_logged_in():
        flash("Please log in.", "error")
        return redirect(url_for("login"))
    
    user_data = users.get(get_current_user(), "")
    # Create a route to serve audio files
    audio_urls = {rec["id"]: url_for('serve_audio', recording_id=rec["id"]) 
                 for rec in user_data["recordings"]}
    return render_template("dashboard.html", recordings=user_data["recordings"], audio_urls=audio_urls)

@app.route("/audio/<int:recording_id>")
def serve_audio(recording_id):
    if not is_logged_in():
        return "Unauthorized", 401
    
    user_data = users.get(get_current_user(), "")
    if recording_id >= len(user_data["recordings"]):
        return "Recording not found", 404
    
    recording = user_data["recordings"][recording_id]
    
    # Ensure the path is absolute and exists
    file_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], recording["filename"]))
    
    if not os.path.exists(file_path):
        return f"Audio file not found: {recording['filename']}", 404
        
    try:
        return send_file(file_path, mimetype="audio/wav")
    except Exception as e:
        app.logger.error(f"Error serving audio file: {str(e)}")
        return "Error serving audio file", 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_from_video(video_path):
    """Extract audio from video file and save as WAV"""
    try:
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load video
            video = VideoFileClip(video_path)
            
            # Generate output audio path
            audio_filename = os.path.splitext(os.path.basename(video_path))[0] + '.wav'
            audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)
            
            # Extract audio and save as WAV
            video.audio.write_audiofile(audio_path)
            
            # Close video to free up resources
            video.close()
            
            return audio_path
    except Exception as e:
        raise Exception(f"Error extracting audio: {str(e)}")

@app.route("/upload", methods=["POST"])
def upload():
    if not is_logged_in():
        flash("Please log in first.", "error")
        return redirect(url_for("login"))

    file = request.files.get("file")
    if not file or file.filename == "":
        flash("No file selected!", "error")
        return redirect(url_for("dashboard"))

    if not allowed_file(file.filename):
        flash("Invalid file type! Please upload MP4, WAV, or MP3 files.", "error")
        return redirect(url_for("dashboard"))

    try:
        filename = secure_filename(file.filename)
        local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(local_path)

        # If file is MP4, extract audio
        if filename.lower().endswith('.mp4'):
            try:
                audio_path = extract_audio_from_video(local_path)
                # Remove original video file to save space
                os.remove(local_path)
                local_path = audio_path
                filename = os.path.basename(audio_path)
            except Exception as e:
                flash(f"Error processing video: {str(e)}", "error")
                return redirect(url_for("dashboard"))

        # Add the recording to user's list
        user_data = users.get(get_current_user(), "")
        recording_id = len(user_data["recordings"])
        user_data["recordings"].append({
            "id": recording_id,
            "filename": filename,
            "path": local_path,
            "transcript": "Not analyzed yet",
            "ai_score": None,
            "analyzed": False
        })
        flash("File uploaded successfully! Click 'Analyze' to process it.", "success")
    except Exception as e:
        flash(f"Error uploading file: {str(e)}", "error")

    return redirect(url_for("dashboard"))

@app.route("/analyze/<int:recording_id>", methods=["POST"])
def analyze_recording(recording_id):
    if not is_logged_in():
        flash("Please log in first.", "error")
        return redirect(url_for("login"))
    
    user_data = users.get(get_current_user(), "")
    
    # Check if recording exists
    if recording_id >= len(user_data["recordings"]):
        flash("Recording not found!", "error")
        return redirect(url_for("dashboard"))
    
    recording = user_data["recordings"][recording_id]
    local_path = recording["path"]
    
    try:
        # Process the audio through our AI pipeline
        y, sr = load_and_preprocess(local_path)
        if y is None or sr is None:
            raise Exception("Failed to load audio file")
            
        y_vad = apply_vad(y, sr)  # Add sr parameter
        if y_vad is None:
            raise Exception("Voice activity detection failed")
            
        feats = extract_features(y_vad, sr)
        if feats is None:
            raise Exception("Feature extraction failed")

        # Transcribe the audio
        if not app.config["GLADIA_API_KEY"]:
            raise Exception("Gladia API key not configured")
            
        transcript = attempt_transcribe(local_path, app.config["GLADIA_API_KEY"])
        if not transcript:
            raise Exception("Transcription failed")
        
        # Compute AI likelihood score
        ai_score = compute_likelihood(transcript, feats)
        if ai_score is None:
            raise Exception("Score computation failed")
        
        # Update the recording with analysis results
        recording["transcript"] = transcript
        recording["ai_score"] = ai_score
        recording["analyzed"] = True
        
        flash("Recording analyzed successfully!", "success")
    except Exception as e:
        flash(f"Error analyzing recording: {str(e)}", "error")
    
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    app.run(debug=True)
