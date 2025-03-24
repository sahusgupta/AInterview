import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename

from preprocessing.preprocess_audio import load_and_preprocess
from preprocessing.vad import apply_vad
from preprocessing.feature_extraction import extract_features
from transcription.whisper_transcribe import attempt_transcribe
from detection.scoring import compute_likelihood

app = Flask(__name__)
app.secret_key = "SOME_SUPER_SECRET_KEY"
app.config["UPLOAD_FOLDER"] = "uploads"

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

app.config["GLADIA_API_KEY"] = os.getenv("GLADIA_API_KEY", "")

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
    return render_template("dashboard.html", recordings=user_data["recordings"])

@app.route("/upload", methods=["POST"])
def upload():
    if not is_logged_in():
        flash("Please log in first.", "error")
        return redirect(url_for("login"))

    file = request.files.get("file")
    if not file or file.filename == "":
        flash("No file selected!", "error")
        return redirect(url_for("dashboard"))

    filename = secure_filename(file.filename)
    local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(local_path)

    try:
        # Add the recording to user's list without processing
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
        flash(f"Error uploading file: {e}", "error")

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
        y_vad = apply_vad(y)
        feats = extract_features(y_vad, sr)
        
        # Transcribe the audio
        transcript = attempt_transcribe(local_path, app.config["GLADIA_API_KEY"])
        
        # Compute AI likelihood score
        ai_score = compute_likelihood(transcript, feats)
        
        # Update the recording with analysis results
        recording["transcript"] = transcript
        recording["ai_score"] = ai_score
        recording["analyzed"] = True
        
        flash("Recording analyzed successfully!", "success")
    except Exception as e:
        flash(f"Error analyzing recording: {e}", "error")
    
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    app.run(debug=True)
