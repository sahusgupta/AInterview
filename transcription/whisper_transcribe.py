import time
import requests
import json

def upload(path, api_key):
    url = "https://api.gladia.io/v2/upload"
    headers = {
        "x-gladia-key": api_key
    }
    with open(path, 'rb') as f:
        files = {"audio": f}
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        data = dict(response.json())
        return data.get("audio_url", "No audio url found")
    
def request_transcript(url: str, api_key: str, diarization: bool = False, translation: bool = False):
    
    url = "https://api.gladia.io/v2/pre-recorded"
    headers = {
        "Content-Type": "application/json",
        "x-gladia-key": api_key
    }
    
    payload = {
        "audio_url": url,
        "diarization": diarization,
        "translation": translation
    }
    
    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()


def poll_for_result(transcript_id, api_key, interval, max_attempts):
    
    for _ in range(max_attempts):
        url = f"https://api.gladia.io/v2/pre-recorded/{transcript_id}"
        headers = {
            "x-gladia-key": api_key
        }
        
        resp = requests.get(url, headers=headers)
         
        if resp.status_code == 404:
            time.sleep(interval)
            continue
        
        resp.raise_for_status()
        data = resp.json()
        
        status = data.get("event", "")
        if status.lower() == "completed":
            return data
        elif status.lower() == "error":
            return f"Couldn't transcribe text. Response: {data}"
        time.sleep(interval)
        
    raise TimeoutError("Couldn't complete transcription within time limit.")

def attempt_transcribe(path, api_key):
    # Step 1: Upload local file to get an audio_url
    audio_url = upload(path, api_key)
    print("Uploaded. Gladia audio_url:", audio_url)

    # Step 2: Transcribe the file at audio_url
    job_response = request_transcript(
        audio_url=audio_url,
        api_key=api_key,
        diarization=False,  # set True if you want multi-speaker detection
        translation=False   # set True for translation
    )
    # job_response should contain e.g. { "id": "...", "result_url": "...", ...}
    transcription_id = job_response["id"]
    print("Transcription requested. ID:", transcription_id)
    print("Result URL (if you want direct GET):", job_response["result_url"])

    # Step 3: Poll for final results
    try:
        final_data = poll_for_result(transcription_id, api_key)
        # final_data should contain the completed transcription.
        # Typically final_data["transcription"] or final_data["prediction"] etc.
        transcript = final_data.get("prediction", "")
        print("Transcription complete!\n")
        print("Transcript:\n", transcript)
    except Exception as e:
        print("Transcription failed:", e)
