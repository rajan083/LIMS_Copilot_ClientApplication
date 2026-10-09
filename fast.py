import os, time
import requests
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io import wavfile

load_dotenv()
client= genai.Client(api_key= os.environ['GEMINI_API_KEY'])
URL= os.environ.get("URL", "http://127.0.0.1:8000").rstrip("/")
HEADERS = {
    "X-API-Key": os.environ.get("LIMS_API_KEY", ""),
    "X-User-Id": os.environ.get("LIMS_API_USER", "user-1"),
}

EXPORTS_DIR= "exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)

conv_id = requests.post(f"{URL}/v1/conversations", headers=HEADERS).json()["conversation_id"]
CONV= f"{URL}/v1/conversations/{conv_id}"

model= "gemini-3.6-flash"

def stt(filename, retries=3):
    with open(filename, "rb") as f:
        audio= f.read()
    for attempt in range(retries):
        response= None
        try:
            response= client.models.generate_content(
                model=model, 
                contents=[
                    "Transcribe this audio exactly. Return only the transcript.",
                    types.Part.from_bytes(data=audio, mime_type="audio/wav")]
                )
            break
        
        except Exception as e:
            if attempt == retries-1:
                raise
            wait= 2*(attempt+1)
            print(f"Retrying in {wait}")
            time.sleep(wait)
    return {"text": response.text or ""}


def save_artifacts(artifacts):
    for a in artifacts:
        r=  requests.get(URL + a["download_url"], headers=HEADERS)
        if r.status_code == 200:
            path= os.path.join(EXPORTS_DIR, a["filename"])
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"saved {a['type']} {path}")
        else:
            print(f"[could not download {a['filename']}: {r.status_code}]")
    if artifacts:
        print()
        
def confirm_email(p):
    names = ", ".join(p.get("files") or [p["file"]])
    ques = f"Send {names} to {p['name'] or ''} <{p['to']}>? (y/n): "
    while True:
        ans= input(ques).strip().lower()
        if ans in ('y', 'yes', 'send'):
            r= requests.post(f"{CONV}/email/confirm", headers= HEADERS)
            data=r.json()
            if r.status_code ==200:
                print(f"[email] {data['message']}\n")
            else:
                print(f"[email error {r.status_code}] {data.get('detail', data)}\n")
            return
        if ans in ('n', 'no', 'cancel'):
            requests.post(f"{CONV}/email/cancel", headers=HEADERS)
            print(f"[email] cancelled\n")
            return
        print("Please type y or n.")


def record(path="voice.wav", rate=16000):
    chunks=[]
    stream= sd.InputStream(samplerate=rate, channels=1, dtype="int16",
                           callback=lambda d,f,t,s: chunks.append(d.copy()))
    input("Press Enter to start recording...")
    with stream:
        input("Recording... press Enter to stop.")
    if not chunks:
        return None
    wavfile.write(path, rate, np.concatenate(chunks))
    return path
    
def chat():
    while True:
        prompt= input("You: (type, or /v for voice): ")
        if prompt.lower().strip() in ("exit","quit"):
            break
        if prompt.strip().lower() == "/v":
            wav=record()
            if not wav:
                print("[no audio captured]\n")
                continue
            try:
                prompt= stt(wav)["text"].strip()
            except Exception as e:
                print(["transcription failed: {e}"])
                continue
            if not prompt:
                print("[Couldn't hear anything]\n")
                continue
            print(f"You (voice): {prompt}")
            ans= input("Send this? (y/n/edit): ").strip().lower()
            if ans == "n":
                continue
            if ans == "edit":
                prompt= input("Edit: ").strip() or prompt
                
        r= requests.post(f"{URL}/v1/conversations/{conv_id}/messages", headers=HEADERS, json={"message": prompt})
        data=r.json()
        print(f"Bot: {data["reply"]}")
        save_artifacts(data["artifacts"])
        if data["pending_email"]:
            confirm_email(data['pending_email'])
        print()
        
chat()