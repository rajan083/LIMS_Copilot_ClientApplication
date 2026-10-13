import os, time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client= genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
model= "gemini-3.1-flash-lite/;,lmkm"

def transcribe_online(filename):
    with open(filename, 'rb') as f:
        audio=f.read()
        start= time.perf_counter()
        response= client.models.generate_content(
            model=model, 
            contents=[
                "Transcribe this audio exactly. Return only the transcript.",
                types.Part.from_bytes(data= audio, mime_type="audio/wav")
                ]
            )
        end= time.perf_counter()
    return (f"Transcription: {response.text}",
            f"Time Taken: {end-start:.2f}s"
            )

def decode_audio(filepath):
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
    
    sr, data= wavfile.read(filepath)
    if sr != 16000:
        g= gcd(16000, sr)
        audio= resample_poly(data, 16000//g,sr//g).astype(np.float32)/32768
    return audio

def transcribe_local(audio):
    import whisper
    model= whisper.load_model('small')
    start= time.perf_counter()
    result= model.transcribe(audio, fp16=False)
    end= time.perf_counter()
    return f'response: {result['text']} \nTime taken: {end-start:.2f}s'

def stt(path, retries=3):
    for attempt in range(retries):
        try:
            return transcribe_online(path)
        except Exception as e:
            if attempt < (retries-1):
                wait= 2*(attempt+1)
                time.sleep(wait)
            print(f'Error: {e}')
    print("Falling back to the local transcription")
    return transcribe_local(decode_audio(path))

path= r"C:\Users\ACER\Desktop\Projects\FastAPI\speech.wav"
print(stt(path))