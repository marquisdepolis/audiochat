import time
import pyaudio
import json
import os
import requests
import whisper
import webrtcvad
import torch
from TTS.api import TTS
import IPython.display as ipd
import threading
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='whisper')

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
KEY_PHRASE = "Hey Loopy"
SILENCE_TIMEOUT_MS = 5000  # Silence duration to stop recording
MAX_RECORD_DURATION_MS = 12000  # Maximum record duration
VAD = webrtcvad.Vad(1)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS with the target model name
tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to(device)

# Load Whisper model
whisper_model = whisper.load_model("base")

# Global variables for conversation history
conversation_history = []

def ask_llm(prompt):
    """
    Function to interact with an LLM and manage conversation history.
    """
    global conversation_history
    conversation_history.append(f"User: {prompt}")
    print(f"Conversation history is: {conversation_history}")
    r = requests.post('http://0.0.0.0:11434/api/generate',
                      json={
                          'model': "mistral",
                          'prompt': prompt,
                      },
                      stream=False)
    full_response = ""    
    for line in r.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            json_line = json.loads(decoded_line)
            full_response += json_line.get("response", "")
            if json_line.get("done"):
                break

    print(full_response)
    conversation_history.append(f"LLM: {full_response}")
    # Truncate history if it exceeds a certain size
    max_history = 5  # for example, keep the last 5 exchanges
    if len(conversation_history) > max_history * 2:
        conversation_history = conversation_history[-max_history * 2:]
    return full_response

def is_speech(chunk):
    return VAD.is_speech(chunk, RATE)

def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    frames, start_time, last_speech_time = [], time.time(), time.time()

    try:
        while True:
            if stream.is_stopped():
                stream.start_stream()

            chunk = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)
            frames.append(chunk)

            if is_speech(chunk.tobytes()):
                last_speech_time = time.time()

            current_duration = (time.time() - start_time) * 1000
            silence_duration = (time.time() - last_speech_time) * 1000

            if silence_duration > SILENCE_TIMEOUT_MS or current_duration > MAX_RECORD_DURATION_MS:
                break
            # stream.stop_stream()
    except Exception as e:
        print(f"Error with audio stream: {e}")
    # finally:
    #     if stream.is_active():
    #         stream.stop_stream()
    #     stream.close()

    return np.concatenate(frames).tobytes()

def transcribe_audio(audio_data):
    """
    Transcribe audio data using Whisper.
    """
    # Convert from 16-bit integers to floating-point
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    # Normalize the audio to the range of -1.0 to 1.0
    audio_normalized = audio_np / np.iinfo(np.int16).max
    
    return whisper_model.transcribe(audio_normalized)

def listen_and_transcribe():
    """
    Continuously listen for the key phrase and transcribe the speech.
    """
    try:
        while True:
            print("Listening for key phrase...")
            audio_data = record_audio()
            print("Transcribing audio...")
            transcription = transcribe_audio(audio_data)
            print(f"Transcribed question: {transcription}")
            if KEY_PHRASE.lower() in transcription["text"].lower():
                question = transcription["text"]
                print("Question:", question)
                prompt = f"Answer the question as asked, precisely and succinctly. This is a conversation, not an essay. Question: {question}"
                response = ask_llm(prompt)
                print("Response:", response)
                talk_audio(response)
    except KeyboardInterrupt:
        print("Stopping...")

def talk_audio(talk):
    tts.tts_to_file(text=talk)

if __name__ == "__main__":
    thread = threading.Thread(target=listen_and_transcribe)
    thread.start()


