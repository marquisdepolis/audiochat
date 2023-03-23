# %%
import os
import openai
import pyaudio
import pyttsx3
import wave
import logging
import warnings
import tempfile
import callgpt
from callgpt import Chatbot
import tkinter as tk
from tkinter import filedialog

logger = logging.getLogger(__name__)

# Catch warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


#directory = filedialog.askdirectory()
#os.chdir(directory)
os.environ["OPENAI_API_KEY"] = open_file('openai_api_key.txt')
openai.api_key = open_file('openai_api_key.txt')
openai_api_key = openai.api_key

# Configure audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5

# Initialize PyAudio
audio = pyaudio.PyAudio()


def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Done recording.")
    stream.stop_stream()
    stream.close()
    return b''.join(frames)


def transcribe_audio(audio_data):
    # Save the audio bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_data)

        # Transcribe the temporary file
        with open(temp_file.name, 'rb') as file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                file,
                sample_rate=RATE,
                encoding="LINEAR16"
                # language="en-US"
            )

    # Delete the temporary file
    os.unlink(temp_file.name)

    if transcript["text"]:
        return transcript["text"]
    else:
        return None


def get_gpt3_response(text):
    if not text:
        return None

    chatbot = Chatbot()
    prompt = f"This was said by my five year old son, and any reply has to be age appropriate: '{text}'"
    response = chatbot.creative_prompt(prompt)
    return response


def synthesize_speech_with_whisper(text):
    if not text:
        return None

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def play_audio(audio_data):
    if not audio_data:
        return

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, output=True)
    print("Playing audio...")
    stream.write(audio_data)
    stream.stop_stream()
    stream.close()
    print("Done playing audio.")


while True:
    try:
        audio_data = record_audio()
        transcription = transcribe_audio(audio_data)
        print(f"Transcription: {transcription}")
        gpt3_response = get_gpt3_response(transcription)
        # print(f"GPT-3 Response: {gpt3_response}")
        synthesized_speech = synthesize_speech_with_whisper(gpt3_response)
        play_audio(synthesized_speech)
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")

# Terminate PyAudio
audio.terminate()
