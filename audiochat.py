# %%
import traceback
import sounddevice as sd
from gtts import gTTS
from pydub import AudioSegment
import numpy as np
import time
import openai
import tempfile
import os
import webrtcvad
import pyaudio
from callgpt import Ask
import warnings
import logging
import wave

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


os.environ["OPENAI_API_KEY"] = open_file('openai_api_key.txt')
openai.api_key = open_file('openai_api_key.txt')
openai_api_key = openai.api_key


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 320
RECORD_SECONDS = 5
# Initialize PyAudio
# audio = pyaudio.PyAudio()


def record_audio():
    vad = webrtcvad.Vad()
    vad.set_mode(2)

    def callback(in_data, *args):
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        if vad.is_speech(audio_data.tobytes(), RATE, CHUNK):
            nonlocal num_silent_chunks
            nonlocal consecutive_silent_chunks_limit
            nonlocal min_audio_length_chunks
            nonlocal audio_buffer

            num_silent_chunks = 0
            audio_buffer.extend(audio_data.tobytes())
        else:
            num_silent_chunks += 1

            if num_silent_chunks > consecutive_silent_chunks_limit and len(audio_buffer) > min_audio_length_chunks:
                return (None, pyaudio.paComplete)

        return (None, pyaudio.paContinue)

    print("Recording...")

    num_silent_chunks = 0
    consecutive_silent_chunks_limit = RATE // CHUNK * \
        1  # Stop recording after 1 second of silence
    # Require at least 0.5 seconds of audio
    min_audio_length_chunks = RATE // CHUNK * 0.5

    audio_buffer = bytearray()

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Done recording.")

    return audio_buffer


def transcribe_audio(audio_data):
    # Save the audio bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            # Set the sample width to 2 bytes, as we're using FORMAT = pyaudio.paInt16 (16-bit)
            wf.setsampwidth(2)
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

    chatbot = Ask()
    prompt = f"This was said by my six year old son. Note he loves science and animals and history and archeology and flags and continents and countries. Don't make any reference to this, but any reply to his question later has to be age appropriate: '{text}'"
    response = chatbot.gpt_creative(prompt)
    return response


def play_audio(file_path):
    if not file_path:
        print("No file path!")
        return

    print("Playing audio...")

    try:
        audio_segment = AudioSegment.from_file(file_path)
        audio_data = np.array(
            audio_segment.get_array_of_samples(), dtype=np.int16)
        samplerate = audio_segment.frame_rate
        sd.play(audio_data, samplerate=samplerate)
        print("Audio playing.")
        sd.wait()
        print("Audio playback finished.")
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())

    print("Done playing audio.")


def synthesize_speech_with_gtts(text, file_name='output_audio.wav'):
    if not text:
        return None

    tts = gTTS(text, lang='en')

    print("CHECKPOINT 1: Audio generated")
    tts.save(file_name)
    print(f"CHECKPOINT 2: Saved synthesized speech to {file_name}")

    return file_name


while True:
    try:
        audio_data = record_audio()
        transcription = transcribe_audio(audio_data)
        print(f"Transcription: {transcription}")
        gpt3_response = get_gpt3_response(transcription)
        synthesized_speech = synthesize_speech_with_gtts(gpt3_response)
        play_audio(synthesized_speech)
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
