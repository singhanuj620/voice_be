import os
from fastapi import HTTPException
import tempfile
from pydub import AudioSegment
from google.cloud import speech


def convertAudioToText(file):
    filename = file.filename.lower()
    # Accept .wav and .webm files
    if not (filename.endswith(".wav") or filename.endswith(".webm")):
        raise HTTPException(
            status_code=400, detail="Only .wav and .webm files are supported."
        )
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(filename)[1]
    ) as temp_audio:
        temp_audio.write(file.file.read())
        temp_audio.flush()
        temp_audio_path = temp_audio.name
    # If .webm, convert to .wav using pydub
    if filename.endswith(".webm"):
        temp_wav_path = temp_audio_path + ".wav"
        audio = AudioSegment.from_file(temp_audio_path, format="webm")
        audio = audio.set_sample_width(2)  # 16-bit PCM
        audio.export(temp_wav_path, format="wav")
        os.remove(temp_audio_path)
    else:
        temp_wav_path = temp_audio_path
        # Ensure 16-bit PCM for wav files as well
        audio = AudioSegment.from_file(temp_wav_path)
        if audio.sample_width != 2:
            audio = audio.set_sample_width(2)
            audio.export(temp_wav_path, format="wav")
    # Detect sample rate using pydub
    audio_segment = AudioSegment.from_file(temp_wav_path)
    detected_sample_rate = audio_segment.frame_rate
    print(f"Detected sample rate: {detected_sample_rate}")
    # Use Google Cloud Speech-to-Text on the .wav file
    print(f"Processing audio file: {temp_wav_path}")
    client = speech.SpeechClient()
    with open(temp_wav_path, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=detected_sample_rate,
        language_code="en-US",  # Always use default; downstream will detect actual language
    )
    response = client.recognize(config=config, audio=audio)
    if not response.results:
        raise HTTPException(status_code=400, detail="No speech recognized.")
    text = response.results[0].alternatives[0].transcript
    os.remove(temp_wav_path)
    print(f"Recognized text from input: {text}")
    return text
