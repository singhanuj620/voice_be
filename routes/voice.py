from fastapi import APIRouter, UploadFile, File, HTTPException, Body
import speech_recognition as sr
import tempfile
import os
from pydub import AudioSegment
from services.llm_service import get_chat_response
from fastapi.responses import StreamingResponse
from services.enchanceUserPrompt import get_best_related_prompt

router = APIRouter()


@router.post("/voice-to-text")
def voice_to_text(file: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    try:
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
            audio.export(temp_wav_path, format="wav")
            os.remove(temp_audio_path)
        else:
            temp_wav_path = temp_audio_path
        # Use speech_recognition on the .wav file
        with sr.AudioFile(temp_wav_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        text = get_best_related_prompt(text)  # Enhance user prompt
        print(f"Recognized text: {text}")
        # Clean up temp file
        os.remove(temp_wav_path)
        response_text, mp3_bytes = get_chat_response(text)
        return StreamingResponse(iter([mp3_bytes]), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
