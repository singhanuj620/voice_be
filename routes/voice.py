from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Form
import tempfile
from pydub import AudioSegment
from services.llm_service import get_chat_response
from fastapi.responses import StreamingResponse
from google.cloud import speech
import os
import uuid

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
#     r"D:\Projects\GenAI\voice\secrets\google_tts_key.json"
# )

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/google_tts_key.json"

router = APIRouter()


@router.post("/voice-to-text")
def voice_to_text(
    file: UploadFile = File(...),
    accent_code: str = Form("en-IN"),
    voice_name: str = Form("en-IN-Wavenet-A"),
    stt_language_code: str = Form("en-US")  # NEW: Accept STT language code
):
    try:
        print(f"Received file: {file.filename}")
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
            language_code=stt_language_code,  # Use user-selected STT language
        )
        response = client.recognize(config=config, audio=audio)
        if not response.results:
            raise HTTPException(status_code=400, detail="No speech recognized.")
        text = response.results[0].alternatives[0].transcript
        print(f"Recognized text: {text}")
        # Clean up temp file
        os.remove(temp_wav_path)
        # Generate a random correlation ID for user/session
        correlation_id = str(uuid.uuid4())
        response_text, mp3_bytes = get_chat_response(
            text, sender="user", session_id=correlation_id, accent_code=accent_code, voice_name=voice_name
        )
        return StreamingResponse(iter([mp3_bytes]), media_type="audio/mpeg")
    except Exception as e:
        print(f"Error processing audio file: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/text-to-voice")
def text_to_voice(
    text: str = Form(...),
    accent_code: str = Form("en-IN"),
    voice_name: str = Form("en-IN-Wavenet-A")
):
    try:
        print(f"Received typed text: {text}")
        # Generate a random correlation ID for user/session
        correlation_id = str(uuid.uuid4())
        response_text, mp3_bytes = get_chat_response(
            text, sender="user", session_id=correlation_id, accent_code=accent_code, voice_name=voice_name
        )
        return StreamingResponse(iter([mp3_bytes]), media_type="audio/mpeg")
    except Exception as e:
        print(f"Error processing typed text: {e}")
        raise HTTPException(status_code=400, detail=str(e))
