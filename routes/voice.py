from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Form
from services.llm_service import get_chat_response
from fastapi.responses import StreamingResponse
import os
import uuid
from utils.convertAudioToText import convertAudioToText

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
    stt_language_code: str = Form("en-US"),  # NEW: Accept STT language code
    userId: str = Form("userId"),
    reportId: str = Form("reportId"),
):
    try:
        userId = userId if userId else "user"
        reportId = reportId if reportId else "reportId"
        print(f"Received file: {file.filename}")
        text = convertAudioToText(
            file=file,
            stt_language_code=stt_language_code,  # Use user-selected STT language
        )
        # Generate a random correlation ID for user/session
        correlation_id = str(uuid.uuid4())
        response_text, mp3_bytes = get_chat_response(
            text,
            sender=userId,
            user_id=userId,  # changed from session_id
            accent_code=accent_code,
            voice_name=voice_name,
            report_id=reportId,
        )
        return StreamingResponse(iter([mp3_bytes]), media_type="audio/mpeg")
    except Exception as e:
        print(f"Error processing audio file: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/text-to-voice")
def text_to_voice(
    text: str = Form(...),
    accent_code: str = Form("en-IN"),
    voice_name: str = Form("en-IN-Wavenet-A"),
):
    try:
        print(f"Received typed text: {text}")
        # Generate a random correlation ID for user/session
        correlation_id = str(uuid.uuid4())
        response_text, mp3_bytes = get_chat_response(
            text,
            sender="user",
            user_id="user",  # changed from session_id
            accent_code=accent_code,
            voice_name=voice_name,
        )
        return StreamingResponse(iter([mp3_bytes]), media_type="audio/mpeg")
    except Exception as e:
        print(f"Error processing typed text: {e}")
        raise HTTPException(status_code=400, detail=str(e))
