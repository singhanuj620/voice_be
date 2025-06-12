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
    userId: str = Form("userId"),
    reportId: str = Form("reportId"),
):
    try:
        userId = userId if userId else "user"
        reportId = reportId if reportId else "reportId"
        print(f"Received file: {file.filename}")
        text = convertAudioToText(file=file)
        # Generate a random correlation ID for user/session
        correlation_id = str(uuid.uuid4())
        response_text, mp3_bytes = get_chat_response(
            text,
            sender=userId,
            user_id=userId,  # changed from session_id
            report_id=reportId,
        )
        return StreamingResponse(iter([mp3_bytes]), media_type="audio/mpeg")
    except Exception as e:
        print(f"Error processing audio file: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/text-to-voice")
def text_to_voice(
    text: str = Form(...),
    userId: str = Form("userId"),
    reportId: str = Form("reportId"),
):
    try:
        print(f"Received typed text: {text}")
        # Generate a random correlation ID for user/session
        correlation_id = str(uuid.uuid4())
        response_text, mp3_bytes = get_chat_response(
            text,
            sender=userId,
            user_id=userId,  # changed from session_id
            report_id=reportId,
        )
        return StreamingResponse(iter([mp3_bytes]), media_type="audio/mpeg")
    except Exception as e:
        print(f"Error processing typed text: {e}")
        raise HTTPException(status_code=400, detail=str(e))
