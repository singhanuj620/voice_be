from google.cloud import texttospeech
from langdetect import detect
import os

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
#     r"D:\Projects\GenAI\voice\secrets\google_tts_key.json"
# )

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/secrets/google_tts_key.json"


def synthesize_text_to_mp3(
    text,
    output_filename="summary.mp3",
    accent_code="en-IN",
    voice_name="en-IN-Female",
):
    """
    Synthesizes speech from the input string of text using Google Cloud TTS, saves it as an MP3 file in the output directory,
    and returns the binary content of the audio file.
    Args:
        text (str): The text to synthesize.
        output_filename (str): The name of the output MP3 file (default: 'summary.mp3').
        accent_code (str): The language code for the accent (default: 'en-IN').
        voice_name (str): 'en-IN-Male', 'en-IN-Female', 'en-US-Male', or 'en-US-Female'.
    Returns:
        bytes: The binary content of the generated MP3 audio file.
    """
    # Detect language if not explicitly set to Hindi
    try:
        detected_lang = detect(text)
    except Exception:
        detected_lang = "en"
    # If Hindi detected, override accent/voice to Hindi
    if detected_lang == "hi":
        accent_code = "hi-IN"
        # Map to Google TTS Hindi voices (A: Male, C: Female)
        if voice_name.endswith("Female"):
            g_voice = "hi-IN-Wavenet-C"
        else:
            g_voice = "hi-IN-Wavenet-A"
        g_accent = accent_code
    else:
        # Map user-friendly voice_name to Google TTS voice
        voice_map = {
            ("en-IN", "en-IN-Male"): ("en-IN", "en-IN-Wavenet-B"),
            ("en-IN", "en-IN-Female"): ("en-IN", "en-IN-Wavenet-A"),
            ("en-US", "en-US-Male"): ("en-US", "en-US-Wavenet-D"),
            ("en-US", "en-US-Female"): ("en-US", "en-US-Wavenet-F"),
        }
        g_accent, g_voice = voice_map.get((accent_code, voice_name), (accent_code, "en-IN-Wavenet-A"))
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=g_accent, name=g_voice
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    tts_response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "wb") as out:
        out.write(tts_response.audio_content)
    print(f"Audio saved as {output_path}")
    with open(output_path, "rb") as audio_file:
        audio_content = audio_file.read()
    return audio_content
