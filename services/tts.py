from google.cloud import texttospeech
import os

def synthesize_text_to_mp3(text, output_filename="summary.mp3"):
    """
    Synthesizes speech from the input string of text using Google Cloud TTS, saves it as an MP3 file in the output directory,
    and returns the binary content of the audio file.
    Args:
        text (str): The text to synthesize.
        output_filename (str): The name of the output MP3 file (default: 'summary.mp3').
    Returns:
        bytes: The binary content of the generated MP3 audio file.
    """
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-IN", name="en-IN-Wavenet-A"
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
