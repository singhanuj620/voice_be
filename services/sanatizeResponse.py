import re


def sanitize_response(response: str) -> str:
    """
    Remove or replace special characters from the response to make it suitable for audio playback.
    """
    return re.sub(r"[\*\^\~\`\|\<\>\[\]\{\}]", "", response)


__all__ = ["sanitize_response"]
