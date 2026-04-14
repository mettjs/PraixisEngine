import os
from openai import OpenAI

_ai_api_url = os.getenv("AI_API_URL", "http://localhost:8081")
_ai_api_key = os.getenv("AI_API_KEY", "")

def get_ai_client() -> OpenAI:
    """Initializes and returns the connection."""
    return OpenAI(
        base_url=_ai_api_url, 
        api_key=_ai_api_key
    )