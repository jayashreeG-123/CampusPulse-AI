import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Set SECRET_KEY in the private .env file before using real data.
    SECRET_KEY = os.getenv("SECRET_KEY", "development-only-change-me")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "")
    CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    CLOUD_MODEL = os.getenv("CLOUD_MODEL", "gpt-4o-mini")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "45"))
