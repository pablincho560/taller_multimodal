"""Utilidades compartidas: cliente de Groq y codificacion de imagenes en base64."""
import base64
import json
import mimetypes
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")

BASE = Path(__file__).parent
IMG_DIR = BASE / "imagenes"
AUDIO_DIR = BASE / "audios"
OUT_DIR = BASE / "resultados"
OUT_DIR.mkdir(exist_ok=True)


def get_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("Falta GROQ_API_KEY. Copia .env.example a .env y completa la clave.")
    return Groq(api_key=key)


def image_to_data_url(path: Path, mime_override: str | None = None) -> str:
    """Convierte una imagen local en una data URL base64 (como en multimodal.py)."""
    mime = mime_override or mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def image_part(path: Path, mime_override: str | None = None) -> dict:
    return {"type": "image_url", "image_url": {"url": image_to_data_url(path, mime_override)}}


def extract_json(text: str) -> dict:
    """Extrae un objeto JSON de la respuesta, quitando cercas ```json si las hay."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("La respuesta no contiene un objeto JSON.")
    return json.loads(text[start : end + 1])
