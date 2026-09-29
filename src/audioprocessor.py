import os
import sys
import re
import json
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
from deep_translator import GoogleTranslator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Ensure local imports and paths work
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

# Optional whisper & torch import guards
try:
    import whisper
    import torch
except ImportError:
    whisper = None
    torch = None

# ── constants ────────────────────────────────────────────────────────────────
SUPPORTED_AUDIO = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4"]
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
CSV_PATH = PROJECT_ROOT / "data" / "articles.csv"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── LLM setup (lazily initialized) ──────────────────────────────────────────
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    return _llm

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior Indian political news analyst working for the Government of India.
Analyze the sentiment of the given news article from an Indian governance perspective.

Guidelines:
- "positive": good news for governance, development, policy success, welfare schemes
- "negative": scandals, removals, failures, protests, criticism of government/policy
- "neutral": factual reporting, transfers, routine announcements with no clear positive/negative impact

Respond in JSON format only with these fields:
{{
    "sentiment": "positive/negative/neutral",
    "score": 0.0 to 1.0,
    "reason": "one line explanation",
    "ministry": "relevant Indian government ministry or scheme if any, else null",
    "keywords": ["key", "terms"]
}}
"""),
    ("human", "Title: {title}\nSummary: {summary}")
])

# ── Whisper singleton ─────────────────────────────────────────────────────────
_whisper_model = None

def load_whisper_model():
    global _whisper_model
    if whisper is None:
        raise ImportError("openai-whisper is not installed. Please install it using `pip install openai-whisper`.")

    if _whisper_model is None:
        device = "cpu"
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            device = "cuda"
        print(f"[Whisper] Loading {MODEL_NAME} on {device}...")
        _whisper_model = whisper.load_model(MODEL_NAME, device=device)
        print("[Whisper] Ready.")
    return _whisper_model

# ── core functions ────────────────────────────────────────────────────────────
def transcribe_audio(file_bytes: bytes, filename: str) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO:
        raise ValueError(f"Unsupported format '{suffix}'. Allowed: {SUPPORTED_AUDIO}")

    # Use secure NamedTemporaryFile
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        model = load_whisper_model()
        use_fp16 = bool(torch and torch.cuda.is_available())

        result = model.transcribe(
            tmp_path,
            task="transcribe",
            verbose=False,
            fp16=use_fp16,
        )

        original_text = result["text"].strip()
        detected_lang = result.get("language", "unknown")

        if detected_lang != "en" and original_text:
            try:
                translated_text = GoogleTranslator(
                    source="auto", target="english"
                ).translate(original_text[:4999])
            except Exception:
                translated_text = original_text
        else:
            translated_text = original_text

        title = f"Audio — {Path(filename).stem}"
        sentiment_result = _analyze_sentiment(title, translated_text)

        return {
            "title":           title,
            "summary":         translated_text,
            "link":            f"audio://{filename}",
            "published":       datetime.now().isoformat(),
            "source":          "audio_upload",
            "category":        "audio",
            "language":        detected_lang,
            "sentiment":       sentiment_result.get("sentiment", "neutral").lower(),
            "sentiment_score": sentiment_result.get("score", 0.5),
            "reason":          sentiment_result.get("reason", ""),
            "ministry":        sentiment_result.get("ministry"),
            "keywords":        sentiment_result.get("keywords") or [],
            "original_text":   original_text,
            "translated_text": translated_text,
        }

    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

def append_to_csv(record: dict, csv_path: str = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df_existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df_updated = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
    df_updated.to_csv(path, index=False)
    return df_updated

# ── internal ──────────────────────────────────────────────────────────────────
def _analyze_sentiment(title: str, summary: str) -> dict:
    fallback = {
        "sentiment": "neutral",
        "score": 0.5,
        "reason": "could not parse",
        "ministry": None,
        "keywords": [],
    }
    try:
        chain = prompt | get_llm()
        result = chain.invoke({"title": title, "summary": summary})
        text = result.content.strip()
        # Strip markdown fences if present
        fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if fence_match:
            text = fence_match.group(1).strip()
        data = json.loads(text)
        if isinstance(data, dict):
            if 'sentiment' in data and isinstance(data['sentiment'], str):
                data['sentiment'] = data['sentiment'].lower()
            return data
        return fallback
    except Exception as e:
        print(f"Error in _analyze_sentiment: {e}")
        return fallback