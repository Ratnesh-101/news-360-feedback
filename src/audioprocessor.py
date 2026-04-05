import whisper
import torch
import os
import uuid
import json
from pathlib import Path
from deep_translator import GoogleTranslator
import pandas as pd
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ── constants ────────────────────────────────────────────────────────────────
SUPPORTED_AUDIO = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4"]
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "articles.csv")

# ── LLM setup ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

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


def load_whisper_model() -> whisper.Whisper:
    global _whisper_model
    if _whisper_model is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Whisper] Loading {MODEL_NAME} on {device}...")
        _whisper_model = whisper.load_model(MODEL_NAME, device=device)
        print("[Whisper] Ready.")
    return _whisper_model


# ── core functions ────────────────────────────────────────────────────────────
def transcribe_audio(file_bytes: bytes, filename: str) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO:
        raise ValueError(f"Unsupported format '{suffix}'. Allowed: {SUPPORTED_AUDIO}")

    tmp_path = os.path.join(os.path.expanduser("~"), f"whisper_tmp_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as tmp:
        tmp.write(file_bytes)

    try:
        model = load_whisper_model()

        result = model.transcribe(
            tmp_path,
            task="transcribe",
            verbose=False,
            fp16=torch.cuda.is_available(),
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
            "sentiment":       sentiment_result["sentiment"],
            "sentiment_score": sentiment_result["score"],
            "reason":          sentiment_result["reason"],
            "ministry":        sentiment_result["ministry"],
            "keywords":        sentiment_result["keywords"],
            "original_text":   original_text,
            "translated_text": translated_text,
        }

    finally:
        os.unlink(tmp_path)


def append_to_csv(record: dict, csv_path: str = CSV_PATH) -> pd.DataFrame:
    path = Path(csv_path)
    df_existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df_updated = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
    df_updated.to_csv(path, index=False)
    return df_updated


# ── internal ──────────────────────────────────────────────────────────────────
def _analyze_sentiment(title: str, summary: str) -> dict:
    chain = prompt | llm
    result = chain.invoke({"title": title, "summary": summary})
    try:
        return json.loads(result.content)
    except Exception:
        return {
            "sentiment": "neutral",
            "score": 0.5,
            "reason": "could not parse",
            "ministry": None,
            "keywords": [],
        }