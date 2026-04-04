import whisper
import torch
import tempfile
import os
from pathlib import Path
from langdetect import detect
from deep_translator import GoogleTranslator
import pandas as pd
from datetime import datetime

# Reuse your existing cleaner logic
from cleaner import analyze_sentiment  # adjust import if needed


SUPPORTED_AUDIO = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4"]
MODEL_NAME = "large-v3"

_model = None  # lazy-load singleton


def load_whisper_model(model_name: str = MODEL_NAME) -> whisper.Whisper:
    """Lazy-load Whisper model onto GPU if available."""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Whisper] Loading {model_name} on {device}...")
        _model = whisper.load_model(model_name, device=device)
        print("[Whisper] Model ready.")
    return _model


def transcribe_audio(file_bytes: bytes, filename: str) -> dict:
    """
    Transcribe audio bytes → dict with transcript, detected language,
    English translation, and sentiment.

    Returns:
        {
            "title": str,
            "original_text": str,
            "translated_text": str,
            "language": str,
            "sentiment": str,   # Positive / Negative / Neutral
            "source": "audio_upload",
            "published": str,   # ISO datetime
        }
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO:
        raise ValueError(f"Unsupported format: {suffix}. Use {SUPPORTED_AUDIO}")

    # Write to temp file (Whisper needs a file path)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        model = load_whisper_model()

        # Transcribe — Whisper auto-detects language
        result = model.transcribe(
            tmp_path,
            task="transcribe",       # keep original language
            verbose=False,
            fp16=torch.cuda.is_available(),
        )

        original_text = result["text"].strip()
        detected_lang = result.get("language", "unknown")

        # Translate to English if not already English
        if detected_lang != "en" and len(original_text) > 0:
            try:
                translated_text = GoogleTranslator(
                    source="auto", target="en"
                ).translate(original_text[:4999])  # API limit guard
            except Exception:
                translated_text = original_text
        else:
            translated_text = original_text

        # Sentiment via your existing GPT pipeline
        sentiment = analyze_sentiment(translated_text)

        return {
            "title": f"Audio Upload — {Path(filename).stem}",
            "original_text": original_text,
            "translated_text": translated_text,
            "language": detected_lang,
            "sentiment": sentiment,
            "source": "audio_upload",
            "published": datetime.now().isoformat(),
        }

    finally:
        os.unlink(tmp_path)  # always clean up temp file


def append_to_articles_csv(record: dict, csv_path: str = "data/articles.csv") -> pd.DataFrame:
    """Append a new audio-derived article to the existing CSV."""
    df_existing = pd.read_csv(csv_path) if Path(csv_path).exists() else pd.DataFrame()

    new_row = pd.DataFrame([{
        "title":      record["title"],
        "text":       record["translated_text"],
        "language":   record["language"],
        "sentiment":  record["sentiment"],
        "source":     record["source"],
        "published":  record["published"],
        "original_text": record["original_text"],
    }])

    df_updated = pd.concat([df_existing, new_row], ignore_index=True)
    df_updated.to_csv(csv_path, index=False)
    return df_updated