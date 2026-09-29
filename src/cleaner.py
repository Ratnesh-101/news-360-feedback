import os
import sys
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from langdetect import detect
from deep_translator import GoogleTranslator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from dotenv import load_dotenv

# Ensure local imports work regardless of execution directory
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from scraper import get_all_articles

DATA_PATH = PROJECT_ROOT / "data" / "articles.csv"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    return _llm

prompt = ChatPromptTemplate.from_messages([
    ('system', '''You are a news sentiment analyst. Analyze the sentiment of the given news article.
Respond in JSON format only with these fields:
{{
    "sentiment": "positive/negative/neutral",
    "score": 0.0 to 1.0,
    "reason": "one line explanation",
    "ministry": "relevant government ministry or scheme if any, else null",
    "keywords": ["key", "terms"]
}}
'''),
    ('human', 'Title: {title}\nSummary: {summary}')
])

def _parse_llm_json(content: str) -> dict:
    """Safely extracts JSON from LLM responses, stripping code fences if present."""
    fallback = {
        'sentiment': 'neutral',
        'score': 0.5,
        'reason': 'could not parse',
        'ministry': None,
        'keywords': []
    }
    if not content or not isinstance(content, str):
        return fallback

    text = content.strip()
    # Strip markdown ```json ... ``` or ``` ... ``` code blocks
    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Normalize sentiment to lowercase
            if 'sentiment' in data and isinstance(data['sentiment'], str):
                data['sentiment'] = data['sentiment'].lower()
            return data
    except Exception:
        pass

    return fallback

def detect_language(text):
    if not text or not isinstance(text, str) or len(text.strip()) < 3:
        return 'unknown'
    try:
        return detect(text)
    except Exception:
        return 'unknown'

def translate_to_english(text, source_lang):
    if not text or not isinstance(text, str):
        return ''
    if source_lang == 'en':
        return text
    try:
        translated = GoogleTranslator(source='auto', target='english').translate(text[:4999])
        return translated or text
    except Exception:
        return text

def analyze_sentiment(title, summary):
    try:
        chain = prompt | get_llm()
        result = chain.invoke({'title': title, 'summary': summary})
        return _parse_llm_json(result.content)
    except Exception as e:
        print(f"Error in analyze_sentiment: {e}")
        return {
            'sentiment': 'neutral',
            'score': 0.5,
            'reason': f'analysis error: {str(e)[:50]}',
            'ministry': None,
            'keywords': []
        }

def clean_dataframe(df):
    df = df.copy()
    df['language'] = df['summary'].apply(detect_language)
    df['title'] = df.apply(lambda row: translate_to_english(row['title'], row['language']), axis=1)
    df['summary'] = df.apply(lambda row: translate_to_english(row['summary'], row['language']), axis=1)

    print('Running sentiment analysis...')
    rows = [row for _, row in df.iterrows()]
    with ThreadPoolExecutor(max_workers=10) as executor:
        sentiment_results = list(executor.map(
            lambda row: analyze_sentiment(row['title'], row['summary']), rows
        ))

    df['sentiment'] = [r.get('sentiment', 'neutral') for r in sentiment_results]
    df['sentiment_score'] = [r.get('score', 0.5) for r in sentiment_results]
    df['reason'] = [r.get('reason', '') for r in sentiment_results]
    df['ministry'] = [r.get('ministry') for r in sentiment_results]
    df['keywords'] = [r.get('keywords', []) for r in sentiment_results]

    return df

def save_articles():
    df = get_all_articles()
    if df.empty:
        print("No articles fetched from scraper.")
        return

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        existing_df = pd.read_csv(DATA_PATH)
        existing_links = set(existing_df['link'].dropna().tolist())
        new_df = df[~df['link'].isin(existing_links)]
        print(f'New articles to analyze: {len(new_df)}')

        if len(new_df) == 0:
            print('No new articles to save!')
            return

        new_df = clean_dataframe(new_df)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        print(f'Creating new dataset with {len(df)} articles...')
        final_df = clean_dataframe(df)

    final_df.to_csv(DATA_PATH, index=False)
    print(final_df[['title', 'sentiment', 'sentiment_score', 'ministry', 'category']].to_string())
    print(f'\nSaved {len(final_df)} articles to {DATA_PATH}')

if __name__ == '__main__':
    save_articles()