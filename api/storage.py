import os
import sys
import re
import ast
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "articles.csv"

def slugify(text: Optional[str]) -> Optional[str]:
    if not text or not str(text).strip() or str(text).strip().lower() in ("none", "nan"):
        return None
    cleaned = re.sub(r'[^a-zA-Z0-9\s-]', '', str(text)).strip().lower()
    return re.sub(r'[\s-]+', '-', cleaned)

def sha1_id(link: Optional[str], title: Optional[str] = "") -> str:
    seed = (link or "") + (title or "")
    if not seed:
        seed = "article"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

def parse_keywords(raw: Any) -> List[str]:
    if not raw or pd.isna(raw):
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    raw_str = str(raw).strip()
    if raw_str.startswith("[") and raw_str.endswith("]"):
        try:
            parsed = ast.literal_eval(raw_str)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    # Fallback comma-split
    return [k.strip().strip("'\"") for k in raw_str.split(",") if k.strip()]

class ArticleStore:
    def __init__(self, csv_path: Path = DATA_PATH):
        self.csv_path = csv_path
        self._articles: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self):
        if not self.csv_path.exists():
            self._articles = []
            self._by_id = {}
            return

        df = pd.read_csv(self.csv_path)
        articles = []
        for _, row in df.iterrows():
            link = str(row.get("link", "")) if pd.notna(row.get("link")) else ""
            title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
            art_id = sha1_id(link, title)

            ministry = str(row.get("ministry", "")).strip() if pd.notna(row.get("ministry")) else None
            if ministry in ("None", "nan", ""):
                ministry = None
            ministry_slug = slugify(ministry)

            sentiment = str(row.get("sentiment", "neutral")).strip().lower()
            if sentiment not in ("positive", "negative", "neutral"):
                sentiment = "neutral"

            try:
                score = float(row.get("sentiment_score", 0.5))
            except (ValueError, TypeError):
                score = 0.5

            published_raw = str(row.get("published", "")).strip() if pd.notna(row.get("published")) else ""

            item = {
                "id": art_id,
                "title": title,
                "summary": str(row.get("summary", "")).strip() if pd.notna(row.get("summary")) else "",
                "link": link,
                "published": published_raw or datetime.now(timezone.utc).isoformat(),
                "category": str(row.get("category", "general")).strip() if pd.notna(row.get("category")) else "general",
                "language": str(row.get("language", "en")).strip().lower() if pd.notna(row.get("language")) else "en",
                "sentiment": sentiment,
                "sentimentScore": round(score, 2),
                "reason": str(row.get("reason", "")).strip() if pd.notna(row.get("reason")) else "",
                "ministry": ministry,
                "ministrySlug": ministry_slug,
                "keywords": parse_keywords(row.get("keywords")),
                "source": str(row.get("source", "")).strip() if pd.notna(row.get("source")) else None,
                "originalText": str(row.get("original_text", "")).strip() if pd.notna(row.get("original_text")) else None,
                "translatedText": str(row.get("translated_text", "")).strip() if pd.notna(row.get("translated_text")) else None,
            }
            articles.append(item)

        # Sort by published desc
        articles.sort(key=lambda x: x["published"], reverse=True)
        self._articles = articles
        self._by_id = {a["id"]: a for a in articles}

    def append_article(self, record: Dict[str, Any]):
        """Append an analyzed record and refresh in-memory store."""
        df_row = {
            "title": record.get("title", ""),
            "summary": record.get("summary", ""),
            "link": record.get("link", ""),
            "published": record.get("published", datetime.now(timezone.utc).isoformat()),
            "category": record.get("category", "audio"),
            "language": record.get("language", "en"),
            "sentiment": record.get("sentiment", "neutral"),
            "sentiment_score": record.get("sentiment_score", record.get("sentimentScore", 0.5)),
            "reason": record.get("reason", ""),
            "ministry": record.get("ministry"),
            "keywords": record.get("keywords", []),
            "source": record.get("source", "audio_upload"),
            "original_text": record.get("original_text", record.get("originalText")),
            "translated_text": record.get("translated_text", record.get("translatedText")),
        }
        df_existing = pd.read_csv(self.csv_path) if self.csv_path.exists() else pd.DataFrame()
        df_updated = pd.concat([df_existing, pd.DataFrame([df_row])], ignore_index=True)
        df_updated.to_csv(self.csv_path, index=False)
        self.load()

    def get_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        return self._by_id.get(article_id)

    def filter_articles(
        self,
        q: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        ministries: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        sentiments: Optional[List[str]] = None,
        sort: str = "published_desc",
        offset: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        results = self._articles

        if q and q.strip():
            query_lower = q.strip().lower()
            results = [
                a for a in results
                if query_lower in a["title"].lower()
                or query_lower in a["summary"].lower()
                or (a["ministry"] and query_lower in a["ministry"].lower())
                or any(query_lower in k.lower() for k in a["keywords"])
            ]

        if from_date:
            results = [a for a in results if a["published"][:10] >= from_date[:10]]
        if to_date:
            results = [a for a in results if a["published"][:10] <= to_date[:10]]

        if ministries:
            m_slugs = set(slugify(m) for m in ministries if m)
            results = [a for a in results if a["ministrySlug"] in m_slugs or a["ministry"] in ministries]

        if languages:
            langs_set = set(l.lower() for l in languages)
            results = [a for a in results if a["language"].lower() in langs_set]

        if categories:
            cats_set = set(c.lower() for c in categories)
            results = [a for a in results if a["category"].lower() in cats_set]

        if sentiments:
            sents_set = set(s.lower() for s in sentiments)
            results = [a for a in results if a["sentiment"].lower() in sents_set]

        if sort == "score_desc":
            results.sort(key=lambda x: x["sentimentScore"], reverse=True)
        else:
            results.sort(key=lambda x: x["published"], reverse=True)

        total = len(results)
        items = results[offset : offset + limit]
        next_cursor = str(offset + limit) if (offset + limit) < total else None

        return {"items": items, "nextCursor": next_cursor, "total": total}

    def get_facets(self) -> Dict[str, Any]:
        ministry_counts = {}
        category_counts = {}
        language_counts = {}

        for a in self._articles:
            if a["ministry"]:
                m = a["ministry"]
                ministry_counts[m] = ministry_counts.get(m, 0) + 1
            cat = a["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
            lang = a["language"]
            language_counts[lang] = language_counts.get(lang, 0) + 1

        return {
            "ministries": sorted([{"name": k, "slug": slugify(k), "count": v} for k, v in ministry_counts.items()], key=lambda x: x["count"], reverse=True),
            "categories": sorted([{"category": k, "count": v} for k, v in category_counts.items()], key=lambda x: x["count"], reverse=True),
            "languages": sorted([{"language": k, "count": v} for k, v in language_counts.items()], key=lambda x: x["count"], reverse=True),
            "totalArticles": len(self._articles),
        }

store = ArticleStore()
