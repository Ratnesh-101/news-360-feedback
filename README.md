# 📰 360° News Feedback System — SIH1329

A real-time multilingual news sentiment analysis dashboard built for the **Smart India Hackathon 2024** (Problem Statement SIH1329). Helps Government of India ministries track public sentiment across Hindi, Marathi, and English news sources.

## 🌐 Live Demo
[https://news-360-feedback-18052906.streamlit.app/](https://news-360-feedback-18052906.streamlit.app/)

---

## 🎯 Problem Statement
Government ministries lack a unified system to monitor how their policies and schemes are being covered across regional and national media in multiple languages. This system provides real-time 360° feedback on news sentiment.

---

## ✨ Features

- **Dashboard** — Sentiment distribution, category breakdown, latest articles
- **Ministry Tracker** — Track which ministries are getting positive/negative coverage
- **Chat with News** — RAG-powered chatbot to query the news database
- **Audio News** — Upload Hindi/Marathi/English audio clips → auto-transcribe → sentiment analysis

---

## 🗞️ News Sources

| Source | Language | Category |
|--------|----------|----------|
| Times of India | English | India, Business |
| NDTV | English | Politics |
| Dainik Bhaskar | Hindi | General |
| Amar Ujala | Hindi | General |
| TV9 Marathi | Marathi | General |

---

## 🏗️ Architecture
```text
RSS Feeds → scraper.py → cleaner.py (translate + LLM sentiment)
                                ↓
                        data/articles.csv
                                ↓
                    FastAPI Bridge (api/main.py)
                                ↓
        ┌────────────────────────────────────────────────────────┐
        │  Vite + React 19 + TypeScript (web/)                   │
        │  • Command Center Dashboard (KPI Grid, Net Index)      │
        │  • Ministry Reputation Radar (5-Axis Radar Chart)      │
        │  • RAG Chat Analyst Drawer (SSE Streaming + Citations) │
        │  • Audio Studio (Whisper Dual-Transcript + Approval)   │
        │  • Daily Executive Situation Brief (Printable Gazette) │
        └────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 19 + Vite + TypeScript (Strict) + Tailwind CSS |
| UI & Charts | Recharts, Lucide Icons, Situation Room Tokens |
| State & Routing | TanStack Query v5, Zustand, React Router |
| Backend API | FastAPI + Uvicorn (REST + SSE Streaming) |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-large |
| Vector Store | FAISS |
| Audio Transcription | OpenAI Whisper |
| Translation | deep-translator |
| Regional Media | Hindi, Marathi, English RSS Feeds (Bhaskar, Amar Ujala, TV9, TOI, NDTV) |

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10 – 3.13
- ffmpeg (`brew install ffmpeg` on macOS, or `conda install ffmpeg -c conda-forge`)

### Installation
```bash
git clone https://github.com/Ratnesh-101/news-360-feedback.git
cd news-360-feedback
pip install -r requirements.txt
```

### Environment Variables
Copy `.env.example` to `.env` in the root:
```bash
cp .env.example .env
```
And add your OpenAI API key:
```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
WHISPER_MODEL=base
```

### Run Application

#### Option 1: Modern Situation Room (React + TypeScript + FastAPI)
```bash
./start_dev.sh
```
Or start individually:
```bash
# Terminal 1: FastAPI Backend
.venv/bin/uvicorn api.main:app --reload --port 8000

# Terminal 2: React + TypeScript Frontend
cd web && npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

#### Option 2: Classic Streamlit Dashboard
```bash
streamlit run app.py
```

---
## 📁 Project Structure
```text
news-360-feedback/
├── api/                    # FastAPI Backend Bridge
│   ├── main.py             # REST + SSE endpoints
│   ├── storage.py          # In-memory indexing & querying
│   └── metrics.py          # Net sentiment, z-score alerts, urgency
├── web/                    # React 19 + TypeScript + Vite Frontend
│   ├── src/
│   │   ├── app/            # Layout, AppShell, router
│   │   ├── features/       # Dashboard, Ministries, Chat, Audio, Briefings
│   │   ├── components/     # SentimentBadge, ScoreMeter, LangTag, ArticleDrawer
│   │   ├── lib/            # API client, SSE streaming, sentiment utils
│   │   ├── stores/         # Zustand filters and UI state
│   │   └── types/          # Strict TypeScript domain interfaces
│   └── package.json
├── data/
│   └── articles.csv        # Scraped & analysed articles (391 records)
├── src/                    # Core Python Scrapers & Pipelines
│   ├── scraper.py          # RSS feed scraper (TOI, NDTV, Bhaskar, TV9)
│   ├── cleaner.py          # Translation + sentiment analysis
│   ├── pipeline.py         # FAISS vector store + RAG agent
│   ├── audioprocessor.py   # Whisper transcription pipeline
│   └── audionews.py        # Audio News component
├── app.py                  # Streamlit application
├── start_dev.sh            # Development launcher script
├── requirements.txt
└── pyproject.toml
```

---

## 👥 Team
Built for Smart India Hackathon 2024 — Problem Statement SIH1329