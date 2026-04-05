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
RSS Feeds → scraper.py → cleaner.py (translate + GPT sentiment)
                                ↓
                        data/articles.csv
                                ↓
                    pipeline.py (FAISS + RAG Agent)
                                ↓
                           app.py (Streamlit)
                                ↓
              ┌─────────────────────────────────┐
              │  Dashboard | Ministry Tracker   │
              │  Chat with News | Audio News    │
              └─────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-large |
| Vector Store | FAISS |
| RAG Agent | LangChain + LangGraph |
| Transcription | OpenAI Whisper (medium) |
| Translation | deep-translator |
| Language Detection | langdetect |
| Data | feedparser + pandas |

---

## 🚀 Local Setup

### Prerequisites
- Python 3.13
- Anaconda
- NVIDIA GPU (recommended for Whisper)
- ffmpeg (`conda install ffmpeg -c conda-forge`)

### Installation
```bash
git clone https://github.com/Ratnesh-101/news-360-feedback.git
cd news-360-feedback
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the root:
OPENAI_API_KEY=your_key_here
WHISPER_MODEL=medium

### Run
```bash
# Windows
$env:KMP_DUPLICATE_LIB_OK="TRUE"
streamlit run app.py
```

---

## 📁 Project Structure
news-360-feedback/
├── app.py                  # Streamlit application
├── data/
│   └── articles.csv        # Scraped & analysed articles
├── src/
│   ├── scraper.py          # RSS feed scraper
│   ├── cleaner.py          # Translation + sentiment analysis
│   ├── pipeline.py         # FAISS vector store + RAG agent
│   ├── audioprocessor.py   # Whisper transcription pipeline
│   └── audionews.py        # Audio News page component
├── requirements.txt
└── pyproject.toml

---

## 👥 Team
Built for Smart India Hackathon 2024 — Problem Statement SIH1329