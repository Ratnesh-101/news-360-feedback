import os
import sys
from pathlib import Path
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audioprocessor import transcribe_audio, append_to_csv
from pipeline import add_documents_to_vectorstore

LANG_NAMES = {
    'hi': 'Hindi', 'mr': 'Marathi', 'en': 'English',
    'pa': 'Punjabi', 'bn': 'Bengali', 'te': 'Telugu',
    'ta': 'Tamil', 'gu': 'Gujarati', 'kn': 'Kannada', 'ur': 'Urdu'
}

def render_audio_news_page(vector_store=None):
    st.title("🎙️ Audio News Analyser")
    st.markdown("Upload a spoken news clip — Whisper transcribes, translates, and runs sentiment analysis.")

    whisper_model = os.getenv("WHISPER_MODEL", "base")
    st.info(f"⚠️ Using Whisper `{whisper_model}`. First run downloads/loads the model; subsequent uploads are fast.", icon="⏳")

    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=["mp3", "wav", "m4a", "ogg", "flac", "mp4"],
        help="Supports Hindi, Marathi, English, and 90+ languages"
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("🔍 Transcribe & Analyse", type="primary", key="transcribe_btn"):
            with st.spinner(f"Transcribing and analyzing with Whisper ({whisper_model})..."):
                try:
                    record = transcribe_audio(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                    )
                    st.session_state['audio_record'] = record
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    return

    if 'audio_record' in st.session_state:
        record = st.session_state['audio_record']
        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            lang_code = record.get('language', 'unknown')
            st.metric("Detected Language", LANG_NAMES.get(lang_code, lang_code.upper()))
        with col2:
            sentiment_raw = record.get('sentiment', 'neutral').lower()
            sentiment_color = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}.get(sentiment_raw, '⚪')
            st.metric("Sentiment", f"{sentiment_color} {sentiment_raw.capitalize()}")
        with col3:
            st.metric("Sentiment Score", round(float(record.get('sentiment_score', 0.5)), 2))

        st.subheader("📝 Transcript (Original)")
        st.write(record.get('original_text', ''))

        if record.get('language') != 'en' and record.get('translated_text'):
            st.subheader("🌐 English Translation")
            st.write(record.get('translated_text', ''))

        with st.expander("🏛️ Ministry & Keywords", expanded=True):
            ministry_val = record.get('ministry') or 'None detected'
            st.write(f"**Ministry:** {ministry_val}")
            kw_list = record.get('keywords') or []
            st.write(f"**Keywords:** {', '.join(kw_list) if kw_list else '—'}")
            st.write(f"**Reason:** {record.get('reason', '')}")

        st.divider()

        if st.button("➕ Add to News Database & Vector Store", type="primary", key="save_audio_btn"):
            with st.spinner("Updating database and vector store..."):
                try:
                    updated_df = append_to_csv(record)
                    if vector_store is not None:
                        add_documents_to_vectorstore(
                            vector_store,
                            texts=[record.get('summary', '')],
                            metadatas=[{
                                'title': record.get('title', ''),
                                'link': record.get('link', ''),
                                'category': record.get('category', 'audio'),
                                'published': record.get('published', ''),
                            }]
                        )
                    del st.session_state['audio_record']
                    st.cache_data.clear()
                    st.success(f"✅ Added! Database now has {len(updated_df)} articles.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save: {e}")