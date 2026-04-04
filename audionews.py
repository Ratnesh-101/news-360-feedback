from src.audio_processor import transcribe_audio, append_to_articles_csv
from src.pipeline import add_documents_to_vectorstore  # adjust to your actual fn name


def render_audio_news_page():
    st.title("🎙️ Audio News Analyser")
    st.markdown("Upload a spoken news clip — Whisper transcribes, translates, and runs sentiment analysis.")

    # Lazy-load model notice
    st.info("⚠️ First run loads Whisper `large-v3` (~3GB). Subsequent uploads are fast.", icon="⏳")

    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=["mp3", "wav", "m4a", "ogg", "flac", "mp4"],
        help="Supports Hindi, Marathi, English, and 90+ languages"
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("🔍 Transcribe & Analyse", type="primary"):
            with st.spinner("Transcribing with Whisper large-v3..."):
                try:
                    record = transcribe_audio(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                    )
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    return

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Language", record["language"].upper())
            with col2:
                sentiment_color = {
                    "Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"
                }.get(record["sentiment"], "⚪")
                st.metric("Sentiment", f"{sentiment_color} {record['sentiment']}")

            st.subheader("📝 Transcript (Original)")
            st.write(record["original_text"])

            if record["language"] != "en":
                st.subheader("🌐 English Translation")
                st.write(record["translated_text"])

            # Add to pipeline
            if st.button("➕ Add to News Database & Vector Store"):
                with st.spinner("Updating database..."):
                    df = append_to_articles_csv(record)
                    # Push to your FAISS vector store
                    add_documents_to_vectorstore([record["translated_text"]])
                    st.success(f"✅ Added! Database now has {len(df)} articles.")