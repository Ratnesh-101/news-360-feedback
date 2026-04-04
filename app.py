import streamlit as st
import pandas as pd
import sys
sys.path.append('src')
import os
from audioprocessor import transcribe_audio, append_to_csv
from pipeline import build_vector_store, build_agent, add_documents_to_vectorstore

st.set_page_config(
    page_title='360° News Feedback',
    page_icon='📰',
    layout='wide'
)

# ------------------ DATA + AGENT ------------------ #

@st.cache_data
def load_data():
    return pd.read_csv('data/articles.csv')

@st.cache_resource
def load_agent(_df):
    vector_store = build_vector_store(_df)
    return build_agent(vector_store), vector_store

df = load_data()
agent, vector_store = load_agent(df)

# ------------------ SIDEBAR ------------------ #

st.sidebar.title('📰 360° News Feedback')
page = st.sidebar.radio(
    'Navigate',
    ['Dashboard', 'Ministry Tracker', 'Chat with News', 'Audio News']
)

# ------------------ DASHBOARD ------------------ #

if page == 'Dashboard':
    st.title('📊 News Sentiment Dashboard')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Articles', len(df))
    col2.metric('🟢 Positive', len(df[df['sentiment'] == 'positive']))
    col3.metric('🔴 Negative', len(df[df['sentiment'] == 'negative']))
    col4.metric('🟡 Neutral', len(df[df['sentiment'] == 'neutral']))

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Sentiment Distribution')
        st.bar_chart(df['sentiment'].value_counts())
    with col2:
        st.subheader('Sentiment by Category')
        st.bar_chart(df.groupby(['category', 'sentiment']).size().unstack(fill_value=0))

    st.divider()

    st.subheader('Latest Articles')
    st.dataframe(
        df[['title', 'category', 'sentiment', 'sentiment_score', 'ministry']],
        use_container_width=True
    )

# ------------------ MINISTRY TRACKER ------------------ #

elif page == 'Ministry Tracker':
    st.title('🏛️ Ministry Coverage Tracker')

    ministry_df = df[df['ministry'].notna() & (df['ministry'] != 'None')]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Most Mentioned Ministries')
        st.bar_chart(ministry_df['ministry'].value_counts().head(10))
    with col2:
        st.subheader('Most Negative Coverage')
        st.bar_chart(
            ministry_df[ministry_df['sentiment'] == 'negative']['ministry'].value_counts().head(10)
        )

    st.divider()

    selected_ministry = st.selectbox(
        'Select Ministry',
        options=['All'] + ministry_df['ministry'].unique().tolist()
    )
    filtered = ministry_df if selected_ministry == 'All' else \
        ministry_df[ministry_df['ministry'] == selected_ministry]

    st.dataframe(
        filtered[['title', 'ministry', 'sentiment', 'sentiment_score', 'reason']],
        use_container_width=True
    )

# ------------------ CHAT ------------------ #

elif page == 'Chat with News':
    st.title('💬 Chat with the News')
    st.caption('Ask anything about the latest news. Powered by RAG')

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input('Ask about the news...'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            with st.spinner('Searching news...'):
                result = agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})
                response = result['messages'][-1].content
            st.markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

# ------------------ AUDIO NEWS ------------------ #

elif page == 'Audio News':
    st.title('🎙️ Audio News Analyser')
    st.markdown('Upload a spoken news clip — Whisper transcribes, translates, and runs sentiment analysis.')
    st.info(
        f"⚠️ Using Whisper `{os.getenv('WHISPER_MODEL', 'medium')}`. First run loads the model, subsequent uploads are fast.",
        icon="⏳"
    )

    uploaded_file = st.file_uploader(
        'Upload audio file',
        type=['mp3', 'wav', 'm4a', 'ogg', 'flac', 'mp4'],
        help='Supports Hindi, Marathi, English, and 90+ languages'
    )

    if uploaded_file:
        st.audio(uploaded_file)

    if uploaded_file and st.button('🔍 Transcribe & Analyse', type='primary'):
        with st.spinner(f"Transcribing with Whisper {os.getenv('WHISPER_MODEL', 'medium')}..."):
            try:
                st.session_state['audio_record'] = transcribe_audio(
                    file_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                )
            except Exception as e:
                st.error(f'Transcription failed: {e}')
                st.stop()

    if 'audio_record' in st.session_state:
        record = st.session_state['audio_record']

        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            LANG_NAMES = {
            'hi': 'Hindi', 'mr': 'Marathi', 'en': 'English',
            'pa': 'Punjabi', 'bn': 'Bengali', 'te': 'Telugu',
            'ta': 'Tamil', 'gu': 'Gujarati', 'kn': 'Kannada', 'ur': 'Urdu'
            }
            st.metric('Detected Language', LANG_NAMES.get(record['language'], record['language'].upper()))
        with col2:
            sentiment_color = {
                'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'
            }.get(record['sentiment'].lower(), '⚪')
            st.metric('Sentiment', f"{sentiment_color} {record['sentiment'].capitalize()}")
        with col3:
            st.metric('Sentiment Score', round(record['sentiment_score'], 2))

        st.subheader('📝 Transcript (Original)')
        st.write(record['original_text'])

        if record['language'] != 'en':
            st.subheader('🌐 English Translation')
            st.write(record['translated_text'])

        with st.expander('🏛️ Ministry & Keywords'):
            st.write(f"**Ministry:** {record['ministry'] or 'None detected'}")
            st.write(f"**Keywords:** {', '.join(record['keywords']) if record['keywords'] else '—'}")
            st.write(f"**Reason:** {record['reason']}")

        st.divider()

        if st.button('➕ Add to News Database & Vector Store'):
            with st.spinner('Updating database and vector store...'):
                try:
                    updated_df = append_to_csv(record)
                    add_documents_to_vectorstore(
                        vector_store,
                        texts=[record['summary']],
                        metadatas=[{
                            'title': record['title'],
                            'link': record['link'],
                            'category': record['category'],
                            'published': record['published'],
                        }]
                    )
                    del st.session_state['audio_record']
                    st.cache_data.clear()
                    st.success(f"✅ Added! Database now has {len(updated_df)} articles.")
                except Exception as e:
                    st.error(f'Failed to save: {e}')