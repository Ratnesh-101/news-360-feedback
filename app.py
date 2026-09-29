import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Path setup
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
DATA_FILE = ROOT_DIR / "data" / "articles.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(ROOT_DIR / ".env")

from audionews import render_audio_news_page
from pipeline import build_vector_store, build_agent, add_documents_to_vectorstore

st.set_page_config(
    page_title='360° News Feedback — SIH1329',
    page_icon='📰',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ------------------ DATA LOADER ------------------ #

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return pd.DataFrame(columns=[
            'title', 'summary', 'link', 'published', 'category', 'language',
            'sentiment', 'sentiment_score', 'reason', 'ministry', 'keywords',
            'source', 'original_text', 'translated_text'
        ])
    df = pd.read_csv(DATA_FILE)
    # Normalize sentiment
    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].fillna('neutral').astype(str).str.lower().str.strip()
    if 'sentiment_score' in df.columns:
        df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0.5)
    return df

@st.cache_resource
def get_cached_agent(_df):
    vector_store = build_vector_store(_df)
    agent = build_agent(vector_store)
    return agent, vector_store

def get_agent_and_vectorstore(df):
    """Lazily load the vector store and agent when needed."""
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("custom_openai_key")
    if not api_key:
        return None, None
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        return get_cached_agent(df)
    except Exception as e:
        st.error(f"Error initializing AI Agent: {e}")
        return None, None

df = load_data()

# ------------------ SIDEBAR ------------------ #

st.sidebar.title('📰 360° News Feedback')
st.sidebar.caption('Real-Time Multilingual News Intelligence for Government of India (SIH1329)')

page = st.sidebar.radio(
    'Navigation',
    ['📊 Dashboard', '🏛️ Ministry Tracker', '💬 Chat with News', '🎙️ Audio News']
)

st.sidebar.divider()
st.sidebar.subheader('System Status')
st.sidebar.text(f"Total Articles: {len(df)}")
if 'published' in df.columns and not df.empty:
    latest_pub = df['published'].dropna().max()
    st.sidebar.text(f"Latest: {str(latest_pub)[:10]}")

has_api_key = bool(os.getenv("OPENAI_API_KEY") or st.session_state.get("custom_openai_key"))
if has_api_key:
    st.sidebar.success("🟢 OpenAI API: Configured")
else:
    st.sidebar.warning("🟡 OpenAI API: Not Configured")

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ------------------ DASHBOARD ------------------ #

if page == '📊 Dashboard':
    st.title('📊 News Sentiment Dashboard')
    st.caption('Comprehensive overview of national and regional media sentiment across Hindi, Marathi, and English sources.')

    total_articles = len(df)
    pos_count = len(df[df['sentiment'] == 'positive'])
    neg_count = len(df[df['sentiment'] == 'negative'])
    neu_count = len(df[df['sentiment'] == 'neutral'])

    pos_pct = f"{round((pos_count / total_articles * 100), 1)}%" if total_articles else "0%"
    neg_pct = f"{round((neg_count / total_articles * 100), 1)}%" if total_articles else "0%"
    neu_pct = f"{round((neu_count / total_articles * 100), 1)}%" if total_articles else "0%"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Articles', total_articles)
    col2.metric('🟢 Positive', pos_count, delta=pos_pct, delta_color="normal")
    col3.metric('🔴 Negative', neg_count, delta=f"-{neg_pct}", delta_color="inverse")
    col4.metric('🟡 Neutral', neu_count, delta=neu_pct, delta_color="off")

    st.divider()

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Sentiment Distribution')
        if not df.empty:
            sentiment_counts = df['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
        else:
            st.info("No data available.")

    with col2:
        st.subheader('Sentiment by Category')
        if not df.empty and 'category' in df.columns:
            category_sentiment = df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
            st.bar_chart(category_sentiment)
        else:
            st.info("No data available.")

    st.divider()

    # Interactive Filter & Search
    st.subheader('Explore News Articles')
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 3])

    all_categories = sorted(df['category'].dropna().unique().tolist()) if 'category' in df.columns else []
    with filter_col1:
        selected_categories = st.multiselect('Filter by Category', options=all_categories, default=[])

    with filter_col2:
        selected_sentiments = st.multiselect('Filter by Sentiment', options=['positive', 'negative', 'neutral'], default=[])

    with filter_col3:
        search_query = st.text_input('🔍 Search Articles (Title / Summary / Ministry)', '')

    # Apply filters
    filtered_df = df.copy()
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    if selected_sentiments:
        filtered_df = filtered_df[filtered_df['sentiment'].isin(selected_sentiments)]
    if search_query.strip():
        q = search_query.lower()
        title_match = filtered_df['title'].astype(str).str.lower().str.contains(q, na=False)
        summary_match = filtered_df['summary'].astype(str).str.lower().str.contains(q, na=False)
        ministry_match = filtered_df['ministry'].astype(str).str.lower().str.contains(q, na=False)
        filtered_df = filtered_df[title_match | summary_match | ministry_match]

    st.caption(f"Showing {len(filtered_df)} of {len(df)} articles")

    display_cols = [c for c in ['title', 'category', 'sentiment', 'sentiment_score', 'ministry', 'published', 'link'] if c in filtered_df.columns]
    st.dataframe(
        filtered_df[display_cols],
        column_config={
            "link": st.column_config.LinkColumn("Source URL"),
            "sentiment_score": st.column_config.ProgressColumn(
                "Score",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
        },
        use_container_width=True,
        hide_index=True
    )

# ------------------ MINISTRY TRACKER ------------------ #

elif page == '🏛️ Ministry Coverage Tracker':
    st.title('🏛️ Ministry Coverage Tracker')
    st.caption('Identify policy reception, public response, and targeted feedback across Central Government ministries.')

    ministry_df = df[df['ministry'].notna() & (df['ministry'] != 'None') & (df['ministry'].str.strip() != '')]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Most Mentioned Ministries')
        if not ministry_df.empty:
            st.bar_chart(ministry_df['ministry'].value_counts().head(10))
        else:
            st.info("No ministry mentions detected.")

    with col2:
        st.subheader('Highest Negative Coverage')
        neg_ministries = ministry_df[ministry_df['sentiment'] == 'negative']
        if not neg_ministries.empty:
            st.bar_chart(neg_ministries['ministry'].value_counts().head(10))
        else:
            st.info("No negative coverage detected for ministries.")

    st.divider()

    st.subheader('Ministry Drill-Down')
    unique_ministries = sorted(ministry_df['ministry'].unique().tolist())
    selected_ministry = st.selectbox(
        'Select Ministry or Scheme',
        options=['All Ministries'] + unique_ministries
    )

    if selected_ministry == 'All Ministries':
        target_df = ministry_df
    else:
        # Match exact or partial match if multiple ministries were tagged in one row
        target_df = ministry_df[ministry_df['ministry'].astype(str).str.contains(selected_ministry, regex=False)]

    if not target_df.empty:
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_total = len(target_df)
        m_pos = len(target_df[target_df['sentiment'] == 'positive'])
        m_neg = len(target_df[target_df['sentiment'] == 'negative'])
        m_avg_score = target_df['sentiment_score'].mean()

        m_col1.metric("Total Mentions", m_total)
        m_col2.metric("Positive Coverage", m_pos)
        m_col3.metric("Negative Coverage", m_neg)
        m_col4.metric("Avg Sentiment Score", f"{m_avg_score:.2f}")

    display_cols = [c for c in ['title', 'ministry', 'sentiment', 'sentiment_score', 'reason', 'published'] if c in target_df.columns]
    st.dataframe(
        target_df[display_cols],
        column_config={
            "sentiment_score": st.column_config.ProgressColumn(
                "Score",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
        },
        use_container_width=True,
        hide_index=True
    )

# ------------------ CHAT WITH NEWS ------------------ #

elif page == '💬 Chat with News':
    st.title('💬 Chat with the News')
    st.caption('Ask questions about national & regional news, policies, or ministerial coverage. Powered by RAG with FAISS.')

    # Check for API key
    agent, vector_store = get_agent_and_vectorstore(df)

    if not agent:
        st.warning('⚠️ OpenAI API Key is required for the RAG News Agent.')
        custom_key = st.text_input('Enter OpenAI API Key to enable chat:', type='password', key='chat_key_input')
        if st.button('Activate Key'):
            st.session_state['custom_openai_key'] = custom_key
            st.rerun()
        st.info("You can also add `OPENAI_API_KEY=your_key` to a `.env` file in the project root.")
    else:
        # Prompt Suggestions
        st.markdown("**Suggested questions:**")
        chip_col1, chip_col2, chip_col3 = st.columns(3)
        sample_q = None
        if chip_col1.button("📉 What are the concerns regarding oil prices or inflation?", use_container_width=True):
            sample_q = "What are the concerns regarding oil prices or inflation?"
        if chip_col2.button("🏛️ Which government schemes received positive media coverage?", use_container_width=True):
            sample_q = "Which government schemes received positive media coverage?"
        if chip_col3.button("🗞️ Summarize the key stories from Hindi and Marathi news.", use_container_width=True):
            sample_q = "Summarize the key stories from Hindi and Marathi news."

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if st.session_state.messages and st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        chat_input = st.chat_input('Ask anything about the latest news...')
        prompt = sample_q or chat_input

        if prompt:
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            with st.chat_message('user'):
                st.markdown(prompt)

            with st.chat_message('assistant'):
                with st.spinner('Searching news database...'):
                    try:
                        result = agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})
                        if isinstance(result, dict) and 'messages' in result:
                            response = result['messages'][-1].content
                        elif isinstance(result, dict) and 'output' in result:
                            response = result['output']
                        else:
                            response = str(result)
                    except Exception as e:
                        response = f"⚠️ Could not process query: {e}"
                    st.markdown(response)

            st.session_state.messages.append({'role': 'assistant', 'content': response})

# ------------------ AUDIO NEWS ------------------ #

elif page == '🎙️ Audio News':
    agent, vector_store = get_agent_and_vectorstore(df)
    render_audio_news_page(vector_store=vector_store)