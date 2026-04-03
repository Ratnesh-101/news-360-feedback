import streamlit as st
import pandas as pd
import sys
sys.path.append('src')
from pipeline import build_vector_store, build_agent


st.set_page_config(
    page_title='360° News Feedback',
    page_icon='📰',
    layout='wide'
)

@st.cache_data
def load_data():
    return pd.read_csv('data/articles.csv')

@st.cache_resource
def load_agent():
    df = load_data()
    vector_store = build_vector_store(df)
    return build_agent(vector_store)

df = load_data()
agent = load_agent()

# sidebar
st.sidebar.title('📰 360° News Feedback')
page = st.sidebar.radio('Navigate', ['Dashboard', 'Ministry Tracker', 'Chat with News'])

if page == 'Dashboard':
    st.title('📊 News Sentiment Dashboard')
    
    # metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Articles', len(df))
    col2.metric('🟢 Positive', len(df[df['sentiment'] == 'positive']))
    col3.metric('🔴 Negative', len(df[df['sentiment'] == 'negative']))
    col4.metric('🟡 Neutral', len(df[df['sentiment'] == 'neutral']))
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Sentiment Distribution')
        sentiment_counts = df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
    
    with col2:
        st.subheader('Sentiment by Category')
        category_sentiment = df.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
        st.bar_chart(category_sentiment)
    
    st.divider()
    
    st.subheader('Latest Articles')
    st.dataframe(
        df[['title', 'category', 'sentiment', 'sentiment_score', 'ministry']],
        use_container_width=True
    )

elif page == 'Ministry Tracker':
    st.title('🏛️ Ministry Coverage Tracker')
    
    # filter out null ministries
    ministry_df = df[df['ministry'].notna() & (df['ministry'] != 'None')]
    
    st.subheader('Ministries in the News')
    
    col1, col2 = st.columns(2)
    
    with col1:
        ministry_counts = ministry_df['ministry'].value_counts().head(10)
        st.subheader('Most Mentioned Ministries')
        st.bar_chart(ministry_counts)
    
    with col2:
        negative_df = ministry_df[ministry_df['sentiment'] == 'negative']
        negative_counts = negative_df['ministry'].value_counts().head(10)
        st.subheader('Most Negative Coverage')
        st.bar_chart(negative_counts)
    
    st.divider()
    
    st.subheader('Browse by Ministry')
    selected_ministry = st.selectbox(
        'Select Ministry',
        options=['All'] + ministry_df['ministry'].unique().tolist()
    )
    
    if selected_ministry == 'All':
        filtered = ministry_df
    else:
        filtered = ministry_df[ministry_df['ministry'] == selected_ministry]
    
    st.dataframe(
        filtered[['title', 'ministry', 'sentiment', 'sentiment_score', 'reason']],
        use_container_width=True
    )

elif page == 'Chat with News':
    st.title('💬 Chat with the News')
    st.caption('Ask anything about the latest news. Powered by RAG + GPT-4o-mini')
    
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
                result = agent.invoke({
                    'messages': st.session_state.messages
                })
                response = result['messages'][-1].content
            st.markdown(response)
        
        st.session_state.messages.append({'role': 'assistant', 'content': response})