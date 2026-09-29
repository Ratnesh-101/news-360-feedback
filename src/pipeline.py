import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

try:
    from langchain.agents import create_agent
except ImportError:
    try:
        from langgraph.prebuilt import create_react_agent as create_agent
    except ImportError:
        create_agent = None

DATA_PATH = PROJECT_ROOT / 'data' / 'articles.csv'
MODEL_NAME = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')

def load_articles():
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=['title', 'summary', 'link', 'category', 'published'])
    df = pd.read_csv(DATA_PATH)
    df['title'] = df['title'].fillna('')
    df['summary'] = df['summary'].fillna('')
    df['content'] = df['title'] + '. ' + df['summary']
    return df

def build_vector_store(df):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if 'content' not in df.columns:
        title = df['title'].fillna('') if 'title' in df.columns else ''
        summary = df['summary'].fillna('') if 'summary' in df.columns else ''
        df['content'] = title + '. ' + summary

    texts = [str(t) for t in df['content'].fillna('').tolist() if str(t).strip()]
    if not texts:
        texts = ["No news articles available yet."]
        metadatas = [{"title": "None", "link": "", "category": "general", "published": ""}]
    else:
        meta_cols = [c for c in ['title', 'link', 'category', 'published'] if c in df.columns]
        metadatas = df[meta_cols].fillna('').to_dict('records')

    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    return vector_store

def add_documents_to_vectorstore(vector_store: FAISS, texts: list[str], metadatas: list[dict] = None):
    """Add new documents to an existing FAISS vector store in-place."""
    if vector_store is None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        return FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas or [{} for _ in texts])

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    new_vs = FAISS.from_texts(
        texts,
        embedding=embeddings,
        metadatas=metadatas or [{} for _ in texts]
    )
    vector_store.merge_from(new_vs)
    return vector_store

def build_agent(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={'k': 4})

    retriever_tool = create_retriever_tool(
        retriever,
        name='news_search',
        description='Search the latest news articles for information about Indian ministries, politics, economy, and schemes'
    )

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    system_prompt = (
        'You are a helpful news analyst assistant for the Government of India 360° Feedback System. '
        'Always use the news_search tool first to retrieve relevant articles, then answer clearly and concisely based on that context. '
        'Always mention which category or source the news is from, and include the sentiment if relevant.'
    )

    if create_agent is not None:
        agent = create_agent(
            model=llm,
            tools=[retriever_tool],
            system_prompt=system_prompt
        )
        return agent
    else:
        # Fallback minimal agent interface if create_agent is unavailable
        class SimpleRetrieverAgent:
            def __init__(self, llm_model, ret):
                self.llm = llm_model
                self.retriever = ret

            def invoke(self, payload: dict):
                msgs = payload.get('messages', [])
                query = msgs[-1].get('content', '') if msgs else payload.get('input', '')
                docs = self.retriever.invoke(query)
                context = "\n\n".join([f"[{d.metadata.get('category', 'news')}] {d.page_content}" for d in docs])
                prompt_text = f"{system_prompt}\n\nContext:\n{context}\n\nUser Question: {query}"
                response = self.llm.invoke(prompt_text)
                return {'messages': msgs + [{'role': 'assistant', 'content': response.content}]}

        return SimpleRetrieverAgent(llm, retriever)

if __name__ == '__main__':
    print('Loading articles...')
    df = load_articles()
    print('Building vector store...')
    vector_store = build_vector_store(df)
    print('Building agent...')
    agent = build_agent(vector_store)
    result = agent.invoke({'messages': [{'role': 'user', 'content': 'What is happening with gold prices and the stock market?'}]})
    response = result['messages'][-1].content if 'messages' in result else result.get('output', '')
    print('\nAnswer:', response)