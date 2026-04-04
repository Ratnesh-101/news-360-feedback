from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent
import pandas as pd

load_dotenv()

DATA_PATH = 'data/articles.csv'  # always relative to project root

def load_articles():
    df = pd.read_csv(DATA_PATH)
    df['content'] = df['title'] + '. ' + df['summary']
    return df

def build_vector_store(df):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    if 'content' not in df.columns:
        df['content'] = df['title'] + '. ' + df['summary']

    texts = df['content'].tolist()
    metadatas = df[['title', 'link', 'category', 'published']].to_dict('records')
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    return vector_store

def add_documents_to_vectorstore(vector_store: FAISS, texts: list[str], metadatas: list[dict] = None):
    """Add new documents to an existing FAISS vector store in-place."""
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    new_vs = FAISS.from_texts(
        texts,
        embedding=embeddings,
        metadatas=metadatas or [{} for _ in texts]
    )
    vector_store.merge_from(new_vs)
    return vector_store

def build_agent(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})

    retriever_tool = create_retriever_tool(
        retriever,
        name='news_search',
        description='Search the latest news articles for information about India, politics, and business'
    )

    llm = ChatOpenAI(model='gpt-5-mini', temperature=0)

    agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt='You are a helpful news analyst assistant. Always use the news_search tool first to retrieve relevant articles, then answer based on that context. Always mention which category the news is from.'
    )

    return agent

if __name__ == '__main__':
    print('Loading articles...')
    df = load_articles()
    print('Building vector store...')
    vector_store = build_vector_store(df)
    print('Building agent...')
    agent = build_agent(vector_store)
    result = agent.invoke({'input': 'What is happening with gold prices and the stock market?'})
    print('\nAnswer:', result['output'])