from langdetect import detect
from deep_translator import GoogleTranslator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import json
from scraper import get_all_articles
from dotenv import load_dotenv
load_dotenv()

# initialize once at module level
llm = ChatOpenAI(model='gpt-5-mini', temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ('system', '''You are a news sentiment analyst. Analyze the sentiment of the given news article.
     Respond in JSON format only with these fields:
     {{
         "sentiment": "positive/negative/neutral",
         "score": 0.0 to 1.0,
         "reason": "one line explanation",
         "ministry": "relevant government ministry or scheme if any, else null",
         "keywords": ["key", "terms"]
     }}
     '''),
    ('human', 'Title: {title}\nSummary: {summary}')
])

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

def translate_to_english(text, source_lang):
    try:
        if source_lang == 'en':
            return text
        translated = GoogleTranslator(source='auto', target='english').translate(text)
        return translated
    except:
        return text

def analyze_sentiment(title, summary):
    chain = prompt | llm
    result = chain.invoke({'title': title, 'summary': summary})
    
    try:
        return json.loads(result.content)
    except:
        return {'sentiment': 'neutral', 'score': 0.5, 'reason': 'could not parse', 'ministry': None, 'keywords': []}

def clean_dataframe(df):
    df = df.copy()
    df['language'] = df['summary'].apply(detect_language)
    df['title'] = df.apply(lambda row: translate_to_english(row['title'], row['language']), axis=1)
    df['summary'] = df.apply(lambda row: translate_to_english(row['summary'], row['language']), axis=1)
    
    print('Running sentiment analysis...')
    sentiment_results = df.apply(lambda row: analyze_sentiment(row['title'], row['summary']), axis=1)
    df['sentiment'] = sentiment_results.apply(lambda x: x['sentiment'])
    df['sentiment_score'] = sentiment_results.apply(lambda x: x['score'])
    df['reason'] = sentiment_results.apply(lambda x: x['reason'])
    df['ministry'] = sentiment_results.apply(lambda x: x['ministry'])
    df['keywords'] = sentiment_results.apply(lambda x: x['keywords'])
    
    return df

def save_articles():
    df = get_all_articles()
    
    try:
        existing_df = pd.read_csv('../data/articles.csv')
        existing_links = set(existing_df['link'].tolist())
        new_df = df[~df['link'].isin(existing_links)]
        print(f'New articles to analyze: {len(new_df)}')
        
        if len(new_df) == 0:
            print('No new articles!')
            return
        
        new_df = clean_dataframe(new_df)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    except FileNotFoundError:
        final_df = clean_dataframe(df)
    
    final_df.to_csv('../data/articles.csv', index=False)
    print(final_df[['title', 'sentiment', 'sentiment_score', 'ministry', 'category']].to_string())
    print(f'\nSaved {len(final_df)} articles to data/articles.csv')

if __name__ == '__main__':
    save_articles()