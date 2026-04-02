from langdetect import detect
from deep_translator import GoogleTranslator
import pandas as pd
from scraper import get_all_articles

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
        return text  # if translation fails, return original

def clean_dataframe(df):
    # detect language
    df['language'] = df['summary'].apply(detect_language)
    
    # translate title and summary
    df['title'] = df.apply(lambda row: translate_to_english(row['title'], row['language']), axis=1)
    df['summary'] = df.apply(lambda row: translate_to_english(row['summary'], row['language']), axis=1)
    
    return df

def save_articles():
    df = get_all_articles()
    df = clean_dataframe(df)
    df.to_csv('../data/articles.csv', index=False)
    print(df[['title', 'language', 'category']].to_string())
    print(f'\nSaved {len(df)} articles to data/articles.csv')

if __name__ == '__main__':
    save_articles()