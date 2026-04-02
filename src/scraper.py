import html
import feedparser
import re
import pandas as pd

FEEDS = {
    'india': 'https://timesofindia.indiatimes.com/rssfeeds/296589292.cms',
    'politics': 'https://timesofindia.indiatimes.com/rssfeeds/4719148.cms',
    'business': 'https://timesofindia.indiatimes.com/rssfeeds/1898055.cms',
}


def clean_summary(summary):
    clean = re.sub(r'<.*?>', '', summary)
    clean = html.unescape(clean)  # fixes &amp; &quot; &lt; etc.
    clean = clean.strip()
    return clean

def scrape_feed(url):
    feed = feedparser.parse(url)
    articles = []

    for entry in feed.entries:
        title = html.unescape(entry.title.strip())
        summary = clean_summary(entry.get('summary', ''))
        link = entry.link.strip()

        if not summary:
            continue

        articles.append({
            'title': title,
            'summary': summary,
            'link': link
        })

    return articles

def get_all_articles():
    all_articles = []

    for category, url in FEEDS.items():
        articles = scrape_feed(url)
        for article in articles:
            article['category'] = category  # tag which feed it came from
        all_articles.extend(articles)

    df = pd.DataFrame(all_articles)
    return df

if __name__ == '__main__':
    df = get_all_articles()
    pd.set_option('display.max_colwidth', None)  # show full text
    print(df[['title', 'category']].to_string())
    print(f'\nTotal articles: {len(df)}')