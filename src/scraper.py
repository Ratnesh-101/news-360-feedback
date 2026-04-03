import html
import feedparser
import re
import pandas as pd

FEEDS = {
    'india': 'https://timesofindia.indiatimes.com/rssfeeds/296589292.cms',
    'politics': 'https://feeds.feedburner.com/ndtvnews-india-news',
    'business': 'https://timesofindia.indiatimes.com/rssfeeds/1898055.cms',
    'hindi_bhaskar': 'https://www.bhaskar.com/rss-feed/1061/',
    'hindi_amarujala': 'https://www.amarujala.com/rss/india-news.xml',
    'marathi_tv9': 'https://www.tv9marathi.com/feed',
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
            'link': link,
            'published': entry.get('published', 'N/A')
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
#     print(f'\nTotal articles: {len(df)}')
# if __name__ == '__main__':
#     df = get_all_articles()
#     print(df.groupby('category')['title'].count())