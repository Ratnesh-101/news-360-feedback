import html
import re
import ssl
import feedparser
import pandas as pd

# Handle SSL certificates across platforms (especially macOS Python installations)
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass

FEEDS = {
    'india': 'https://timesofindia.indiatimes.com/rssfeeds/296589292.cms',
    'politics': 'https://feeds.feedburner.com/ndtvnews-india-news',
    'business': 'https://timesofindia.indiatimes.com/rssfeeds/1898055.cms',
    'hindi_bhaskar': 'https://www.bhaskar.com/rss-feed/1061/',
    'hindi_amarujala': 'https://www.amarujala.com/rss/india-news.xml',
    'marathi_tv9': 'https://www.tv9marathi.com/feed',
}

def clean_summary(summary):
    if not summary or not isinstance(summary, str):
        return ""
    clean = re.sub(r'<.*?>', '', summary)
    clean = html.unescape(clean)  # fixes &amp; &quot; &lt; etc.
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()

def scrape_feed(url):
    try:
        # Request with standard User-Agent header
        feed = feedparser.parse(url, request_headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    except Exception as e:
        print(f"Error fetching feed {url}: {e}")
        return []

    articles = []

    for entry in getattr(feed, 'entries', []):
        raw_title = getattr(entry, 'title', None) or entry.get('title', '')
        title = html.unescape(str(raw_title).strip()) if raw_title else ''

        # Look in summary, description, or content
        raw_summary = (
            entry.get('summary') or
            entry.get('description') or
            (entry.get('content', [{}])[0].get('value', '') if entry.get('content') else '')
        )
        summary = clean_summary(raw_summary)

        raw_link = getattr(entry, 'link', None) or entry.get('link', '')
        link = str(raw_link).strip() if raw_link else ''

        # Skip if no useful summary or link
        if not summary or not link:
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
        try:
            articles = scrape_feed(url)
            for article in articles:
                article['category'] = category  # tag which feed it came from
            all_articles.extend(articles)
            print(f"[{category}] Fetched {len(articles)} articles.")
        except Exception as e:
            print(f"[{category}] Failed to scrape {url}: {e}")

    df = pd.DataFrame(all_articles)
    return df

if __name__ == '__main__':
    df = get_all_articles()
    pd.set_option('display.max_colwidth', None)
    print(df[['title', 'category']].to_string())
    print(f'\nTotal articles fetched: {len(df)}')