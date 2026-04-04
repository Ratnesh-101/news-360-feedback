import feedparser

urls = [
    'https://timesofindia.indiatimes.com/rssfeeds/4719167.cms',
    'https://timesofindia.indiatimes.com/rssfeeds/4719162.cms',
    'https://timesofindia.indiatimes.com/rssfeeds/4719160.cms',
    'https://timesofindia.indiatimes.com/rssfeeds/4719155.cms',
    'https://timesofindia.indiatimes.com/rssfeeds/4719150.cms',
    'https://timesofindia.indiatimes.com/rssfeeds/4719145.cms',
]

for url in urls:
    feed = feedparser.parse(url)
    if feed.entries:
        print(f'{url} → {feed.entries[0].title}')
    else:
        print(f'{url} → no entries')