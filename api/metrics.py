import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

def compute_net_sentiment(positive: int, negative: int, total: int) -> float:
    """Net Sentiment: -100 to +100 = (positive - negative) / total * 100."""
    if total == 0:
        return 0.0
    return round(((positive - negative) / total) * 100, 1)

def compute_negative_urgency(articles: List[Dict[str, Any]]) -> float:
    """
    Negative urgency (0..100):
    neg_share = neg / total
    recency_weighted_neg = sum(exp(-age_hours / 48) for negative articles)
    negative_urgency = clamp(100 * (0.5 * neg_share + 0.5 * min(recency_weighted_neg / 20, 1)))
    """
    total = len(articles)
    if total == 0:
        return 0.0

    neg_articles = [a for a in articles if a.get("sentiment") == "negative"]
    neg_count = len(neg_articles)
    neg_share = neg_count / total

    now = datetime.now(timezone.utc)
    recency_weighted_neg = 0.0
    for a in neg_articles:
        pub_str = a.get("published")
        age_hours = 24.0  # default fallback age
        if pub_str:
            try:
                # Handle ISO format or fallback
                dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                diff = (now - dt).total_seconds() / 3600.0
                if diff > 0:
                    age_hours = diff
            except Exception:
                pass
        recency_weighted_neg += math.exp(-age_hours / 48.0)

    urgency = 100.0 * (0.5 * neg_share + 0.5 * min(recency_weighted_neg / 20.0, 1.0))
    return round(min(100.0, max(0.0, urgency)), 1)

def compute_z_score_alerts(articles_by_date: Dict[str, List[Dict[str, Any]]], ministry_name: str) -> Tuple[float, int, float, str]:
    """
    Computes negative volume surge z-score for a ministry across the last 14 days.
    z = (neg_24h - mean(neg_daily_last_14d)) / max(std(neg_daily_last_14d), 1.0)
    severity = 'critical' if z >= 3 else 'warning' if z >= 2 else 'info'
    """
    dates = sorted(articles_by_date.keys())
    if not dates:
        return 0.0, 0, 0.0, "info"

    latest_date = dates[-1]
    neg_24h = sum(1 for a in articles_by_date[latest_date] if a.get("sentiment") == "negative")

    past_14 = dates[-15:-1] if len(dates) > 1 else [latest_date]
    past_counts = [sum(1 for a in articles_by_date[d] if a.get("sentiment") == "negative") for d in past_14]

    mean_neg = sum(past_counts) / max(len(past_counts), 1)
    variance = sum((c - mean_neg) ** 2 for c in past_counts) / max(len(past_counts), 1)
    std_neg = math.sqrt(variance)

    z = (neg_24h - mean_neg) / max(std_neg, 1.0)

    severity = "info"
    if z >= 3.0 and neg_24h >= 5:
        severity = "critical"
    elif z >= 2.0 and neg_24h >= 3:
        severity = "warning"

    return round(z, 2), neg_24h, round(mean_neg, 1), severity
