"""
News Fetcher - Fetches market news from Alpaca API
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY


class AlpacaNewsFetcher:
    """Fetches news from Alpaca's News API."""

    BASE_URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or ALPACA_API_KEY
        self.secret_key = secret_key or ALPACA_SECRET_KEY
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }

    def get_news(
        self,
        symbols: List[str] = None,
        limit: int = 50,
        start: datetime = None,
        end: datetime = None,
        include_content: bool = True
    ) -> List[Dict]:
        """
        Fetch news articles from Alpaca.

        Args:
            symbols: List of stock symbols to filter news
            limit: Maximum number of articles (max 50)
            start: Start datetime for news
            end: End datetime for news
            include_content: Include full article content

        Returns:
            List of news articles with sentiment-ready data
        """
        params = {
            "limit": min(limit, 50),
            "include_content": str(include_content).lower()
        }

        if symbols:
            params["symbols"] = ",".join(symbols)

        if start:
            params["start"] = start.isoformat() + "Z"

        if end:
            params["end"] = end.isoformat() + "Z"

        try:
            response = requests.get(
                self.BASE_URL,
                headers=self.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("news", []):
                article = {
                    "id": item.get("id"),
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "content": item.get("content", ""),
                    "author": item.get("author", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "symbols": item.get("symbols", []),
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                    "images": item.get("images", [])
                }
                articles.append(article)

            return articles

        except requests.exceptions.RequestException as e:
            print(f"[News] Error fetching news: {e}")
            return []

    def get_symbol_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get news for a specific symbol."""
        return self.get_news(symbols=[symbol], limit=limit)

    def get_market_news(self, limit: int = 20) -> List[Dict]:
        """Get general market news (SPY, QQQ, major indices)."""
        return self.get_news(
            symbols=["SPY", "QQQ", "DIA", "IWM"],
            limit=limit
        )

    def get_recent_news(self, hours: int = 4, limit: int = 50) -> List[Dict]:
        """Get news from the last N hours."""
        start = datetime.utcnow() - timedelta(hours=hours)
        return self.get_news(start=start, limit=limit)

    def get_breaking_news(self, minutes: int = 30, limit: int = 20) -> List[Dict]:
        """Get very recent news (potential market movers)."""
        start = datetime.utcnow() - timedelta(minutes=minutes)
        return self.get_news(start=start, limit=limit)


# Singleton instance
_news_fetcher = None


def get_news_fetcher() -> AlpacaNewsFetcher:
    """Get singleton news fetcher instance."""
    global _news_fetcher
    if _news_fetcher is None:
        _news_fetcher = AlpacaNewsFetcher()
    return _news_fetcher
