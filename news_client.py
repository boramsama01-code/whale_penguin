import asyncio
import logging

logger = logging.getLogger(__name__)

_news_cache: dict[str, dict] = {}
_CACHE_TTL = 600  # 10분

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 11) AppleWebKit/537.36 Chrome/120",
    "Referer": "https://m.stock.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


async def get_stock_news(ticker: str, name: str = "", limit: int = 6) -> list[dict]:
    import time
    ticker = str(ticker).zfill(6)
    cached = _news_cache.get(ticker)
    if cached and time.time() - cached["ts"] < _CACHE_TTL:
        return cached["items"][:limit]

    items = await asyncio.to_thread(_fetch, ticker, name, limit)
    _news_cache[ticker] = {"items": items, "ts": time.time()}
    return items[:limit]


def _fetch(ticker: str, name: str, limit: int) -> list[dict]:
    try:
        import httpx
        # page=1 부터 실제 데이터가 옴 (page=0 은 빈 배열)
        r = httpx.get(
            f"https://m.stock.naver.com/api/news/stock/{ticker}",
            params={"pageSize": limit, "page": 1},
            headers=HEADERS,
            timeout=8,
            follow_redirects=True,
        )
        if r.status_code != 200:
            return []

        data = r.json()
        # 응답: list[{total: int, items: list[{...}]}]
        results = []
        for group in data if isinstance(data, list) else []:
            for item in group.get("items", []):
                title = item.get("title") or item.get("titleFull", "")
                url = item.get("mobileNewsUrl") or item.get("url", "")
                dt = _fmt_date(item.get("datetime", ""))
                source = item.get("officeName", "")
                if title and url:
                    results.append({"title": title, "url": url, "date": dt, "source": source})
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        return results
    except Exception as e:
        logger.warning("뉴스 크롤링 실패 (%s): %s", ticker, e)
        return []


def _fmt_date(raw: str) -> str:
    raw = str(raw).strip()
    if len(raw) >= 12:
        return f"{raw[4:6]}/{raw[6:8]} {raw[8:10]}:{raw[10:12]}"
    if len(raw) == 8:
        return f"{raw[4:6]}/{raw[6:8]}"
    return raw[:16]
