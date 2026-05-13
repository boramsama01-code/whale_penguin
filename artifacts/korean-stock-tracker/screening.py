import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from startup_event import ticker_cache

logger = logging.getLogger(__name__)


def _recent_dates(days_back: int = 35) -> tuple[str, str]:
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
    return start, end


async def get_ohlcv(ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
    ticker = str(ticker).zfill(6)
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")
    try:
        from pykrx import stock as pykrx_stock
        df = await asyncio.to_thread(
            pykrx_stock.get_market_ohlcv_by_date, start, end, ticker
        )
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df.tail(days)
    except Exception as e:
        logger.debug("OHLCV 조회 오류 %s: %s", ticker, e)
        return None


async def get_supply(ticker: str, days: int = 20) -> Optional[pd.DataFrame]:
    ticker = str(ticker).zfill(6)
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")
    try:
        from pykrx import stock as pykrx_stock
        df = await asyncio.to_thread(
            pykrx_stock.get_market_net_purchases_of_equities_by_date,
            start, end, ticker
        )
        if df is None or df.empty:
            return None
        return df.tail(days)
    except Exception as e:
        logger.debug("수급 조회 오류 %s: %s", ticker, e)
        return None


async def get_52week_range(ticker: str) -> tuple[float, float]:
    ticker = str(ticker).zfill(6)
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    try:
        from pykrx import stock as pykrx_stock
        df = await asyncio.to_thread(
            pykrx_stock.get_market_ohlcv_by_date, start, end, ticker
        )
        if df is None or df.empty:
            return 0.0, 0.0
        low_col = "저가" if "저가" in df.columns else df.columns[2]
        high_col = "고가" if "고가" in df.columns else df.columns[1]
        return float(df[low_col].min()), float(df[high_col].max())
    except Exception:
        return 0.0, 0.0


def _get_col(df: pd.DataFrame, *names) -> str:
    for n in names:
        if n in df.columns:
            return n
    return df.columns[0]


async def _check_single_ticker(ticker: str, ohlcv_start: str, ohlcv_end: str) -> Optional[dict]:
    ticker = str(ticker).zfill(6)
    try:
        from pykrx import stock as pykrx_stock

        ohlcv = await asyncio.to_thread(
            pykrx_stock.get_market_ohlcv_by_date, ohlcv_start, ohlcv_end, ticker
        )
        if ohlcv is None or len(ohlcv) < 5:
            return None

        vol_col = _get_col(ohlcv, "거래량", "Volume")
        amt_col = _get_col(ohlcv, "거래대금", "Amount")
        close_col = _get_col(ohlcv, "종가", "Close")

        if vol_col not in ohlcv.columns or close_col not in ohlcv.columns:
            return None

        vols = ohlcv[vol_col].values.astype(float)
        closes = ohlcv[close_col].values.astype(float)

        avg20 = np.mean(vols[-20:]) if len(vols) >= 20 else np.mean(vols)
        today_vol = vols[-1]

        if avg20 <= 0 or today_vol < avg20 * 1.5:
            return None

        if amt_col in ohlcv.columns:
            today_amount = float(ohlcv[amt_col].values[-1])
        else:
            today_amount = today_vol * closes[-1]

        if today_amount < 100_000_000:
            return None

        name = ticker_cache.get(ticker, ticker)
        current_price = float(closes[-1])
        change_rate = float((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0.0

        return {
            "ticker": ticker,
            "name": name,
            "price": current_price,
            "change_rate": round(change_rate, 2),
            "volume": int(today_vol),
            "amount": int(today_amount),
            "mktcap": 0,
            "vol_ratio": round(today_vol / avg20, 2),
        }

    except Exception as e:
        logger.debug("종목 필터 오류 %s: %s", ticker, e)
        return None


def _get_all_tickers() -> list[str]:
    """
    DART ticker_cache 기반으로 전체 종목 목록 반환.
    pykrx get_market_ticker_list 는 KRX 서버 상태에 따라 실패하므로 사용 안 함.
    """
    tickers = list(ticker_cache.keys())
    if len(tickers) < 50:
        # 캐시가 너무 작으면 dart_client에서 직접 가져옴
        try:
            from dart_client import get_all_ticker_names
            dart_names = get_all_ticker_names()
            if dart_names:
                ticker_cache.update(dart_names)
                tickers = list(ticker_cache.keys())
        except Exception:
            pass
    return tickers


async def scan_market(max_results: int = 50) -> list[dict]:
    try:
        start_str, end_str = _recent_dates(35)
        tickers = _get_all_tickers()
        logger.info("시장 스캔 시작: %d개 종목", len(tickers))

        if len(tickers) < 10:
            logger.warning("종목 캐시 부족 — 스캔 불가")
            return []

        semaphore = asyncio.Semaphore(8)

        async def safe_check(t):
            async with semaphore:
                return await _check_single_ticker(t, start_str, end_str)

        results = []
        batch_size = 80
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            batch_results = await asyncio.gather(*[safe_check(t) for t in batch], return_exceptions=True)
            for r in batch_results:
                if isinstance(r, dict):
                    results.append(r)
            if len(results) >= max_results * 3:
                break
            await asyncio.sleep(0.3)

        results.sort(key=lambda x: x.get("vol_ratio", 0), reverse=True)

        from scoring import calculate_score
        final = []
        for item in results[:max_results]:
            try:
                ohlcv = await get_ohlcv(item["ticker"])
                supply = await get_supply(item["ticker"])
                if ohlcv is not None:
                    score_data = calculate_score(ohlcv, supply)
                    item["score"] = score_data["total"]
                    item["grade"] = score_data["grade"]
                    item["score_detail"] = score_data
                else:
                    item["score"] = 0.0
                    item["grade"] = "D"
                    item["score_detail"] = {}
            except Exception:
                item["score"] = 0.0
                item["grade"] = "D"
            final.append(item)

        final.sort(key=lambda x: x.get("score", 0), reverse=True)
        logger.info("스캔 완료: %d개 종목", len(final))
        return final

    except Exception as e:
        logger.error("시장 스캔 오류: %s", e)
        return []


async def get_stock_detail(ticker: str) -> Optional[dict]:
    ticker = str(ticker).zfill(6)
    name = ticker_cache.get(ticker, ticker)

    # 이름이 코드 그대로면 pykrx/DART로 조회
    if name == ticker:
        try:
            from pykrx import stock as pykrx_stock
            fetched = await asyncio.to_thread(pykrx_stock.get_market_ticker_name, ticker)
            if fetched:
                name = fetched
                ticker_cache[ticker] = name
        except Exception:
            pass

        if name == ticker:
            try:
                from dart_client import get_all_ticker_names
                dart_names = get_all_ticker_names()
                if ticker in dart_names:
                    name = dart_names[ticker]
                    ticker_cache[ticker] = name
            except Exception:
                pass

    ohlcv = await get_ohlcv(ticker, 60)
    supply = await get_supply(ticker, 20)
    low52, high52 = await get_52week_range(ticker)

    if ohlcv is None or ohlcv.empty:
        logger.warning("OHLCV 없음 %s", ticker)
        return None

    close_col = _get_col(ohlcv, "종가", "Close")
    vol_col = _get_col(ohlcv, "거래량", "Volume")
    open_col = _get_col(ohlcv, "시가", "Open")
    high_col = _get_col(ohlcv, "고가", "High")
    low_col = _get_col(ohlcv, "저가", "Low")

    closes = ohlcv[close_col].values.astype(float)
    vols = ohlcv[vol_col].values.astype(float)

    current_price = float(closes[-1])
    change_rate = float((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0.0

    from scoring import calculate_score
    score_data = calculate_score(ohlcv, supply)

    ohlcv_list = []
    for idx, row in ohlcv.tail(30).iterrows():
        try:
            ohlcv_list.append({
                "date": str(idx)[:10],
                "open": float(row[open_col]),
                "high": float(row[high_col]),
                "low": float(row[low_col]),
                "close": float(row[close_col]),
                "volume": int(row[vol_col]),
            })
        except Exception:
            pass

    supply_list = []
    if supply is not None and not supply.empty:
        for idx, row in supply.tail(20).iterrows():
            entry = {"date": str(idx)[:10]}
            for col in supply.columns:
                try:
                    entry[col] = int(row[col])
                except Exception:
                    pass
            supply_list.append(entry)

    return {
        "ticker": ticker,
        "name": name,
        "price": current_price,
        "change_rate": round(change_rate, 2),
        "volume": int(vols[-1]),
        "high52": high52,
        "low52": low52,
        "score": score_data,
        "ohlcv": ohlcv_list,
        "supply": supply_list,
    }
