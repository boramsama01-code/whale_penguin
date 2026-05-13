import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from startup_event import ticker_cache

logger = logging.getLogger(__name__)


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
        logger.error("OHLCV 조회 오류 %s: %s", ticker, e)
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
        return float(df["저가"].min()), float(df["고가"].max())
    except Exception:
        return 0.0, 0.0


async def _check_single_ticker(ticker: str, date_str: str) -> Optional[dict]:
    ticker = str(ticker).zfill(6)
    try:
        from pykrx import stock as pykrx_stock

        fundamental = await asyncio.to_thread(
            pykrx_stock.get_market_cap_by_ticker, date_str, market="ALL"
        )
        if fundamental is None or ticker not in fundamental.index.astype(str).str.zfill(6).values:
            return None

        fundamental.index = fundamental.index.astype(str).str.zfill(6)
        if ticker not in fundamental.index:
            return None

        row = fundamental.loc[ticker]
        mktcap = float(row.get("시가총액", 0))

        min_cap = 50_000_000_000
        max_cap = 5_000_000_000_000
        if not (min_cap <= mktcap <= max_cap):
            return None

        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=35)).strftime("%Y%m%d")
        ohlcv = await asyncio.to_thread(
            pykrx_stock.get_market_ohlcv_by_date, start, end, ticker
        )
        if ohlcv is None or len(ohlcv) < 5:
            return None

        vols = ohlcv["거래량"].values.astype(float)
        amounts = ohlcv["거래대금"].values.astype(float)

        avg20 = np.mean(vols[-20:]) if len(vols) >= 20 else np.mean(vols)
        today_vol = vols[-1]
        today_amount = amounts[-1]

        if today_vol < avg20 * 2:
            return None
        if today_amount < 500_000_000:
            return None

        name = ticker_cache.get(ticker, ticker)
        current_price = float(ohlcv["종가"].values[-1])
        change_rate = float(
            (ohlcv["종가"].values[-1] - ohlcv["종가"].values[-2])
            / ohlcv["종가"].values[-2] * 100
        ) if len(ohlcv) >= 2 else 0.0

        return {
            "ticker": ticker,
            "name": name,
            "price": current_price,
            "change_rate": round(change_rate, 2),
            "volume": int(today_vol),
            "amount": int(today_amount),
            "mktcap": int(mktcap),
            "vol_ratio": round(today_vol / avg20, 2),
        }

    except Exception as e:
        logger.debug("종목 필터 오류 %s: %s", ticker, e)
        return None


async def scan_market(max_results: int = 30) -> list[dict]:
    try:
        from pykrx import stock as pykrx_stock

        date_str = datetime.now().strftime("%Y%m%d")
        logger.info("시장 스캔 시작...")

        tickers_raw = await asyncio.to_thread(
            pykrx_stock.get_market_ticker_list, market="ALL"
        )
        tickers = [str(t).zfill(6) for t in tickers_raw]

        logger.info("전체 종목 수: %d개", len(tickers))

        semaphore = asyncio.Semaphore(5)

        async def safe_check(t):
            async with semaphore:
                return await _check_single_ticker(t, date_str)

        results = []
        batch_size = 50
        for i in range(0, min(len(tickers), 500), batch_size):
            batch = tickers[i:i + batch_size]
            batch_results = await asyncio.gather(*[safe_check(t) for t in batch], return_exceptions=True)
            for r in batch_results:
                if isinstance(r, dict):
                    results.append(r)
            if len(results) >= max_results * 2:
                break
            await asyncio.sleep(0.5)

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

    ohlcv = await get_ohlcv(ticker, 60)
    supply = await get_supply(ticker, 20)
    low52, high52 = await get_52week_range(ticker)

    if ohlcv is None:
        return None

    closes = ohlcv["종가"].values.astype(float)
    vols = ohlcv["거래량"].values.astype(float)

    current_price = float(closes[-1])
    change_rate = float((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0.0

    from scoring import calculate_score
    score_data = calculate_score(ohlcv, supply)

    ohlcv_list = []
    for idx, row in ohlcv.tail(30).iterrows():
        ohlcv_list.append({
            "date": str(idx)[:10],
            "open": float(row.get("시가", 0)),
            "high": float(row.get("고가", 0)),
            "low": float(row.get("저가", 0)),
            "close": float(row.get("종가", 0)),
            "volume": int(row.get("거래량", 0)),
        })

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
