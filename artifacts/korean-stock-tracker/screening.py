import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from startup_event import ticker_cache

logger = logging.getLogger(__name__)


async def get_ohlcv(ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
    """일별 OHLCV 조회 — KIS API 우선, pykrx fallback."""
    ticker = str(ticker).zfill(6)
    # KIS API 먼저 시도
    try:
        from kis_data import get_ohlcv_kis
        df = await get_ohlcv_kis(ticker, days)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.debug("KIS OHLCV 실패 %s: %s", ticker, e)

    # pykrx fallback (한국 IP 환경)
    try:
        from pykrx import stock as pykrx_stock
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")
        df = await asyncio.to_thread(
            pykrx_stock.get_market_ohlcv_by_date, start, end, ticker
        )
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            return df.tail(days)
    except Exception as e:
        logger.debug("pykrx OHLCV 실패 %s: %s", ticker, e)

    return None


async def get_supply(ticker: str, days: int = 20) -> Optional[pd.DataFrame]:
    """수급 데이터 조회 — pykrx (KIS API 미지원)."""
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
    """52주 고저가 — KIS 현재가 API 우선 (52주 고저 포함), pykrx fallback."""
    ticker = str(ticker).zfill(6)
    try:
        from kis_data import get_current_price_kis
        info = await get_current_price_kis(ticker)
        if info and info.get("high52", 0) > 0:
            return float(info["low52"]), float(info["high52"])
    except Exception:
        pass

    # pykrx fallback
    try:
        from pykrx import stock as pykrx_stock
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
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


async def scan_market(max_results: int = 50) -> list[dict]:
    """
    시장 스캔 — KIS 거래량 순위 API 우선 (Replit IP 지원),
    실패 시 pykrx bulk fallback.
    """
    try:
        from kis_data import get_volume_rank_kis
        from scoring import calculate_score

        logger.info("세력 스캐너 시작 (KIS API)")

        # KOSPI + KOSDAQ 거래량 순위 동시 조회
        kospi_list, kosdaq_list = await asyncio.gather(
            get_volume_rank_kis("J", 100),
            get_volume_rank_kis("Q", 100),
        )

        candidates = []
        seen = set()
        for item in (kospi_list or []) + (kosdaq_list or []):
            t = item.get("ticker", "")
            if not t or t in seen:
                continue
            seen.add(t)
            # 거래대금 1억 이상, 거래량 증가율 150% 이상
            if item.get("amount", 0) < 100_000_000:
                continue
            if item.get("vol_ratio", 0) < 1.5:
                continue
            name = ticker_cache.get(t) or item.get("name") or t
            candidates.append({
                "ticker": t,
                "name": name,
                "price": item["price"],
                "change_rate": item["change_rate"],
                "volume": item["volume"],
                "amount": int(item["amount"]),
                "mktcap": 0,
                "vol_ratio": item["vol_ratio"],
            })

        if candidates:
            candidates.sort(key=lambda x: x["vol_ratio"], reverse=True)
            top_candidates = candidates[:min(80, len(candidates))]
            logger.info("1차 필터 통과: %d개 → 점수 계산 중...", len(top_candidates))

            semaphore = asyncio.Semaphore(8)

            async def score_candidate(item):
                async with semaphore:
                    try:
                        ohlcv = await asyncio.wait_for(get_ohlcv(item["ticker"], 60), timeout=15.0)
                        supply = await asyncio.wait_for(get_supply(item["ticker"], 20), timeout=10.0)
                        if ohlcv is not None and not ohlcv.empty:
                            mktcap = item.get("mktcap", 0) or 0
                            score_data = calculate_score(ohlcv, supply, mktcap=mktcap)
                            item["score"] = round(score_data["total"], 2)
                            item["grade"] = score_data["grade"]
                        else:
                            item["score"] = 0.0
                            item["grade"] = "D"
                    except Exception:
                        item["score"] = 0.0
                        item["grade"] = "D"
                    return item

            scored = await asyncio.gather(*[score_candidate(c) for c in top_candidates])
            final = [s for s in scored if isinstance(s, dict)]
            final.sort(key=lambda x: x.get("score", 0), reverse=True)
            logger.info("스캔 완료 (KIS): %d개 반환", len(final[:max_results]))
            return final[:max_results]

        # KIS API 결과 없으면 pykrx fallback
        logger.warning("KIS 거래량 순위 결과 없음 — pykrx fallback 시도")
        return await _scan_market_pykrx(max_results)

    except Exception as e:
        logger.error("시장 스캔 오류: %s", e)
        return await _scan_market_pykrx(max_results)


def _recent_biz_dates(n: int = 3) -> list[str]:
    dates = []
    d = datetime.now()
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))
        d -= timedelta(days=1)
    return dates


async def _bulk_ohlcv_by_ticker(date_str: str, market: str = "ALL") -> Optional[pd.DataFrame]:
    try:
        from pykrx import stock as pykrx_stock
        df = await asyncio.wait_for(
            asyncio.to_thread(pykrx_stock.get_market_ohlcv_by_ticker, date_str, market=market),
            timeout=35.0
        )
        if df is None or (hasattr(df, 'empty') and df.empty):
            return None
        return df
    except Exception as e:
        logger.warning("bulk_ohlcv %s 실패: %s", date_str, e)
        return None


async def _scan_market_pykrx(max_results: int = 50) -> list[dict]:
    """pykrx 기반 스캔 (한국 IP fallback)."""
    try:
        biz_dates = _recent_biz_dates(3)
        today_str = biz_dates[0]
        yesterday_str = biz_dates[1] if len(biz_dates) > 1 else None
        two_ago_str = biz_dates[2] if len(biz_dates) > 2 else None

        logger.info("세력 스캐너 (pykrx) — 기준일: %s", today_str)

        async def _none():
            return None

        today_df, yesterday_df = await asyncio.gather(
            _bulk_ohlcv_by_ticker(today_str),
            _bulk_ohlcv_by_ticker(yesterday_str) if yesterday_str else _none()
        )

        if today_df is None or today_df.empty:
            today_df = yesterday_df
            yesterday_df = await _bulk_ohlcv_by_ticker(two_ago_str) if two_ago_str else None

        if today_df is None or today_df.empty:
            logger.warning("시장 데이터 조회 실패 (pykrx)")
            return []

        vol_col = next((c for c in ['거래량', 'Volume'] if c in today_df.columns), None)
        amt_col = next((c for c in ['거래대금'] if c in today_df.columns), None)
        close_col = next((c for c in ['종가', 'Close'] if c in today_df.columns), None)
        chg_col = next((c for c in ['등락률'] if c in today_df.columns), None)

        if not vol_col or not close_col:
            return []

        candidates = []
        for ticker_raw in today_df.index:
            ticker = str(ticker_raw).zfill(6)
            row = today_df.loc[ticker_raw]
            today_vol = float(row[vol_col]) if vol_col else 0
            amount = float(row[amt_col]) if amt_col else 0
            close = float(row[close_col]) if close_col else 0
            change_rate = float(row[chg_col]) if chg_col else 0.0
            if amount < 100_000_000 or today_vol <= 0 or close <= 0:
                continue
            vol_ratio = 1.0
            if yesterday_df is not None and not yesterday_df.empty and ticker_raw in yesterday_df.index:
                try:
                    yd_vol = float(yesterday_df.loc[ticker_raw, vol_col]) if vol_col in yesterday_df.columns else 0
                    if yd_vol > 0:
                        vol_ratio = round(today_vol / yd_vol, 2)
                except Exception:
                    pass
            if vol_ratio < 1.5:
                continue
            name = ticker_cache.get(ticker, ticker)
            candidates.append({
                "ticker": ticker, "name": name,
                "price": round(close), "change_rate": round(change_rate, 2),
                "volume": int(today_vol), "amount": int(amount),
                "mktcap": 0, "vol_ratio": vol_ratio,
            })

        candidates.sort(key=lambda x: x["vol_ratio"], reverse=True)
        top_candidates = candidates[:min(120, len(candidates))]
        logger.info("pykrx 1차 필터: %d개", len(top_candidates))

        from scoring import calculate_score
        semaphore = asyncio.Semaphore(10)

        async def score_candidate(item):
            async with semaphore:
                try:
                    ohlcv = await asyncio.wait_for(get_ohlcv(item["ticker"], 60), timeout=15.0)
                    supply = await asyncio.wait_for(get_supply(item["ticker"], 20), timeout=10.0)
                    if ohlcv is not None and not ohlcv.empty:
                        mktcap = item.get("mktcap", 0) or 0
                        score_data = calculate_score(ohlcv, supply, mktcap=mktcap)
                        item["score"] = round(score_data["total"], 2)
                        item["grade"] = score_data["grade"]
                    else:
                        item["score"] = 0.0
                        item["grade"] = "D"
                except Exception:
                    item["score"] = 0.0
                    item["grade"] = "D"
                return item

        scored = await asyncio.gather(*[score_candidate(c) for c in top_candidates])
        final = [s for s in scored if isinstance(s, dict)]
        final.sort(key=lambda x: x.get("score", 0), reverse=True)
        return final[:max_results]

    except Exception as e:
        logger.error("pykrx 스캔 오류: %s", e)
        return []


async def get_stock_detail(ticker: str) -> Optional[dict]:
    ticker = str(ticker).zfill(6)
    name = ticker_cache.get(ticker, ticker)

    # KIS 현재가에서 이름 보완
    if name == ticker:
        try:
            from kis_data import get_current_price_kis
            info = await get_current_price_kis(ticker)
            if info and info.get("name"):
                name = info["name"]
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

    ohlcv = await get_ohlcv(ticker, 252)
    supply = await get_supply(ticker, 60)
    low52, high52 = await get_52week_range(ticker)

    # KIS 현재가로 52주 고저 보완
    if high52 == 0 or low52 == 0:
        try:
            from kis_data import get_current_price_kis
            info = await get_current_price_kis(ticker)
            if info:
                high52 = info.get("high52", 0) or high52
                low52 = info.get("low52", 0) or low52
        except Exception:
            pass

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

    mktcap_val = 0.0
    try:
        from kis_data import get_current_price_kis
        info2 = await get_current_price_kis(ticker)
        if info2:
            mktcap_val = info2.get("mktcap", 0) or 0
    except Exception:
        pass

    from scoring import calculate_score
    score_data = calculate_score(ohlcv, supply, mktcap=mktcap_val, low52=low52, high52=high52)

    ohlcv_list = []
    for idx, row in ohlcv.tail(60).iterrows():
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
