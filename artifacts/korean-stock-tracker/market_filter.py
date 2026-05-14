import asyncio
import logging
from datetime import datetime, time as dtime, timezone, timedelta
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

MARKET_OPEN = dtime(9, 0)
MARKET_CLOSE = dtime(15, 30)
KST = timezone(timedelta(hours=9))

_market_state: dict = {
    "is_open": False,
    "kospi": {"status": "CLOSED", "ma20_bullish": False, "index": 0.0},
    "kosdaq": {"status": "CLOSED", "ma20_bullish": False, "index": 0.0},
    "strategy_on": False,
}


def now_kst() -> datetime:
    return datetime.now(KST)


def is_market_open() -> bool:
    kst = now_kst()
    if kst.weekday() >= 5:
        return False
    return MARKET_OPEN <= kst.time() <= MARKET_CLOSE


def get_market_state() -> dict:
    _market_state["is_open"] = is_market_open()
    return _market_state


async def update_market_state():
    global _market_state
    _market_state["is_open"] = is_market_open()

    try:
        import pandas as pd
        from pykrx import stock as pykrx_stock
        from datetime import timedelta

        end = now_kst().strftime("%Y%m%d")
        start = (now_kst() - timedelta(days=60)).strftime("%Y%m%d")

        try:
            kospi_df = await asyncio.to_thread(
                pykrx_stock.get_index_ohlcv_by_date, start, end, "1001"
            )
            if kospi_df is not None and len(kospi_df) >= 1:
                close_col = "종가" if "종가" in kospi_df.columns else kospi_df.columns[3]
                close = kospi_df[close_col].values.astype(float)
                ma20 = float(np.mean(close[-min(20, len(close)):]))
                current = float(close[-1])
                _market_state["kospi"]["index"] = current
                _market_state["kospi"]["ma20_bullish"] = current > ma20
                _market_state["kospi"]["status"] = "OPEN" if is_market_open() else "CLOSED"
        except Exception as e:
            logger.warning("KOSPI 지수 조회 실패: %s", str(e))

        try:
            kosdaq_df = await asyncio.to_thread(
                pykrx_stock.get_index_ohlcv_by_date, start, end, "2001"
            )
            if kosdaq_df is not None and len(kosdaq_df) >= 1:
                close_col = "종가" if "종가" in kosdaq_df.columns else kosdaq_df.columns[3]
                close = kosdaq_df[close_col].values.astype(float)
                ma20 = float(np.mean(close[-min(20, len(close)):]))
                current = float(close[-1])
                _market_state["kosdaq"]["index"] = current
                _market_state["kosdaq"]["ma20_bullish"] = current > ma20
                _market_state["kosdaq"]["status"] = "OPEN" if is_market_open() else "CLOSED"
        except Exception as e:
            logger.warning("KOSDAQ 지수 조회 실패: %s", str(e))

        kospi_bull = _market_state["kospi"]["ma20_bullish"]
        kosdaq_bull = _market_state["kosdaq"]["ma20_bullish"]
        _market_state["strategy_on"] = kospi_bull or kosdaq_bull

    except Exception as e:
        logger.error("시장 상태 업데이트 오류: %s", e)
        _market_state["is_open"] = is_market_open()
