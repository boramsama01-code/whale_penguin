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


async def _fetch_index_kis(index_code: str) -> Optional[dict]:
    """KIS REST API로 지수 현재값 조회."""
    try:
        from kis_data import get_index_kis
        return await get_index_kis(index_code)
    except Exception as e:
        logger.debug("KIS 지수 조회 실패 %s: %s", index_code, e)
        return None


async def update_market_state():
    global _market_state
    _market_state["is_open"] = is_market_open()

    # KIS REST API로 KOSPI(0001), KOSDAQ(1001) 지수 동시 조회
    kospi_data, kosdaq_data = await asyncio.gather(
        _fetch_index_kis("0001"),
        _fetch_index_kis("1001"),
        return_exceptions=True,
    )

    status = "OPEN" if is_market_open() else "CLOSED"

    # KIS 실패 시 네이버 금융 fallback
    from kis_data import get_index_naver

    if not (isinstance(kospi_data, dict) and kospi_data.get("index", 0) > 0):
        logger.warning("KOSPI KIS 조회 실패 — 네이버 fallback 시도")
        try:
            kospi_data = await get_index_naver("KOSPI")
        except Exception:
            kospi_data = None

    if not (isinstance(kosdaq_data, dict) and kosdaq_data.get("index", 0) > 0):
        logger.warning("KOSDAQ KIS 조회 실패 — 네이버 fallback 시도")
        try:
            kosdaq_data = await get_index_naver("KOSDAQ")
        except Exception:
            kosdaq_data = None

    if isinstance(kospi_data, dict) and kospi_data.get("index", 0) > 0:
        idx = float(kospi_data["index"])
        _market_state["kospi"]["index"] = idx
        _market_state["kospi"]["change"] = float(kospi_data.get("change", 0))
        _market_state["kospi"]["change_rate"] = float(kospi_data.get("change_rate", 0))
        _market_state["kospi"]["status"] = status
        _market_state["kospi"]["ma20_bullish"] = float(kospi_data.get("change_rate", 0)) >= 0
    else:
        logger.warning("KOSPI 지수 조회 최종 실패")
        _market_state["kospi"]["status"] = status

    if isinstance(kosdaq_data, dict) and kosdaq_data.get("index", 0) > 0:
        idx = float(kosdaq_data["index"])
        _market_state["kosdaq"]["index"] = idx
        _market_state["kosdaq"]["change"] = float(kosdaq_data.get("change", 0))
        _market_state["kosdaq"]["change_rate"] = float(kosdaq_data.get("change_rate", 0))
        _market_state["kosdaq"]["status"] = status
        _market_state["kosdaq"]["ma20_bullish"] = float(kosdaq_data.get("change_rate", 0)) >= 0
    else:
        logger.warning("KOSDAQ 지수 조회 최종 실패")
        _market_state["kosdaq"]["status"] = status

    kospi_bull = _market_state["kospi"]["ma20_bullish"]
    kosdaq_bull = _market_state["kosdaq"]["ma20_bullish"]
    _market_state["strategy_on"] = kospi_bull or kosdaq_bull

    if _market_state["kospi"]["index"] > 0 or _market_state["kosdaq"]["index"] > 0:
        logger.info(
            "시장 상태 갱신 — KOSPI: %.2f, KOSDAQ: %.2f, 장중: %s",
            _market_state["kospi"]["index"],
            _market_state["kosdaq"]["index"],
            _market_state["is_open"],
        )
