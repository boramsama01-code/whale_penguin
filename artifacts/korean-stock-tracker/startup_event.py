import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

ticker_cache: dict[str, str] = {}
market_state: dict = {
    "is_open": False,
    "kospi_status": "CLOSED",
    "kosdaq_status": "CLOSED",
}


async def load_ticker_cache():
    try:
        from pykrx import stock as pykrx_stock
        logger.info("pykrx 종목 캐시 로딩 중...")

        tickers = await asyncio.to_thread(pykrx_stock.get_market_ticker_list, market="ALL")
        for t in tickers:
            ticker = str(t).zfill(6)
            try:
                name = await asyncio.to_thread(pykrx_stock.get_market_ticker_name, ticker)
                if name:
                    ticker_cache[ticker] = name
            except Exception:
                ticker_cache[ticker] = ticker

        logger.info("종목 캐시 완료: %d개", len(ticker_cache))
    except Exception as e:
        logger.error("종목 캐시 로딩 실패: %s", e)
        _load_fallback_tickers()


def _load_fallback_tickers():
    fallback = {
        "005930": "삼성전자",
        "000660": "SK하이닉스",
        "373220": "LG에너지솔루션",
        "207940": "삼성바이오로직스",
        "005490": "POSCO홀딩스",
        "005380": "현대차",
        "051910": "LG화학",
        "006400": "삼성SDI",
        "035420": "NAVER",
        "000270": "기아",
        "035720": "카카오",
        "068270": "셀트리온",
        "028260": "삼성물산",
        "012330": "현대모비스",
        "066570": "LG전자",
        "009830": "한화솔루션",
        "096770": "SK이노베이션",
        "003670": "포스코퓨처엠",
        "010130": "고려아연",
        "017670": "SK텔레콤",
    }
    ticker_cache.update(fallback)
    logger.info("Fallback 종목 캐시 로딩: %d개", len(fallback))


async def load_dart_corp_codes():
    try:
        from dart_client import load_corp_codes
        await load_corp_codes()
        logger.info("DART corp code 캐시 완료")
    except Exception as e:
        logger.error("DART corp code 로딩 실패: %s", e)


async def init_market_state():
    try:
        from market_filter import update_market_state
        await update_market_state()
        logger.info("시장 상태 초기화 완료: %s", market_state)
    except Exception as e:
        logger.error("시장 상태 초기화 실패: %s", e)


async def on_startup():
    logger.info("앱 시작 이벤트 실행 중...")
    _load_fallback_tickers()
    # 모든 무거운 작업을 백그라운드로 실행 (서버 포트 오픈을 블로킹하지 않음)
    asyncio.create_task(_background_init())
    logger.info("앱 시작 이벤트 완료 (백그라운드 초기화 진행 중)")


async def _background_init():
    await asyncio.gather(
        load_dart_corp_codes(),
        init_market_state(),
        return_exceptions=True,
    )
    asyncio.create_task(load_ticker_cache())
    logger.info("백그라운드 초기화 완료")


async def on_shutdown():
    logger.info("앱 종료 이벤트 실행 중...")
    from kis_realtime import stop_ws
    await stop_ws()
    logger.info("앱 종료 완료")
