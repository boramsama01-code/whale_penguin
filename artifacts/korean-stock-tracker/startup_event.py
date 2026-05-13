import asyncio
import logging
import os
import json
from typing import Optional

logger = logging.getLogger(__name__)

ticker_cache: dict[str, str] = {}

CACHE_FILE = os.path.join(os.path.dirname(__file__), "cache", "ticker_cache.json")

_FALLBACK = {
    "005930": "삼성전자", "000660": "SK하이닉스", "373220": "LG에너지솔루션",
    "207940": "삼성바이오로직스", "005490": "POSCO홀딩스", "005380": "현대차",
    "051910": "LG화학", "006400": "삼성SDI", "035420": "NAVER", "000270": "기아",
    "035720": "카카오", "068270": "셀트리온", "028260": "삼성물산",
    "012330": "현대모비스", "066570": "LG전자", "009830": "한화솔루션",
    "096770": "SK이노베이션", "003670": "포스코퓨처엠", "010130": "고려아연",
    "017670": "SK텔레콤", "089860": "롯데렌탈", "253450": "스튜디오드래곤",
    "041510": "에스엠", "035900": "JYP Ent.", "122870": "와이지엔터테인먼트",
}


def _load_from_disk() -> bool:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                ticker_cache.update(data)
                logger.info("디스크 종목 캐시 로딩: %d개", len(ticker_cache))
                return True
    except Exception as e:
        logger.error("디스크 캐시 로딩 실패: %s", e)
    return False


def _save_to_disk():
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(ticker_cache, f, ensure_ascii=False)
        logger.info("종목 캐시 디스크 저장: %d개", len(ticker_cache))
    except Exception as e:
        logger.error("디스크 캐시 저장 실패: %s", e)


async def load_dart_corp_codes():
    """DART corp codes + 회사명 로드. ticker_names.json에서 ticker_cache도 채움."""
    try:
        from dart_client import load_corp_codes, get_all_ticker_names
        await load_corp_codes()

        # DART 이름 → ticker_cache에 병합
        dart_names = get_all_ticker_names()
        if dart_names:
            before = len(ticker_cache)
            for t, name in dart_names.items():
                if t not in ticker_cache or ticker_cache[t] == t:
                    ticker_cache[t] = name
            logger.info("DART 회사명 → ticker_cache 병합: %d개 → %d개", before, len(ticker_cache))
            _save_to_disk()

        logger.info("DART corp code 캐시 완료")
    except Exception as e:
        logger.error("DART corp code 로딩 실패: %s", e)


async def load_ticker_names_from_pykrx():
    """
    pykrx get_market_ticker_name 으로 DART 종목 이름 보강.
    DART 이름은 법인명 기준이라 KRX 표준 이름과 다를 수 있음.
    ticker_cache < 500개일 때만 실행.
    """
    from dart_client import get_corp_code
    from pykrx import stock as pykrx_stock

    tickers_to_fetch = [t for t in ticker_cache if ticker_cache[t] == t][:200]
    if not tickers_to_fetch:
        logger.info("pykrx 이름 보강 불필요 (이름 모두 있음)")
        return

    logger.info("pykrx 이름 보강 시작: %d개 ticker", len(tickers_to_fetch))
    semaphore = asyncio.Semaphore(10)
    updated = 0

    async def fetch_one(t):
        nonlocal updated
        async with semaphore:
            try:
                name = await asyncio.to_thread(pykrx_stock.get_market_ticker_name, t)
                if name and name != t:
                    ticker_cache[t] = name
                    updated += 1
            except Exception:
                pass

    await asyncio.gather(*[fetch_one(t) for t in tickers_to_fetch], return_exceptions=True)
    logger.info("pykrx 이름 보강 완료: %d개 업데이트", updated)
    if updated > 0:
        _save_to_disk()


async def init_market_state():
    try:
        from market_filter import update_market_state
        await update_market_state()
    except Exception as e:
        logger.error("시장 상태 초기화 실패: %s", e)


async def on_startup():
    logger.info("앱 시작 이벤트 실행 중...")

    # 즉시: fallback + 디스크 캐시 로딩
    ticker_cache.update(_FALLBACK)
    _load_from_disk()

    # 백그라운드: DART, pykrx 보강, 시장 상태
    asyncio.create_task(_background_init())
    logger.info("앱 시작 이벤트 완료 (백그라운드 초기화 진행 중)")


async def _background_init():
    # DART + 시장 상태 병렬
    await asyncio.gather(
        load_dart_corp_codes(),
        init_market_state(),
        return_exceptions=True,
    )
    logger.info("백그라운드 초기화 완료 (ticker_cache: %d개)", len(ticker_cache))


async def on_shutdown():
    logger.info("앱 종료 이벤트 실행 중...")
    from kis_realtime import stop_ws
    await stop_ws()
    logger.info("앱 종료 완료")
