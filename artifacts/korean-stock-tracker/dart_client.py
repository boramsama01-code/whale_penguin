import os
import io
import json
import zipfile
import logging
import asyncio
import httpx
from xml.etree import ElementTree as ET
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

DART_BASE = "https://opendart.fss.or.kr/api"
CORP_CODES_PATH = Path(__file__).parent / "cache" / "corp_codes.json"
TICKER_NAMES_PATH = Path(__file__).parent / "cache" / "ticker_names.json"

_corp_code_cache: dict[str, str] = {}   # stock_code -> corp_code
_ticker_name_cache: dict[str, str] = {}  # stock_code -> corp_name


async def load_corp_codes():
    global _corp_code_cache, _ticker_name_cache

    # 네임 캐시 파일 먼저 로드
    if TICKER_NAMES_PATH.exists():
        try:
            with open(TICKER_NAMES_PATH, "r", encoding="utf-8") as f:
                _ticker_name_cache = json.load(f)
            logger.info("ticker_names 캐시 로드: %d개", len(_ticker_name_cache))
        except Exception as e:
            logger.warning("ticker_names 캐시 읽기 실패: %s", e)

    if CORP_CODES_PATH.exists():
        try:
            with open(CORP_CODES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 구형 포맷 (ticker: corp_code) vs 신형 (ticker: {corp_code, name})
            if data and isinstance(list(data.values())[0], str):
                _corp_code_cache = data
                logger.info("DART corp codes 캐시 로드: %d개", len(_corp_code_cache))
                # 구형 포맷 → 이름 없음, 백그라운드에서 재다운로드
                asyncio.create_task(_fetch_corp_codes())
            else:
                for ticker, val in data.items():
                    _corp_code_cache[ticker] = val.get("corp_code", "")
                    if val.get("name") and ticker not in _ticker_name_cache:
                        _ticker_name_cache[ticker] = val["name"]
                logger.info("DART corp codes(신형) 캐시 로드: %d개", len(_corp_code_cache))
            return
        except Exception as e:
            logger.warning("corp codes 캐시 파일 읽기 실패: %s", e)

    await _fetch_corp_codes()


async def _fetch_corp_codes():
    global _corp_code_cache, _ticker_name_cache
    api_key = os.getenv("DART_API_KEY") or os.getenv("OPEN_DART", "")
    if not api_key:
        logger.warning("DART_API_KEY 없음 — corp codes 미로드")
        return

    try:
        logger.info("DART corpCode.xml 다운로드 중...")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                f"{DART_BASE}/corpCode.xml",
                params={"crtfc_key": api_key},
            )
            resp.raise_for_status()
            raw = resp.content

        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            xml_name = [n for n in zf.namelist() if n.endswith(".xml")][0]
            xml_content = zf.read(xml_name)

        root = ET.fromstring(xml_content)
        corp_codes: dict[str, str] = {}
        names: dict[str, str] = {}

        for item in root.findall(".//list"):
            corp_code = item.findtext("corp_code", "").strip()
            stock_code = item.findtext("stock_code", "").strip()
            corp_name = item.findtext("corp_name", "").strip()
            if stock_code and corp_code:
                stock_code = str(stock_code).zfill(6)
                corp_codes[stock_code] = corp_code
                if corp_name:
                    names[stock_code] = corp_name

        _corp_code_cache = corp_codes
        _ticker_name_cache.update(names)

        CORP_CODES_PATH.parent.mkdir(exist_ok=True)
        with open(CORP_CODES_PATH, "w", encoding="utf-8") as f:
            json.dump(corp_codes, f, ensure_ascii=False)
        with open(TICKER_NAMES_PATH, "w", encoding="utf-8") as f:
            json.dump(names, f, ensure_ascii=False)

        logger.info("DART corp codes 다운로드 완료: %d개 (이름 %d개)", len(corp_codes), len(names))

    except Exception as e:
        logger.error("DART corp codes 로드 실패: %s", e)


def get_corp_code(ticker: str) -> Optional[str]:
    ticker = str(ticker).zfill(6)
    return _corp_code_cache.get(ticker)


def get_all_ticker_names() -> dict[str, str]:
    """DART에서 로드된 ticker -> 회사명 딕셔너리 반환"""
    return _ticker_name_cache


async def get_disclosures(ticker: str, limit: int = 10) -> list[dict]:
    ticker = str(ticker).zfill(6)
    corp_code = get_corp_code(ticker)
    if not corp_code:
        logger.warning("ticker %s의 corp_code 없음", ticker)
        return []

    api_key = os.getenv("DART_API_KEY") or os.getenv("OPEN_DART", "")
    if not api_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{DART_BASE}/list.json",
                params={
                    "crtfc_key": api_key,
                    "corp_code": corp_code,
                    "bgn_de": _days_ago(90),
                    "last_reprt_at": "Y",
                    "pblntf_ty": "A",
                    "page_count": limit,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") != "000":
            logger.warning("DART API 오류: %s", data.get("message"))
            return []

        items = data.get("list", [])
        result = []
        for item in items:
            title = item.get("report_nm", "")
            sentiment = _classify_disclosure(title)
            result.append({
                "rcept_no": item.get("rcept_no", ""),
                "corp_name": item.get("corp_name", ""),
                "report_nm": title,
                "rcept_dt": item.get("rcept_dt", ""),
                "sentiment": sentiment,
                "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item.get('rcept_no', '')}",
            })
        return result

    except Exception as e:
        logger.error("DART 공시 조회 오류 (%s): %s", ticker, e)
        return []


def _classify_disclosure(title: str) -> str:
    positive_keywords = ["자사주", "공급계약", "수주", "실적", "상향", "배당", "취득"]
    caution_keywords = ["유상증자", "전환사채", "CB발행", "최대주주변경", "CB", "BW"]
    for kw in positive_keywords:
        if kw in title:
            return "POSITIVE"
    for kw in caution_keywords:
        if kw in title:
            return "CAUTION"
    return "NEUTRAL"


def _days_ago(days: int) -> str:
    from datetime import datetime, timedelta
    dt = datetime.now() - timedelta(days=days)
    return dt.strftime("%Y%m%d")
