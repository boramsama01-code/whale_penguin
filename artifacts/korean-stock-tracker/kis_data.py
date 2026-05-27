"""
KIS REST API 데이터 모듈.
pykrx 대체 — Replit 해외 IP에서도 정상 동작.
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))


def _now_kst() -> datetime:
    return datetime.now(KST)


async def _kis_get(path: str, params: dict, tr_id: str) -> Optional[dict]:
    """KIS REST API 인증 GET 요청."""
    try:
        from kis_auth import get_access_token, KIS_BASE_URL
        import httpx
        token = await get_access_token()
        app_key = os.getenv("KIS_APP_KEY") or os.getenv("KIS_Developers_app_key", "")
        app_secret = os.getenv("KIS_APP_SECRET") or os.getenv("KIS_Developers_app_secret", "")
        if not app_key or not app_secret:
            return None
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": app_key,
            "appsecret": app_secret,
            "tr_id": tr_id,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{KIS_BASE_URL}{path}",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.debug("KIS API 실패 %s: %s", path, e)
        return None


async def _fetch_ohlcv_range(ticker: str, start: str, end: str) -> list[dict]:
    """단일 날짜 범위의 OHLCV 조회 (최대 ~100건)."""
    data = await _kis_get(
        "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
        params={
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": start,
            "FID_INPUT_DATE_2": end,
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        },
        tr_id="FHKST03010100",
    )

    if not data or data.get("rt_cd") != "0":
        return []

    rows = []
    for item in (data.get("output2") or []):
        try:
            date_str = item.get("stck_bsop_date", "")
            if not date_str or len(date_str) != 8:
                continue
            close = float(item.get("stck_clpr", 0) or 0)
            if close <= 0:
                continue
            rows.append({
                "date": pd.to_datetime(date_str, format="%Y%m%d"),
                "시가": float(item.get("stck_oprc", 0) or 0),
                "고가": float(item.get("stck_hgpr", 0) or 0),
                "저가": float(item.get("stck_lwpr", 0) or 0),
                "종가": close,
                "거래량": float(item.get("acml_vol", 0) or 0),
                "거래대금": float(item.get("acml_tr_pbmn", 0) or 0),
            })
        except Exception:
            continue
    return rows


async def get_ohlcv_kis(ticker: str, days: int = 252) -> Optional[pd.DataFrame]:
    """
    KIS API로 일별 OHLCV 조회 — 90일 단위 페이지네이션으로 최대 1년 데이터 수집.
    columns: 시가, 고가, 저가, 종가, 거래량, 거래대금
    """
    ticker = str(ticker).zfill(6)
    now = _now_kst()

    # 90일 단위로 분할 요청 (KIS API 응답 한계 극복)
    chunk_days = 90
    num_chunks = (days + chunk_days - 1) // chunk_days  # 올림 나누기
    num_chunks = min(num_chunks, 5)  # 최대 5회 요청 (450일)

    all_rows: list[dict] = []
    tasks = []
    for i in range(num_chunks):
        end_dt = now - timedelta(days=i * chunk_days)
        start_dt = end_dt - timedelta(days=chunk_days + 10)
        tasks.append(_fetch_ohlcv_range(
            ticker,
            start_dt.strftime("%Y%m%d"),
            end_dt.strftime("%Y%m%d"),
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, list):
            all_rows.extend(r)

    if not all_rows:
        logger.debug("KIS OHLCV 데이터 없음: %s", ticker)
        return None

    df = pd.DataFrame(all_rows).set_index("date")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df.tail(days)


async def get_current_price_kis(ticker: str) -> Optional[dict]:
    """KIS API로 현재가 + 52주 고저가 + 시가총액 조회."""
    ticker = str(ticker).zfill(6)
    try:
        data = await _kis_get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": ticker,
            },
            tr_id="FHKST01010100",
        )

        if not data or data.get("rt_cd") != "0":
            return None

        o = data.get("output", {})
        mktcap_uk = float(o.get("hts_avls", 0) or 0)  # 단위: 억원
        return {
            "price": float(o.get("stck_prpr", 0) or 0),
            "change_rate": float(o.get("prdy_ctrt", 0) or 0),
            "volume": int(float(o.get("acml_vol", 0) or 0)),
            "amount": float(o.get("acml_tr_pbmn", 0) or 0),
            "high52": float(o.get("w52_hgpr", 0) or 0),
            "low52": float(o.get("w52_lwpr", 0) or 0),
            "mktcap": mktcap_uk * 100_000_000,
            "name": o.get("hts_kor_isnm", ""),
            "per": float(o.get("per", 0) or 0),
            "pbr": float(o.get("pbr", 0) or 0),
        }
    except Exception as e:
        logger.debug("KIS 현재가 오류 %s: %s", ticker, e)
        return None


async def get_index_kis(index_code: str = "0001") -> Optional[dict]:
    """
    KIS API로 지수 현재가 조회.
    index_code: "0001"=KOSPI, "1001"=KOSDAQ

    업종 시세 API (FHPUP03500100) 우선, 실패 시 ETF 가격으로 추정.
    - KOSPI  → KODEX 200 (069500) × 스케일
    - KOSDAQ → KODEX KOSDAQ150 (229200) × 스케일
    """
    # 1차: KIS 업종 현재 시세 API
    try:
        data = await _kis_get(
            "/uapi/domestic-stock/v1/quotations/inquire-index-price",
            params={
                "FID_COND_MRKT_DIV_CODE": "U",
                "FID_INPUT_ISCD": index_code,
            },
            tr_id="FHPUP03500100",
        )
        if data:
            rt = data.get("rt_cd", "")
            o = data.get("output", {})
            idx_val = float(
                o.get("bstp_nmix_prpr") or
                o.get("bstp_nmix_prdy_vrss") or
                o.get("futs_prpr") or
                0
            )
            if rt == "0" and idx_val > 0:
                change_rate = float(
                    o.get("bstp_nmix_prdy_ctrt") or
                    o.get("prdy_ctrt") or
                    0
                )
                return {"index": idx_val, "change_rate": change_rate}
            logger.info("KIS 지수 API 파싱 실패 %s: rt_cd=%s, output_keys=%s, msg=%s",
                        index_code, rt, list(o.keys())[:15], data.get("msg1", ""))
    except Exception as e:
        logger.info("KIS 지수 API 예외 %s: %s", index_code, e)

    return None


async def get_volume_rank_kis(market: str = "J", top_n: int = 100) -> list[dict]:
    """
    KIS API 거래량 순위 조회 (스캐너용).
    market: "J"=KOSPI, "Q"=KOSDAQ
    """
    try:
        data = await _kis_get(
            "/uapi/domestic-stock/v1/quotations/volume-rank",
            params={
                "FID_COND_MRKT_DIV_CODE": market,
                "FID_COND_SCR_DIV_CODE": "20171",
                "FID_INPUT_ISCD": "0000",
                "FID_DIV_CLS_CODE": "0",
                "FID_BLNG_CLS_CODE": "0",
                "FID_TRGT_CLS_CODE": "111111111",
                "FID_TRGT_EXLS_CLS_CODE": "000000",
                "FID_INPUT_PRICE_1": "",
                "FID_INPUT_PRICE_2": "",
                "FID_VOL_CNT": "",
                "FID_INPUT_DATE_1": "",
            },
            tr_id="FHPST01710000",
        )

        if not data or data.get("rt_cd") != "0":
            return []

        output = data.get("output", [])
        results = []
        for item in (output or [])[:top_n]:
            try:
                ticker = str(item.get("mksc_shrn_iscd", "")).zfill(6)
                price = float(item.get("stck_prpr", 0) or 0)
                amount = float(item.get("acml_tr_pbmn", 0) or 0)
                if price <= 0 or amount < 100_000_000:
                    continue
                mktcap_uk = float(item.get("hts_avls", 0) or 0)  # 억원
                results.append({
                    "ticker": ticker,
                    "name": item.get("hts_kor_isnm", ""),
                    "price": price,
                    "change_rate": float(item.get("prdy_ctrt", 0) or 0),
                    "volume": int(float(item.get("acml_vol", 0) or 0)),
                    "amount": amount,
                    "vol_ratio": float(item.get("vol_inrt", 0) or 1.0),
                    "mktcap": int(mktcap_uk * 100_000_000),
                })
            except Exception:
                continue
        return results
    except Exception as e:
        logger.debug("KIS 거래량 순위 오류: %s", e)
        return []


async def get_index_naver(symbol: str = "KOSPI") -> Optional[dict]:
    """네이버 금융 모바일 API로 지수 조회 (KIS 대체)."""
    try:
        import httpx
        url = f"https://m.stock.naver.com/api/index/{symbol}/basic"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; KRX-Tracker/1.0)"},
            )
            resp.raise_for_status()
            data = resp.json()
        def _f(key: str) -> float:
            try:
                return float(str(data.get(key, "0")).replace(",", ""))
            except Exception:
                return 0.0
        idx = _f("closePrice")
        rate = _f("fluctuationsRatio")
        chg = _f("compareToPreviousClosePrice")
        if idx > 0:
            logger.info("네이버 지수 조회 성공 %s: %.2f (%.2f%%)", symbol, idx, rate)
            return {"index": idx, "change_rate": rate, "change": chg}
        return None
    except Exception as e:
        logger.debug("네이버 지수 조회 실패 %s: %s", symbol, e)
        return None
