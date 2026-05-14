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
        app_key = os.getenv("KIS_APP_KEY", "")
        app_secret = os.getenv("KIS_APP_SECRET", "")
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


async def get_ohlcv_kis(ticker: str, days: int = 252) -> Optional[pd.DataFrame]:
    """
    KIS API로 일별 OHLCV 조회.
    columns: 시가, 고가, 저가, 종가, 거래량, 거래대금
    """
    ticker = str(ticker).zfill(6)
    now = _now_kst()
    end = now.strftime("%Y%m%d")
    start = (now - timedelta(days=days + 60)).strftime("%Y%m%d")

    try:
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
            logger.debug("KIS OHLCV 응답 오류 %s: %s", ticker, data.get("msg1") if data else "no data")
            return None

        output2 = data.get("output2", [])
        if not output2:
            return None

        rows = []
        for item in output2:
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

        if not rows:
            return None

        df = pd.DataFrame(rows).set_index("date").sort_index()
        return df.tail(days)

    except Exception as e:
        logger.debug("KIS OHLCV 오류 %s: %s", ticker, e)
        return None


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
            # 가능한 모든 필드명 시도 (KIS API 버전에 따라 다름)
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
            # HTTP 200이지만 파싱 실패 — 전체 응답 키 로그
            logger.info("KIS 지수 API 파싱 실패 %s: rt_cd=%s, output_keys=%s, msg=%s",
                        index_code, rt, list(o.keys())[:15], data.get("msg1", ""))
    except Exception as e:
        logger.info("KIS 지수 API 예외 %s: %s", index_code, e)

    # 2차 fallback: 대표 ETF 현재가로 지수 추정
    # KODEX 200(069500) NAV ≈ KOSPI / 100.0 (누적 스플릿 감안 약 100배)
    # KODEX KOSDAQ150(229200) NAV ≈ KOSDAQ / 10.0 (약 10배)
    etf_map = {"0001": ("069500", 100.0), "1001": ("229200", 10.0)}
    etf_ticker, scale = etf_map.get(index_code, (None, 1.0))
    if etf_ticker:
        try:
            info = await get_current_price_kis(etf_ticker)
            if info and info.get("price", 0) > 0:
                estimated = round(info["price"] * scale, 2)
                logger.info("KIS 지수 ETF fallback %s → %.2f (ETF×%.0f)", index_code, estimated, scale)
                return {
                    "index": estimated,
                    "change_rate": info.get("change_rate", 0),
                    "estimated": True,
                }
        except Exception as e:
            logger.debug("KIS 지수 ETF fallback 실패 %s: %s", index_code, e)

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
                results.append({
                    "ticker": ticker,
                    "name": item.get("hts_kor_isnm", ""),
                    "price": price,
                    "change_rate": float(item.get("prdy_ctrt", 0) or 0),
                    "volume": int(float(item.get("acml_vol", 0) or 0)),
                    "amount": amount,
                    "vol_ratio": float(item.get("vol_inrt", 0) or 1.0),
                })
            except Exception:
                continue
        return results
    except Exception as e:
        logger.debug("KIS 거래량 순위 오류: %s", e)
        return []
