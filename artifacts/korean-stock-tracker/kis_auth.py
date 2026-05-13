import os
import asyncio
import time
import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"

_token_cache: dict = {
    "access_token": None,
    "expires_at": 0,
    "approval_key": None,
}
_token_lock = asyncio.Lock()


def _is_prod() -> bool:
    return os.getenv("KIS_ENV", "PROD").upper() == "PROD"


async def get_access_token(force_refresh: bool = False) -> str:
    async with _token_lock:
        now = time.time()
        if (
            not force_refresh
            and _token_cache["access_token"]
            and now < _token_cache["expires_at"] - 60
        ):
            return _token_cache["access_token"]

        app_key = os.getenv("KIS_APP_KEY", "")
        app_secret = os.getenv("KIS_APP_SECRET", "")

        if not app_key or not app_secret:
            raise RuntimeError("KIS_APP_KEY 또는 KIS_APP_SECRET 환경변수가 없습니다.")

        payload = {
            "grant_type": "client_credentials",
            "appkey": app_key,
            "appsecret": app_secret,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{KIS_BASE_URL}/oauth2/tokenP",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()

            token = data.get("access_token")
            expires_in = int(data.get("expires_in", 86400))

            if not token:
                raise RuntimeError(f"토큰 발급 실패: {data}")

            _token_cache["access_token"] = token
            _token_cache["expires_at"] = now + expires_in
            logger.info("KIS access token 갱신 완료 (만료: %ds)", expires_in)
            return token

        except httpx.HTTPStatusError as e:
            logger.error("KIS 토큰 발급 HTTP 오류: %s", e)
            raise
        except Exception as e:
            logger.error("KIS 토큰 발급 오류: %s", e)
            raise


async def get_approval_key() -> str:
    if _token_cache.get("approval_key"):
        return _token_cache["approval_key"]

    app_key = os.getenv("KIS_APP_KEY", "")
    app_secret = os.getenv("KIS_APP_SECRET", "")

    payload = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{KIS_BASE_URL}/oauth2/Approval",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        key = data.get("approval_key")
        if not key:
            raise RuntimeError(f"Approval Key 발급 실패: {data}")

        _token_cache["approval_key"] = key
        logger.info("KIS Approval Key 발급 완료")
        return key

    except Exception as e:
        logger.error("Approval Key 발급 오류: %s", e)
        raise


async def kis_get(path: str, params: dict, retry: bool = True) -> dict:
    token = await get_access_token()
    app_key = os.getenv("KIS_APP_KEY", "")
    app_secret = os.getenv("KIS_APP_SECRET", "")

    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": app_key,
        "appsecret": app_secret,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{KIS_BASE_URL}{path}",
                params=params,
                headers=headers,
            )

            if resp.status_code == 401 and retry:
                logger.warning("401 발생, 토큰 재발급 후 재시도")
                await get_access_token(force_refresh=True)
                return await kis_get(path, params, retry=False)

            resp.raise_for_status()
            return resp.json()

    except httpx.HTTPStatusError as e:
        logger.error("KIS API HTTP 오류 %s: %s", path, e)
        raise
    except Exception as e:
        logger.error("KIS API 오류 %s: %s", path, e)
        raise


async def token_refresh_loop():
    while True:
        try:
            await asyncio.sleep(3600)
            await get_access_token(force_refresh=True)
        except Exception as e:
            logger.error("토큰 자동 갱신 실패: %s", e)
