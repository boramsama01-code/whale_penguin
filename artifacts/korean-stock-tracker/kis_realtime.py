import os
import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Optional
import websockets

logger = logging.getLogger(__name__)

_ws_connection = None
_ws_task = None
_running = False
_subscribed_tickers: set[str] = set()
_whale_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

_whale_accumulator: dict[str, dict] = defaultdict(lambda: {
    "total_amount": 0,
    "events": [],
    "last_reset": time.time(),
})

# 종목별 틱 거래량 통계 (상대적 이상거래량 감지용)
_ticker_tick_stats: dict[str, dict] = defaultdict(lambda: {
    "tick_count": 0,
    "vol_sum": 0.0,
    "vol_sq_sum": 0.0,
    "avg_vol": 0.0,
    "std_vol": 0.0,
})

RECONNECT_DELAY = 5
MAX_RECONNECT = 10


def get_ws_url() -> str:
    env = os.getenv("KIS_ENV", "PROD").upper()
    if env == "PROD":
        return "ws://ops.koreainvestment.com:21000"
    return "ws://ops.koreainvestment.com:31000"


def _parse_pipe_message(msg: str) -> Optional[dict]:
    try:
        parts = msg.split("|")
        if len(parts) < 4:
            return None

        tr_id = parts[1]
        body = parts[3]

        if tr_id == "H0STCNT0":
            fields = body.split("^")
            if len(fields) < 15:
                return None

            ticker = str(fields[0]).zfill(6)
            price = float(fields[2]) if fields[2] else 0.0
            volume = float(fields[9]) if fields[9] else 0.0
            amount = price * volume

            return {
                "type": "execution",
                "ticker": ticker,
                "price": price,
                "volume": int(volume),
                "amount": amount,
                "time": fields[1] if len(fields) > 1 else "",
            }
        return None
    except Exception as e:
        logger.debug("메시지 파싱 오류: %s | msg: %s", e, msg[:100])
        return None


def _update_tick_stats(ticker: str, volume: float):
    """틱 거래량 통계 업데이트 (온라인 알고리즘)"""
    stats = _ticker_tick_stats[ticker]
    stats["tick_count"] += 1
    n = stats["tick_count"]
    old_avg = stats["avg_vol"]
    stats["vol_sum"] += volume
    stats["avg_vol"] = stats["vol_sum"] / n
    # Welford's online variance
    stats["vol_sq_sum"] += (volume - old_avg) * (volume - stats["avg_vol"])
    if n > 1:
        stats["std_vol"] = (stats["vol_sq_sum"] / (n - 1)) ** 0.5


def _classify_whale_by_volume(ticker: str, volume: float, price: float) -> Optional[str]:
    """
    거래량 이상 기준 고래 분류 (절대 금액이 아닌 상대 거래량 기준)
    - 틱 수가 충분히 쌓이면 z-score 기반
    - 초기엔 절대 금액 기반 폴백
    """
    _update_tick_stats(ticker, volume)
    stats = _ticker_tick_stats[ticker]
    n = stats["tick_count"]

    if n >= 20 and stats["avg_vol"] > 0:
        avg = stats["avg_vol"]
        std = stats["std_vol"] if stats["std_vol"] > 0 else avg * 0.5
        z = (volume - avg) / std

        # z-score 기반 분류
        if z >= 8:
            return "LARGE"
        elif z >= 5:
            return "MEDIUM"
        elif z >= 3:
            return "SMALL"
        return None
    else:
        # 초기 틱 — 거래량 배율 기반 (첫 5틱은 건너뜀)
        if n < 5:
            return None
        avg = stats["avg_vol"] if stats["avg_vol"] > 0 else 1
        ratio = volume / avg
        if ratio >= 20:
            return "LARGE"
        elif ratio >= 8:
            return "MEDIUM"
        elif ratio >= 4:
            return "SMALL"

        # 거래량 기반으로 판단 안 되면 절대 금액 폴백
        amount = price * volume
        if amount >= 50_000_000:
            return "LARGE"
        elif amount >= 10_000_000:
            return "MEDIUM"
        elif amount >= 3_000_000:
            return "SMALL"
        return None


async def _process_execution(data: dict):
    ticker = data["ticker"]
    volume = float(data["volume"])
    price = data["price"]
    amount = data["amount"]

    now = time.time()
    acc = _whale_accumulator[ticker]

    if now - acc["last_reset"] > 300:
        acc["total_amount"] = 0
        acc["events"] = []
        acc["last_reset"] = now

    level = _classify_whale_by_volume(ticker, volume, price)
    if level:
        acc["total_amount"] += amount
        acc["events"].append({"time": data["time"], "amount": amount, "level": level, "volume": int(volume)})

        is_emergency = acc["total_amount"] >= 500_000_000

        from startup_event import ticker_cache
        name = ticker_cache.get(ticker, ticker)

        stats = _ticker_tick_stats[ticker]
        vol_ratio = volume / stats["avg_vol"] if stats["avg_vol"] > 0 else 1.0

        event = {
            "ticker": ticker,
            "name": name,
            "price": price,
            "volume": int(volume),
            "amount": int(amount),
            "vol_ratio": round(vol_ratio, 1),
            "level": "EMERGENCY" if is_emergency else level,
            "accumulated_5m": int(acc["total_amount"]),
            "is_emergency": is_emergency,
            "time": data["time"],
            "timestamp": int(now * 1000),
        }

        try:
            _whale_queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                _whale_queue.get_nowait()
            except Exception:
                pass
            try:
                _whale_queue.put_nowait(event)
            except Exception:
                pass


async def _subscribe(ws, ticker: str, approval_key: str):
    ticker = str(ticker).zfill(6)
    msg = json.dumps({
        "header": {
            "approval_key": approval_key,
            "custtype": "P",
            "tr_type": "1",
            "content-type": "utf-8",
        },
        "body": {
            "input": {
                "tr_id": "H0STCNT0",
                "tr_key": ticker,
            }
        }
    })
    await ws.send(msg)
    logger.info("구독 등록: %s", ticker)


async def _ws_loop():
    global _ws_connection, _running
    reconnect_count = 0

    from market_filter import is_market_open

    while _running and reconnect_count < MAX_RECONNECT:
        if not is_market_open():
            logger.info("장 미개장 — WebSocket 연결 대기")
            await asyncio.sleep(60)
            continue

        try:
            from kis_auth import get_approval_key
            approval_key = await get_approval_key()
            ws_url = get_ws_url()

            logger.info("KIS WebSocket 연결 시도: %s", ws_url)

            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                _ws_connection = ws
                reconnect_count = 0
                logger.info("KIS WebSocket 연결 성공")

                for ticker in list(_subscribed_tickers):
                    await _subscribe(ws, ticker, approval_key)

                async for raw_msg in ws:
                    if not _running:
                        break

                    if not is_market_open():
                        logger.info("장 종료 — WebSocket 종료")
                        break

                    try:
                        if isinstance(raw_msg, str):
                            if raw_msg.startswith("{"):
                                data = json.loads(raw_msg)
                                logger.debug("JSON 메시지: %s", str(data)[:100])
                            else:
                                parsed = _parse_pipe_message(raw_msg)
                                if parsed:
                                    await _process_execution(parsed)
                    except Exception as e:
                        logger.error("메시지 처리 오류: %s", e)

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning("WebSocket 연결 끊김: %s", e)
        except Exception as e:
            logger.error("WebSocket 오류: %s", e)
        finally:
            _ws_connection = None

        reconnect_count += 1
        if _running and reconnect_count < MAX_RECONNECT:
            delay = RECONNECT_DELAY * reconnect_count
            logger.info("재연결 대기 %ds (%d/%d)", delay, reconnect_count, MAX_RECONNECT)
            await asyncio.sleep(delay)

    if reconnect_count >= MAX_RECONNECT:
        logger.error("최대 재연결 횟수 초과 — WebSocket 종료")
    _ws_connection = None


async def start_ws():
    global _ws_task, _running
    if _running:
        return

    _running = True
    _ws_task = asyncio.create_task(_ws_loop())
    logger.info("KIS WebSocket 태스크 시작")


async def stop_ws():
    global _running, _ws_task, _ws_connection
    _running = False

    if _ws_connection:
        try:
            await _ws_connection.close()
        except Exception:
            pass
        _ws_connection = None

    if _ws_task:
        _ws_task.cancel()
        try:
            await _ws_task
        except asyncio.CancelledError:
            pass
        _ws_task = None
    logger.info("KIS WebSocket 중지")


async def subscribe_ticker(ticker: str):
    ticker = str(ticker).zfill(6)
    _subscribed_tickers.add(ticker)

    if _ws_connection and not _ws_connection.closed:
        try:
            from kis_auth import get_approval_key
            approval_key = await get_approval_key()
            await _subscribe(_ws_connection, ticker, approval_key)
        except Exception as e:
            logger.error("구독 오류 %s: %s", ticker, e)


async def get_whale_events_sse():
    while True:
        try:
            item = await asyncio.wait_for(_whale_queue.get(), timeout=30.0)
            yield item
        except asyncio.TimeoutError:
            yield {"heartbeat": True, "timestamp": int(time.time() * 1000)}


def get_whale_summary() -> dict:
    summary = {}
    now = time.time()
    for ticker, acc in _whale_accumulator.items():
        if now - acc["last_reset"] <= 300 and acc["total_amount"] > 0:
            summary[ticker] = {
                "total_amount": int(acc["total_amount"]),
                "event_count": len(acc["events"]),
                "last_reset": acc["last_reset"],
            }
    return summary
