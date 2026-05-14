import os
import asyncio
import logging
import json
import time
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from startup_event import on_startup, on_shutdown
    from kis_auth import token_refresh_loop
    from kis_realtime import start_ws

    await on_startup()
    asyncio.create_task(token_refresh_loop())
    asyncio.create_task(start_ws())
    yield
    await on_shutdown()


app = FastAPI(title="한국 주식 고래 추적기", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(static_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    clerk_pk = os.getenv("CLERK_PUBLISHABLE_KEY", "")
    html = html.replace("__CLERK_PK__", clerk_pk)
    return HTMLResponse(content=html)


@app.get("/api/search")
async def search_ticker(q: str = Query(..., min_length=1)):
    from startup_event import ticker_cache
    from pykrx import stock as pykrx_stock

    q_strip = q.strip()
    q_lower = q_strip.lower()
    results = []

    # 1. 캐시에서 먼저 검색
    for ticker, name in ticker_cache.items():
        if q_strip in name or q_lower in ticker.lower():
            results.append({"ticker": ticker, "name": name})
        if len(results) >= 20:
            break

    # 2. 캐시 결과가 없고 6자리 숫자처럼 보이면 pykrx 직접 조회
    if not results and q_strip.isdigit():
        t = q_strip.zfill(6)
        try:
            name = await asyncio.to_thread(pykrx_stock.get_market_ticker_name, t)
            if name:
                ticker_cache[t] = name
                results.append({"ticker": t, "name": name})
        except Exception:
            pass

    # 3. 캐시가 작을 때 (로딩 중) pykrx 전체 리스트에서 이름 검색
    if not results and len(ticker_cache) < 500 and not q_strip.isdigit():
        try:
            tickers_raw = await asyncio.to_thread(pykrx_stock.get_market_ticker_list, market="ALL")
            semaphore = asyncio.Semaphore(10)

            async def check_name(t):
                t = str(t).zfill(6)
                if t in ticker_cache:
                    name = ticker_cache[t]
                    if q_strip in name:
                        return {"ticker": t, "name": name}
                    return None
                async with semaphore:
                    try:
                        name = await asyncio.to_thread(pykrx_stock.get_market_ticker_name, t)
                        if name:
                            ticker_cache[t] = name
                            if q_strip in name:
                                return {"ticker": t, "name": name}
                    except Exception:
                        pass
                return None

            found = await asyncio.gather(*[check_name(t) for t in tickers_raw[:300]], return_exceptions=True)
            for r in found:
                if isinstance(r, dict):
                    results.append(r)
                    if len(results) >= 20:
                        break
        except Exception as e:
            logger.error("pykrx 직접 검색 오류: %s", e)

    return {"results": results[:20], "query": q_strip}


@app.get("/api/scan")
async def scan():
    try:
        from screening import scan_market
        results = await asyncio.wait_for(scan_market(max_results=50), timeout=120.0)
        return {"results": results, "count": len(results), "timestamp": int(time.time() * 1000)}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="스캔 타임아웃")
    except Exception as e:
        logger.error("스캔 오류: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyze/{ticker}")
async def analyze_ticker(ticker: str):
    ticker = str(ticker).zfill(6)
    try:
        from screening import get_stock_detail
        from dart_client import get_disclosures
        from claude_analyst import analyze_stock, store_analysis_cache
        from strategy import generate_strategy
        from kis_realtime import get_whale_summary, subscribe_ticker

        await subscribe_ticker(ticker)

        detail = await asyncio.wait_for(get_stock_detail(ticker), timeout=60.0)
        if not detail:
            raise HTTPException(status_code=404, detail=f"종목 {ticker} 데이터 없음 (pykrx 조회 실패 — 장 마감 후 일부 종목은 데이터가 없을 수 있습니다)")

        disclosures = await get_disclosures(ticker, limit=5)

        whale_summary = get_whale_summary()
        whale_data = whale_summary.get(ticker)

        # 고래 신호(H)를 점수에 반영
        if whale_data:
            from scoring import calculate_score, score_whale_signal
            ws, wr = score_whale_signal(whale_data)
            detail["score"]["scores"]["H"] = ws
            detail["score"]["reasons"]["H"] = wr
            weights = detail["score"].get("weights", {"A":2,"B":1.5,"C":1.5,"D":1,"E":1,"F":2,"G":0.5,"H":2})
            total_weight = sum(weights.values())
            weighted_sum = sum(detail["score"]["scores"][k] * weights.get(k,1) for k in detail["score"]["scores"])
            detail["score"]["total"] = round(weighted_sum / total_weight, 2)

        # G 업종모멘텀: 코스피 대비 상대수익률 계산
        market_change_20d = 0.0
        try:
            from pykrx import stock as pykrx_stock
            from datetime import datetime as _dt, timedelta as _td
            _end = _dt.now().strftime("%Y%m%d")
            _start = (_dt.now() - _td(days=40)).strftime("%Y%m%d")
            _kospi = await asyncio.wait_for(
                asyncio.to_thread(pykrx_stock.get_index_ohlcv_by_date, _start, _end, "1001"),
                timeout=5.0,
            )
            if _kospi is not None and len(_kospi) >= 20:
                _cc = "종가" if "종가" in _kospi.columns else _kospi.columns[3]
                market_change_20d = float((_kospi[_cc].values[-1] / _kospi[_cc].values[-20] - 1) * 100)
        except Exception:
            pass

        # G 점수 재계산
        if detail.get("ohlcv") and len(detail["ohlcv"]) >= 20:
            import pandas as pd
            from scoring import score_relative_momentum
            _closes = [r["close"] for r in detail["ohlcv"]]
            _odf = pd.DataFrame({"종가": _closes})
            _gs, _gr = score_relative_momentum(_odf, market_change_20d)
            detail["score"]["scores"]["G"] = _gs
            detail["score"]["reasons"]["G"] = _gr
            _weights = detail["score"].get("weights", {"A":2,"B":1.5,"C":1.5,"D":1,"E":1,"F":2,"G":0.5,"H":2})
            _tw = sum(_weights.values())
            _ws = sum(detail["score"]["scores"][k] * _weights.get(k,1) for k in detail["score"]["scores"])
            detail["score"]["total"] = round(_ws / _tw, 2)
            from scoring import _grade
            detail["score"]["grade"] = _grade(detail["score"]["total"])

        ohlcv_summary = {}
        if detail.get("ohlcv"):
            recent = detail["ohlcv"][-5:]
            all_ohlcv = detail["ohlcv"]
            ohlcv_summary = {
                "최근5일종가": [r["close"] for r in recent],
                "최근5일거래량": [r["volume"] for r in recent],
                "현재가": detail["price"],
                "등락률": detail["change_rate"],
                "52주고": detail["high52"],
                "52주저": detail["low52"],
                "총데이터일수": len(all_ohlcv),
                "20일전종가": all_ohlcv[-20]["close"] if len(all_ohlcv) >= 20 else None,
                "60일전종가": all_ohlcv[-60]["close"] if len(all_ohlcv) >= 60 else None,
                "120일전종가": all_ohlcv[-120]["close"] if len(all_ohlcv) >= 120 else None,
                "240일전종가": all_ohlcv[-240]["close"] if len(all_ohlcv) >= 240 else None,
                "52주최저대비": round((detail["price"] - detail["low52"]) / max(detail["low52"],1) * 100, 1) if detail.get("low52") else None,
                "52주최고대비": round((detail["price"] - detail["high52"]) / max(detail["high52"],1) * 100, 1) if detail.get("high52") else None,
            }

        supply_summary = {}
        if detail.get("supply"):
            recent_s = detail["supply"][-5:]
            supply_summary = {"최근수급": recent_s}

        ai_result = await analyze_stock(
            ticker=ticker,
            name=detail["name"],
            score_data=detail["score"],
            ohlcv_summary=ohlcv_summary,
            supply_summary=supply_summary,
            whale_data=whale_data,
            disclosures=disclosures,
        )

        # 채팅 컨텍스트용 캐시 저장
        store_analysis_cache(ticker, detail, ai_result, disclosures)

        strategy = generate_strategy(
            ticker=ticker,
            name=detail["name"],
            current_price=detail["price"],
            score=detail["score"].get("total", 0),
            high52=detail["high52"],
            low52=detail["low52"],
        )

        return {
            "ticker": ticker,
            "detail": detail,
            "ai_analysis": ai_result,
            "strategy": strategy,
            "disclosures": disclosures,
            "whale_data": whale_data,
            "timestamp": int(time.time() * 1000),
        }

    except HTTPException:
        raise
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="분석 타임아웃")
    except Exception as e:
        logger.error("분석 오류 %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dart/{ticker}")
async def dart_disclosures(ticker: str):
    ticker = str(ticker).zfill(6)
    try:
        from dart_client import get_disclosures
        items = await get_disclosures(ticker, limit=15)
        return {"ticker": ticker, "disclosures": items, "count": len(items)}
    except Exception as e:
        logger.error("DART 조회 오류 %s: %s", ticker, e)
        raise HTTPException(status_code=500, detail=str(e))


_breadth_cache: dict = {"data": {}, "ts": 0.0}


@app.get("/api/market-status")
async def market_status():
    import datetime as _dt
    from market_filter import get_market_state, update_market_state
    try:
        await update_market_state()
    except Exception:
        pass
    result = get_market_state()

    # 시장 breadth (상승/하락/보합 종목 수) — KIS 거래량 순위에서 등락률로 추정, 10분 캐시
    now_ts = time.time()
    if now_ts - _breadth_cache["ts"] > 600:
        try:
            from kis_data import get_volume_rank_kis
            kospi_list, kosdaq_list = await asyncio.gather(
                get_volume_rank_kis("J", 200),
                get_volume_rank_kis("Q", 200),
            )

            def _breadth_from_rank(lst):
                if not lst:
                    return {"up": 0, "down": 0, "flat": 0}
                up = sum(1 for x in lst if x.get("change_rate", 0) > 0)
                down = sum(1 for x in lst if x.get("change_rate", 0) < 0)
                flat = len(lst) - up - down
                return {"up": up, "down": down, "flat": flat}

            _breadth_cache["data"] = {
                "kospi": _breadth_from_rank(kospi_list),
                "kosdaq": _breadth_from_rank(kosdaq_list),
            }
            _breadth_cache["ts"] = now_ts
        except Exception as e:
            logger.debug("breadth 조회 실패: %s", e)

    bd = _breadth_cache.get("data", {})
    if isinstance(result.get("kospi"), dict) and bd.get("kospi"):
        result["kospi"].update(bd["kospi"])
    if isinstance(result.get("kosdaq"), dict) and bd.get("kosdaq"):
        result["kosdaq"].update(bd["kosdaq"])

    return result


@app.get("/api/whale/realtime")
async def whale_realtime(request: Request):
    from kis_realtime import get_whale_events_sse

    async def event_generator() -> AsyncGenerator:
        try:
            async for event in get_whale_events_sse():
                if await request.is_disconnected():
                    break
                if event.get("heartbeat"):
                    yield {"event": "heartbeat", "data": json.dumps(event)}
                else:
                    yield {"event": "whale", "data": json.dumps(event, ensure_ascii=False)}
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("SSE 오류: %s", e)
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())


@app.get("/api/whale/summary")
async def whale_summary():
    from kis_realtime import get_whale_summary
    from startup_event import ticker_cache
    summary = get_whale_summary()
    result = []
    for ticker, data in summary.items():
        result.append({
            "ticker": ticker,
            "name": ticker_cache.get(ticker, ticker),
            **data,
        })
    result.sort(key=lambda x: x["total_amount"], reverse=True)
    return {"summary": result, "count": len(result)}


@app.get("/api/whale/daily")
async def whale_daily():
    import asyncio
    import datetime as _dt
    from kis_realtime import get_daily_whale_summary
    from startup_event import ticker_cache

    items = get_daily_whale_summary()
    if items:
        return {"items": items, "count": len(items), "source": "realtime", "timestamp": int(time.time() * 1000)}

    # 실시간 데이터 없으면 pykrx 일별 데이터로 대체
    try:
        from pykrx import stock as pykrx_stock

        now = _dt.datetime.now()
        today_str = now.strftime("%Y%m%d")

        def prev_biz_days(n):
            days = []
            d = now - _dt.timedelta(days=1)
            while len(days) < n:
                if d.weekday() < 5:
                    days.append(d.strftime("%Y%m%d"))
                d -= _dt.timedelta(days=1)
            return days

        prev_days = prev_biz_days(2)
        yesterday_str = prev_days[0]

        async def get_ohlcv(date):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(pykrx_stock.get_market_ohlcv_by_ticker, date, market="ALL"),
                    timeout=30.0
                )
            except Exception:
                return None

        today_df, yesterday_df = await asyncio.gather(
            get_ohlcv(today_str),
            get_ohlcv(yesterday_str)
        )

        if today_df is None or (hasattr(today_df, 'empty') and today_df.empty):
            today_df = yesterday_df
            yesterday_df = await get_ohlcv(prev_days[1]) if len(prev_days) > 1 else None

        if today_df is None or (hasattr(today_df, 'empty') and today_df.empty):
            return {"items": [], "count": 0, "source": "none", "timestamp": int(time.time() * 1000)}

        vol_col = next((c for c in ['거래량', 'Volume'] if c in today_df.columns), None)
        amt_col = next((c for c in ['거래대금'] if c in today_df.columns), None)
        close_col = next((c for c in ['종가', 'Close'] if c in today_df.columns), None)
        chg_col = next((c for c in ['등락률'] if c in today_df.columns), None)

        results = []
        for ticker_raw in today_df.index:
            ticker = str(ticker_raw).zfill(6)
            row = today_df.loc[ticker_raw]

            vol = int(row[vol_col]) if vol_col else 0
            amount = int(row[amt_col]) if amt_col else 0
            close = int(row[close_col]) if close_col else 0
            change_rate = float(row[chg_col]) if chg_col else 0.0

            if amount < 500_000_000:
                continue

            vol_ratio = 1.0
            if yesterday_df is not None and not yesterday_df.empty and ticker_raw in yesterday_df.index:
                yd_row = yesterday_df.loc[ticker_raw]
                yd_vol = int(yd_row[vol_col]) if vol_col and vol_col in yesterday_df.columns else 0
                if yd_vol > 0:
                    vol_ratio = round(vol / yd_vol, 1)

            level = "SMALL"
            if vol_ratio >= 10:
                level = "EMERGENCY"
            elif vol_ratio >= 5:
                level = "LARGE"
            elif vol_ratio >= 3:
                level = "MEDIUM"

            name = ticker_cache.get(ticker, ticker)
            results.append({
                "ticker": ticker,
                "name": name,
                "price": close,
                "change_rate": change_rate,
                "volume": vol,
                "total_amount": amount,
                "vol_ratio": vol_ratio,
                "event_count": 0,
                "top_level": level,
                "first_seen": "09:00",
                "last_seen": "15:30",
            })

        results.sort(key=lambda x: x["total_amount"], reverse=True)
        return {"items": results[:30], "count": len(results), "source": "pykrx", "timestamp": int(time.time() * 1000)}

    except Exception as e:
        logger.error("whale_daily pykrx 오류: %s", e)
        return {"items": [], "count": 0, "source": "error", "error": str(e), "timestamp": int(time.time() * 1000)}


class ChatRequest(BaseModel):
    question: str
    ticker: Optional[str] = None
    history: Optional[list] = None


@app.post("/api/chat")
async def chat(req: ChatRequest):
    from claude_analyst import chat_with_analyst

    context = None
    if req.ticker:
        from startup_event import ticker_cache
        t = str(req.ticker).zfill(6)
        context = {"ticker": t, "name": ticker_cache.get(t, t)}

    try:
        answer = await chat_with_analyst(
            question=req.question,
            context=context,
            history=req.history,
        )
        return {"answer": answer, "ticker": req.ticker}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PortfolioPosition(BaseModel):
    ticker: str
    avg_price: float
    quantity: int


class PortfolioRequest(BaseModel):
    positions: list[PortfolioPosition]


@app.post("/api/portfolio/analyze")
async def portfolio_analyze(req: PortfolioRequest):
    try:
        from claude_analyst import analyze_portfolio
        positions = [p.model_dump() for p in req.positions]
        results = await asyncio.wait_for(analyze_portfolio(positions), timeout=120.0)
        total_pnl = sum(r.get("pnl_amount", 0) for r in results)
        total_invested = sum(r.get("avg_price", 0) * r.get("quantity", 0) for r in results)
        return {
            "positions": results,
            "summary": {
                "total_invested": int(total_invested),
                "total_pnl": int(total_pnl),
                "total_pnl_rate": round(total_pnl / total_invested * 100, 2) if total_invested > 0 else 0,
                "count": len(results),
            }
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="포트폴리오 분석 타임아웃")
    except Exception as e:
        logger.error("포트폴리오 분석 오류: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class SettingsRequest(BaseModel):
    kis_env: Optional[str] = None
    whale_small_threshold: Optional[float] = None
    whale_medium_threshold: Optional[float] = None
    whale_large_threshold: Optional[float] = None
    anthropic_api_key: Optional[str] = None


_settings: dict = {
    "kis_env": os.getenv("KIS_ENV", "PROD"),
    "whale_small_threshold": 3_000_000,
    "whale_medium_threshold": 10_000_000,
    "whale_large_threshold": 50_000_000,
}


@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    if req.kis_env:
        _settings["kis_env"] = req.kis_env
    if req.whale_small_threshold is not None:
        _settings["whale_small_threshold"] = req.whale_small_threshold
    if req.whale_medium_threshold is not None:
        _settings["whale_medium_threshold"] = req.whale_medium_threshold
    if req.whale_large_threshold is not None:
        _settings["whale_large_threshold"] = req.whale_large_threshold
    if req.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = req.anthropic_api_key
        _settings["has_anthropic_key"] = True
        logger.info("Anthropic API 키가 런타임에 설정되었습니다")
    return {"success": True, "settings": _settings}


@app.get("/api/settings")
async def get_settings():
    result = dict(_settings)
    result["has_anthropic_key"] = bool(
        os.getenv("ANTHROPIC_API_KEY") or os.getenv("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
    )
    return result


@app.get("/api/config")
async def config():
    return {
        "clerk_publishable_key": os.getenv("CLERK_PUBLISHABLE_KEY", ""),
    }


@app.get("/api/news/{ticker}")
async def stock_news(ticker: str):
    ticker = str(ticker).zfill(6)
    from startup_event import ticker_cache
    name = ticker_cache.get(ticker, ticker)
    try:
        from news_client import get_stock_news
        items = await asyncio.wait_for(get_stock_news(ticker, name, limit=6), timeout=10.0)
        return {"ticker": ticker, "name": name, "items": items, "count": len(items)}
    except Exception as e:
        logger.error("뉴스 조회 오류 %s: %s", ticker, e)
        return {"ticker": ticker, "name": name, "items": [], "count": 0}


@app.get("/api/health")
async def health():
    from startup_event import ticker_cache
    return {
        "status": "ok",
        "timestamp": int(time.time() * 1000),
        "ticker_cache_size": len(ticker_cache),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
