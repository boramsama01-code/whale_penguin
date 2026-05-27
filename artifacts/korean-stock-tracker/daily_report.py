import os
import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
TRIGGER_HOUR = 15
TRIGGER_MINUTE = 35
CHECK_INTERVAL = 600

REPORT_PATH = os.path.join(os.path.dirname(__file__), "cache", "daily_report.json")

_store: dict = {
    "date": None,
    "generated_at": None,
    "status": "idle",
    "report": None,
    "whale_snapshot": [],
    "error": None,
}


def _load_cache():
    try:
        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            today = datetime.now(KST).strftime("%Y-%m-%d")
            if data.get("date") == today:
                _store.update(data)
                logger.info("일일 리포트 캐시 복원: %s", today)
    except Exception as e:
        logger.warning("일일 리포트 캐시 로드 실패: %s", e)


def _save_cache():
    try:
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(_store, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("일일 리포트 캐시 저장 실패: %s", e)


def get_daily_report() -> dict:
    return dict(_store)


async def generate_report(force: bool = False) -> dict:
    today = datetime.now(KST).strftime("%Y-%m-%d")
    now = datetime.now(KST)

    if not force and _store.get("date") == today and _store.get("status") == "done":
        return dict(_store)

    if _store.get("status") == "generating":
        return dict(_store)

    # 장중(09:00~15:30) 자동 생성 방지 — force=True(수동)면 허용하되 경고 표시
    is_market_hours = (
        now.weekday() < 5
        and (9, 0) <= (now.hour, now.minute) <= (15, 30)
    )
    if is_market_hours and not force:
        logger.info("장중(%s) — 일일 리포트 자동 생성 건너뜀 (장 마감 후 자동 생성)", now.strftime("%H:%M"))
        return dict(_store)

    _store["status"] = "generating"
    _store["date"] = today
    _store["error"] = None

    try:
        from kis_realtime import get_daily_whale_summary
        from claude_analyst import _get_anthropic_client
        from market_filter import get_market_state

        whale_list = get_daily_whale_summary()
        market = get_market_state()
        now_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M")

        top_whales = whale_list[:10]

        whale_lines = ""
        for i, w in enumerate(top_whales, 1):
            amt_ok = int(w.get("total_amount", 0))
            cnt = w.get("event_count", 0)
            lvl = w.get("top_level", "SMALL")
            price = w.get("price", 0)
            whale_lines += (
                f"{i}. {w['name']}({w['ticker']}) — 거래대금 {amt_ok//100000000:.0f}억원, "
                f"체결 {cnt}회, 등급 {lvl}, 현재가 {price:,}원\n"
            )

        if not whale_lines:
            whale_lines = "오늘 포착된 고래 신호 없음\n"

        kospi = market.get("kospi") or {}
        kosdaq = market.get("kosdaq") or {}
        kospi_idx = kospi.get("index", "N/A")
        kospi_cr = float(kospi.get("change_rate") or 0)
        kosdaq_idx = kosdaq.get("index", "N/A")
        kosdaq_cr = float(kosdaq.get("change_rate") or 0)

        prompt = f"""오늘({now_kst} KST) 한국 주식 시장 고래 추적 일일 리포트를 작성하라.

[시장 지표]
KOSPI: {kospi_idx} ({kospi_cr:+.2f}%)
KOSDAQ: {kosdaq_idx} ({kosdaq_cr:+.2f}%)

[오늘의 상위 고래 출몰 종목 TOP {len(top_whales)}]
{whale_lines}
위 데이터를 바탕으로 아래 JSON 형식으로 일일 리포트를 작성하라.

{{
  "제목": "YYYY-MM-DD 고래 추적 일일 리포트",
  "시장요약": "<KOSPI·KOSDAQ 오늘 흐름 2~3문장>",
  "핵심발견": "<오늘 가장 주목할 고래 활동 요약 2~3문장>",
  "주목종목": [
    {{
      "ticker": "<종목코드>",
      "name": "<종목명>",
      "이유": "<이 종목을 주목하는 이유 1~2문장>",
      "투자의견": "<매수검토|관망|주의>",
      "리스크": "<한 줄 리스크>"
    }}
  ],
  "섹터동향": "<오늘 고래가 집중된 섹터 또는 테마 분석 1~2문장>",
  "내일전망": "<내일 시장 및 주목 종목 전망 2~3문장>",
  "종합결론": "<오늘 하루 총평 및 투자자 행동 지침 2~3문장>"
}}

주목종목은 최대 5개까지. JSON 외 다른 텍스트 없이 JSON만 응답하라."""

        client = _get_anthropic_client()
        response = await asyncio.wait_for(
            client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=90.0,
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            lines = [l for l in raw.split("\n") if not l.startswith("```")]
            raw = "\n".join(lines)

        report = json.loads(raw)

        _store.update({
            "date": today,
            "generated_at": now_kst,
            "status": "done",
            "report": report,
            "whale_snapshot": top_whales,
            "error": None,
        })
        _save_cache()
        logger.info("일일 리포트 생성 완료: %s", today)

    except asyncio.TimeoutError:
        _store["status"] = "error"
        _store["error"] = "AI 응답 타임아웃 (90초 초과)"
        logger.error("일일 리포트 생성 타임아웃")
    except json.JSONDecodeError as e:
        _store["status"] = "error"
        _store["error"] = f"JSON 파싱 오류: {e}"
        logger.error("일일 리포트 JSON 파싱 오류: %s", e)
    except Exception as e:
        _store["status"] = "error"
        _store["error"] = str(e)
        logger.error("일일 리포트 생성 오류: %s", e)

    return dict(_store)


async def _scheduler_loop():
    _load_cache()
    logger.info("일일 리포트 스케줄러 시작 (매일 %02d:%02d KST 자동 생성)", TRIGGER_HOUR, TRIGGER_MINUTE)

    while True:
        try:
            now = datetime.now(KST)
            today = now.strftime("%Y-%m-%d")

            already_done = (
                _store.get("date") == today and
                _store.get("status") == "done"
            )
            market_closed = now.hour > TRIGGER_HOUR or (
                now.hour == TRIGGER_HOUR and now.minute >= TRIGGER_MINUTE
            )

            if market_closed and not already_done:
                logger.info("장 마감 감지 (%s) — 일일 리포트 자동 생성 시작", now.strftime("%H:%M"))
                await generate_report()

            await asyncio.sleep(CHECK_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("스케줄러 오류: %s", e)
            await asyncio.sleep(60)


def start_scheduler():
    return asyncio.create_task(_scheduler_loop())
