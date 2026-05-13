import os
import json
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
TIMEOUT = 30.0

SYSTEM_PROMPT = """당신은 한국 주식 시장의 세력 매매 패턴 분석 전문가입니다.
제공된 데이터를 바탕으로 세력(기관·외국인·작전세력)의 매집 또는 분산 단계를 판단하고,
반드시 JSON 형식으로만 응답하십시오.

JSON 형식:
{
  "종합점수": <0-10 숫자>,
  "세력단계": "<매집초기|매집중기|매집완료|분산시작|분산중|관망>",
  "신뢰도": "<높음|보통|낮음>",
  "펌핑가능성": <true|false>,
  "펌핑근거": "<근거 설명>",
  "진입추천": <true|false>,
  "리스크요인": ["<리스크1>", "<리스크2>"],
  "핵심근거": "<분석 핵심 설명>"
}"""


def _get_anthropic_client():
    base_url = os.getenv("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
    api_key = os.getenv("AI_INTEGRATIONS_ANTHROPIC_API_KEY", "dummy")

    import anthropic
    if base_url:
        return anthropic.AsyncAnthropic(base_url=base_url, api_key=api_key)
    else:
        real_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not real_key:
            raise RuntimeError("ANTHROPIC_API_KEY 또는 AI_INTEGRATIONS_ANTHROPIC_BASE_URL 없음")
        return anthropic.AsyncAnthropic(api_key=real_key)


async def analyze_stock(
    ticker: str,
    name: str,
    score_data: dict,
    ohlcv_summary: dict,
    supply_summary: dict,
    whale_data: Optional[dict] = None,
    disclosures: Optional[list] = None,
) -> dict:
    score = score_data.get("total", 0)
    has_whale = whale_data and whale_data.get("level") not in (None, "")

    if score < 5 and not has_whale:
        return {
            "종합점수": score,
            "세력단계": "관망",
            "신뢰도": "낮음",
            "펌핑가능성": False,
            "펌핑근거": "점수 미달",
            "진입추천": False,
            "리스크요인": ["세력 점수 낮음"],
            "핵심근거": "AI 분석 조건 미충족 (점수 5 미만, 고래 신호 없음)",
            "ai_called": False,
        }

    user_content = f"""
종목: {ticker} ({name})
세력 점수: {score}/10 (등급: {score_data.get('grade', 'N/A')})

[점수 세부]
{json.dumps(score_data.get('scores', {}), ensure_ascii=False)}

[점수 사유]
{json.dumps(score_data.get('reasons', {}), ensure_ascii=False)}

[가격 현황]
{json.dumps(ohlcv_summary, ensure_ascii=False)}

[수급 현황]
{json.dumps(supply_summary, ensure_ascii=False)}

[실시간 고래 신호]
{json.dumps(whale_data or {}, ensure_ascii=False)}

[최근 공시]
{json.dumps(disclosures or [], ensure_ascii=False)}

위 데이터를 기반으로 세력 매집/분산 단계를 분석하고 JSON으로만 응답하라.
"""

    try:
        client = _get_anthropic_client()
        response = await asyncio.wait_for(
            client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            ),
            timeout=TIMEOUT,
        )

        raw = response.content[0].text.strip()

        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines)

        result = json.loads(raw)
        result["ai_called"] = True
        return result

    except asyncio.TimeoutError:
        logger.error("Claude 응답 타임아웃 (%s)", ticker)
        return _fallback(score, "AI 응답 타임아웃")
    except json.JSONDecodeError as e:
        logger.error("Claude JSON 파싱 실패 (%s): %s", ticker, e)
        return _fallback(score, f"JSON 파싱 실패: {e}")
    except Exception as e:
        logger.error("Claude 분석 오류 (%s): %s", ticker, e)
        return _fallback(score, str(e))


def _fallback(score: float, reason: str) -> dict:
    return {
        "종합점수": score,
        "세력단계": "관망",
        "신뢰도": "낮음",
        "펌핑가능성": False,
        "펌핑근거": reason,
        "진입추천": False,
        "리스크요인": [reason],
        "핵심근거": f"AI 분석 실패: {reason}",
        "ai_called": True,
        "error": reason,
    }


async def chat_with_analyst(
    question: str,
    context: Optional[dict] = None,
    history: Optional[list] = None,
) -> str:
    messages = []
    if history:
        for h in history[-6:]:
            messages.append({"role": h["role"], "content": h["content"]})

    ctx_str = ""
    if context:
        ctx_str = f"\n[현재 분석 종목 컨텍스트]\n{json.dumps(context, ensure_ascii=False)}\n"

    messages.append({"role": "user", "content": ctx_str + question})

    system = """당신은 한국 주식 시장 전문 AI 애널리스트입니다.
세력 매매, 수급 분석, 기술적 분석, 공시 해석을 전문으로 합니다.
한국어로 명확하고 간결하게 답변하십시오."""

    try:
        client = _get_anthropic_client()
        response = await asyncio.wait_for(
            client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system,
                messages=messages,
            ),
            timeout=TIMEOUT,
        )
        return response.content[0].text.strip()
    except asyncio.TimeoutError:
        return "AI 응답 타임아웃이 발생했습니다. 다시 시도해 주세요."
    except Exception as e:
        logger.error("채팅 오류: %s", e)
        return f"AI 응답 오류: {str(e)}"
