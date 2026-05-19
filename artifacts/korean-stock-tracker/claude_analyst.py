import os
import json
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
TIMEOUT = 60.0

_analysis_cache: dict[str, dict] = {}

SYSTEM_PROMPT = """당신은 한국 주식 시장의 세력 매매 패턴 분석 전문가입니다.
최소 6개월~1년 데이터를 기반으로 세력(기관·외국인·작전세력)의 매집 또는 분산 단계를 판단합니다.

다음 사항을 반드시 분석하십시오:
1. 장기(6~12개월) 차원의 매집/분산 패턴 — 단기 급등락이 아닌 서서히 진행된 매집 여부
2. 세력이 이미 나갔는지(분산 완료), 매집 후 횡보 중인지, 아직 매집 중인지 판단
3. 매집완료 또는 매집중으로 판단되는 경우: 추정 매집가격대 (지지선/저점 구간)
4. 목표가격대: 52주 고점, 전고점, 박스권 상단 등을 기준으로 단기/중기 목표가 추정
5. 시간대별 주가 방향 예상: 1일·1주·1달·6개월·1년 후 각각 상승/하락/횡보 및 확률(%)

반드시 JSON 형식으로만 응답하십시오.

JSON 형식:
{
  "종합점수": <0-10 숫자>,
  "세력단계": "<매집초기|매집중기|매집완료|분산시작|분산중|관망|불명확>",
  "신뢰도": "<높음|보통|낮음>",
  "분석기간": "<분석에 활용한 기간 — 예: 약 1년(252거래일)>",
  "장기패턴": "<장기 패턴 요약>",
  "매집가격대": {
    "추정여부": <true|false>,
    "하단": <추정 매집 하단가격 또는 null>,
    "상단": <추정 매집 상단가격 또는 null>,
    "근거": "<가격대 추정 근거>"
  },
  "목표가격대": {
    "단기": <단기 목표가 또는 null>,
    "중기": <중기 목표가 또는 null>,
    "근거": "<목표가 산정 근거>"
  },
  "예상전망": {
    "1일": {"방향": "<상승|하락|횡보>", "확률": <0-100 정수>, "근거": "<한 줄 근거>"},
    "1주": {"방향": "<상승|하락|횡보>", "확률": <0-100 정수>, "근거": "<한 줄 근거>"},
    "1달": {"방향": "<상승|하락|횡보>", "확률": <0-100 정수>, "근거": "<한 줄 근거>"},
    "6개월": {"방향": "<상승|하락|횡보>", "확률": <0-100 정수>, "근거": "<한 줄 근거>"},
    "1년": {"방향": "<상승|하락|횡보>", "확률": <0-100 정수>, "근거": "<한 줄 근거>"}
  },
  "펌핑가능성": <true|false>,
  "펌핑근거": "<근거 설명>",
  "진입추천": <true|false>,
  "리스크요인": ["<리스크1>", "<리스크2>"],
  "핵심근거": "<분석 핵심 설명 — 장기 관점 포함>"
}"""


def _repair_json(raw: str) -> str:
    """잘린 JSON을 닫힌 상태로 복구 시도"""
    raw = raw.strip()
    if not raw.startswith("{"):
        idx = raw.find("{")
        if idx >= 0:
            raw = raw[idx:]
    # 이미 유효한 JSON이면 그대로 반환
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass
    # 열린 괄호/따옴표를 추적해서 닫기
    stack = []
    in_string = False
    escape_next = False
    for ch in raw:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            if in_string:
                in_string = False
            else:
                in_string = True
        elif not in_string:
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack and stack[-1] == ch:
                    stack.pop()
    # 열린 문자열이 있으면 먼저 닫기
    if in_string:
        raw += '"'
    # 나머지 닫기 괄호 추가
    raw += "".join(reversed(stack))
    return raw


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


def store_analysis_cache(ticker: str, detail: dict, ai_result: dict, disclosures: list):
    _analysis_cache[ticker] = {
        "ticker": ticker,
        "name": detail.get("name", ticker),
        "price": detail.get("price", 0),
        "change_rate": detail.get("change_rate", 0),
        "volume": detail.get("volume", 0),
        "high52": detail.get("high52", 0),
        "low52": detail.get("low52", 0),
        "score": detail.get("score", {}),
        "ai_analysis": ai_result,
        "ohlcv_len": len(detail.get("ohlcv", [])),
        "recent_disclosures": [d.get("report_nm", "") for d in (disclosures or [])[:5]],
        "supply_summary": detail.get("supply", [])[-5:] if detail.get("supply") else [],
    }


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
    price = ohlcv_summary.get("현재가", 0)
    high52 = ohlcv_summary.get("52주고", 0)
    low52 = ohlcv_summary.get("52주저", 0)
    data_days = ohlcv_summary.get("총데이터일수", 0)

    def _pct(p_now, p_prev):
        if not p_prev or not p_now: return "N/A"
        return f"{(p_now/p_prev-1)*100:+.1f}%"

    long_term_ctx = f"""
[장기 가격 현황 — {data_days}거래일 데이터 기준]
현재가: {price:,}원
52주 최고가: {high52:,}원 (현재 대비 {_pct(price, high52)})
52주 최저가: {low52:,}원 (현재 대비 {_pct(price, low52)})
52주 최저 대비: {ohlcv_summary.get('52주최저대비','N/A')}%
52주 최고 대비: {ohlcv_summary.get('52주최고대비','N/A')}%

20일 전 종가: {ohlcv_summary.get('20일전종가','N/A')}원 → 현재 {_pct(price, ohlcv_summary.get('20일전종가'))}
60일 전 종가: {ohlcv_summary.get('60일전종가','N/A')}원 → 현재 {_pct(price, ohlcv_summary.get('60일전종가'))}
120일 전 종가: {ohlcv_summary.get('120일전종가','N/A')}원 → 현재 {_pct(price, ohlcv_summary.get('120일전종가'))}
240일 전 종가: {ohlcv_summary.get('240일전종가','N/A')}원 → 현재 {_pct(price, ohlcv_summary.get('240일전종가'))}
등락률(당일): {ohlcv_summary.get('등락률', 0):+.2f}%
최근5일 종가: {ohlcv_summary.get('최근5일종가', [])}
최근5일 거래량: {ohlcv_summary.get('최근5일거래량', [])}"""

    user_content = f"""
종목: {ticker} ({name})
세력 점수: {score:.1f}/10 (등급: {score_data.get('grade', 'N/A')})

[점수 세부 - 항목별 0~10점]
거래량이상(A): {score_data.get('scores', {}).get('A', 0):.1f} — {score_data.get('reasons', {}).get('A', '')}
가격/거래량괴리(B): {score_data.get('scores', {}).get('B', 0):.1f} — {score_data.get('reasons', {}).get('B', '')}
수급신뢰도(C): {score_data.get('scores', {}).get('C', 0):.1f} — {score_data.get('reasons', {}).get('C', '')}
기술지표(D): {score_data.get('scores', {}).get('D', 0):.1f} — {score_data.get('reasons', {}).get('D', '')}
거래량프로파일(E): {score_data.get('scores', {}).get('E', 0):.1f} — {score_data.get('reasons', {}).get('E', '')}
박스권돌파(F): {score_data.get('scores', {}).get('F', 0):.1f} — {score_data.get('reasons', {}).get('F', '')}
업종모멘텀(G): {score_data.get('scores', {}).get('G', 0):.1f} — {score_data.get('reasons', {}).get('G', '')}
고래신호(H): {score_data.get('scores', {}).get('H', 0):.1f} — {score_data.get('reasons', {}).get('H', '')}

{long_term_ctx}

[수급 현황 (최근 60일 기반)]
{json.dumps(supply_summary, ensure_ascii=False)}

[실시간 고래 신호]
{json.dumps(whale_data or {}, ensure_ascii=False)}

[최근 공시]
{json.dumps(disclosures or [], ensure_ascii=False)}

위 데이터를 기반으로 장기(6~12개월) 관점에서 세력 매집/분산 단계를 분석하고 JSON으로만 응답하라.
매집완료 또는 매집중으로 판단되면 반드시 추정 매집가격대와 목표가격대를 제시하라.
"""

    try:
        client = _get_anthropic_client()
        response = await asyncio.wait_for(
            client.messages.create(
                model=MODEL,
                max_tokens=3500,
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

        # 잘린 JSON 자동 복구 시도
        raw = _repair_json(raw)

        result = json.loads(raw)
        result["ai_called"] = True
        return result

    except asyncio.TimeoutError:
        logger.error("Claude 응답 타임아웃 (%s)", ticker)
        return _fallback(score, "AI 응답 타임아웃", ohlcv_summary)
    except json.JSONDecodeError as e:
        logger.error("Claude JSON 파싱 실패 (%s): %s\n원본(마지막100자): %s", ticker, e, raw[-100:] if 'raw' in dir() else '')
        return _fallback(score, "AI 분석 실패 (응답 길이 부족)", ohlcv_summary)
    except Exception as e:
        logger.error("Claude 분석 오류 (%s): %s", ticker, e)
        return _fallback(score, str(e), ohlcv_summary)


def _fallback(score: float, reason: str, ohlcv_summary: Optional[dict] = None) -> dict:
    if any(kw in reason.lower() for kw in ["error code", "credit", "billing", "rate limit", "quota", "overload"]):
        display_reason = "AI 분석 서비스 일시 불가 — 퀀트 점수 기반 자동 추정"
    elif any(kw in reason.lower() for kw in ["timeout", "타임아웃", "timed out"]):
        display_reason = "AI 응답 타임아웃 — 퀀트 점수 기반 자동 추정"
    elif any(kw in reason.lower() for kw in ["key", "auth", "permission", "forbidden"]):
        display_reason = "AI API 인증 오류 — 퀀트 점수 기반 자동 추정"
    else:
        short = reason[:50] if len(reason) > 50 else reason
        display_reason = f"AI 분석 불가 ({short}) — 퀀트 점수 기반 자동 추정"

    if score >= 7.5:
        stage, confidence, pumping, entry_rec = "매집완료", "보통", True, True
        pattern = f"퀀트 종합점수 {score:.1f}점 — 강한 매집 신호 다수 감지. 세력 매집 완료 단계로 추정됩니다."
        reason_text = f"세력점수 {score:.1f}/10 고점대 — 거래량 이상·수급 집중·기술지표 복합 신호 동시 감지. 매집 완료 구간 추정."
    elif score >= 6.0:
        stage, confidence, pumping, entry_rec = "매집중기", "보통", False, True
        pattern = f"퀀트 종합점수 {score:.1f}점 — 매집 중기 단계 신호. 지속적 수급 집중 패턴 감지."
        reason_text = f"세력점수 {score:.1f}/10 상위권 — 수급 신뢰도 및 거래량 이상 지표 동반 상승. 중기 매집 가능성 높음."
    elif score >= 4.5:
        stage, confidence, pumping, entry_rec = "매집초기", "낮음", False, False
        pattern = f"퀀트 종합점수 {score:.1f}점 — 초기 매집 가능성 있으나 추가 확인 필요."
        reason_text = f"세력점수 {score:.1f}/10 중위권 — 일부 매집 신호 감지. 추이 추가 관찰 권고."
    elif score >= 2.5:
        stage, confidence, pumping, entry_rec = "관망", "낮음", False, False
        pattern = f"퀀트 종합점수 {score:.1f}점 — 뚜렷한 세력 신호 미감지. 관망 구간."
        reason_text = f"세력점수 {score:.1f}/10 하위권 — 유의미한 세력 활동 신호 부재."
    else:
        stage, confidence, pumping, entry_rec = "관망", "낮음", False, False
        pattern = f"퀀트 종합점수 {score:.1f}점 — 세력 활동 없음 또는 분산 단계."
        reason_text = f"세력점수 {score:.1f}/10 저점 — 세력 매집 신호 없음."

    # ── 가격 추정 (OHLCV 데이터 기반) ──────────────────────────────────
    def _round_price(p: float) -> int:
        """100원 단위 반올림"""
        if p <= 0:
            return 0
        if p < 1000:
            return max(1, round(p / 10) * 10)
        if p < 10000:
            return round(p / 100) * 100
        return round(p / 500) * 500

    acc_data: dict = {"추정여부": False, "하단": None, "상단": None, "근거": display_reason}
    tgt_data: dict = {"단기": None, "중기": None, "근거": display_reason}

    if ohlcv_summary and entry_rec:
        price  = ohlcv_summary.get("현재가", 0) or 0
        high52 = ohlcv_summary.get("52주고", 0) or 0
        low52  = ohlcv_summary.get("52주저", 0) or 0
        p60    = ohlcv_summary.get("60일전종가", 0) or 0
        p120   = ohlcv_summary.get("120일전종가", 0) or 0
        p240   = ohlcv_summary.get("240일전종가", 0) or 0

        if price > 0:
            # 매집가격대: 52주저와 최근 역사적 저점들의 중간 구간
            candidate_lows = [v for v in [low52, p60, p120, p240] if v > 0]
            if candidate_lows:
                bottom = min(candidate_lows)
                # 상단 = 현재가와 저점들의 중간값 (매집이 이뤄진 구간 상단)
                mid = (bottom + price) / 2
                acc_lower = _round_price(bottom * 1.02)
                acc_upper = _round_price(mid)
                if acc_lower < acc_upper and acc_lower > 0:
                    acc_data = {
                        "추정여부": True,
                        "하단": acc_lower,
                        "상단": acc_upper,
                        "근거": f"52주 저점({_round_price(low52):,}원)~현재가 중간 구간 기술적 추정 (AI 미사용)",
                    }

            # 목표가격대
            if high52 > price:
                short_tgt = _round_price(price + (high52 - price) * 0.4)
                mid_tgt   = _round_price(high52 * 0.97)
            else:
                short_tgt = _round_price(price * 1.10)
                mid_tgt   = _round_price(price * 1.25)
            tgt_data = {
                "단기": short_tgt,
                "중기": mid_tgt,
                "근거": f"52주 고점({_round_price(high52):,}원) 기반 기술적 목표가 추정 (AI 미사용)",
            }

    up_dir = "상승" if entry_rec else "횡보"
    return {
        "종합점수": score,
        "세력단계": stage,
        "신뢰도": confidence,
        "분석기간": "퀀트 자동 추정",
        "장기패턴": pattern,
        "매집가격대": acc_data,
        "목표가격대": tgt_data,
        "예상전망": {
            "1일": {"방향": "횡보", "확률": 50, "근거": display_reason},
            "1주": {"방향": up_dir, "확률": 55 if entry_rec else 45, "근거": display_reason},
            "1달": {"방향": up_dir, "확률": 60 if score >= 7.5 else (55 if entry_rec else 45), "근거": display_reason},
            "6개월": {"방향": up_dir if score >= 6.0 else "횡보", "확률": 55 if entry_rec else 45, "근거": display_reason},
            "1년": {"방향": "횡보", "확률": 50, "근거": display_reason},
        },
        "펌핑가능성": pumping,
        "펌핑근거": "퀀트 점수 기반 추정 (실제 AI 분석 시 갱신 필요)" if pumping else "없음",
        "진입추천": entry_rec,
        "리스크요인": ["AI 분석 부재로 정확도 제한", "퀀트 점수만으로 최종 판단 지양"],
        "핵심근거": reason_text,
        "ai_called": True,
        "_fallback": True,
    }


async def stream_chat_with_analyst(
    question: str,
    context: Optional[dict] = None,
    history: Optional[list] = None,
):
    """AI 채팅 스트리밍 버전 — async generator, yields text tokens"""
    messages = []
    if history:
        for h in history[-6:]:
            if h.get("role") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})

    ctx_parts = []
    if context:
        ticker = str(context.get("ticker", "")).zfill(6)
        cached = _analysis_cache.get(ticker)
        if cached:
            score = cached.get("score", {})
            ai = cached.get("ai_analysis", {})
            acc = ai.get("매집가격대", {})
            tgt = ai.get("목표가격대", {})
            ctx_parts.append(f"""[현재 분석 종목]
종목: {ticker} ({cached.get('name', ticker)})
현재가: {cached.get('price', 0):,}원 ({cached.get('change_rate', 0):+.2f}%)
52주 범위: {cached.get('low52', 0):,} ~ {cached.get('high52', 0):,}원
분석 데이터: {cached.get('ohlcv_len', 0)}거래일

[세력 점수]
종합: {score.get('total', 0):.1f}/10 (등급: {score.get('grade', 'N/A')})
거래량이상: {score.get('scores', {}).get('A', 0):.1f} — {score.get('reasons', {}).get('A', '')}
가격/거래량: {score.get('scores', {}).get('B', 0):.1f} — {score.get('reasons', {}).get('B', '')}
수급신뢰도: {score.get('scores', {}).get('C', 0):.1f} — {score.get('reasons', {}).get('C', '')}
기술지표: {score.get('scores', {}).get('D', 0):.1f} — {score.get('reasons', {}).get('D', '')}
거래량프로파일: {score.get('scores', {}).get('E', 0):.1f} — {score.get('reasons', {}).get('E', '')}
박스권돌파: {score.get('scores', {}).get('F', 0):.1f} — {score.get('reasons', {}).get('F', '')}
업종모멘텀: {score.get('scores', {}).get('G', 0):.1f} — {score.get('reasons', {}).get('G', '')}
고래신호: {score.get('scores', {}).get('H', 0):.1f} — {score.get('reasons', {}).get('H', '')}

[AI 분석 결과]
세력단계: {ai.get('세력단계', 'N/A')}
장기패턴: {ai.get('장기패턴', 'N/A')}
신뢰도: {ai.get('신뢰도', 'N/A')}
펌핑가능성: {ai.get('펌핑가능성', False)}
진입추천: {ai.get('진입추천', False)}
핵심근거: {ai.get('핵심근거', 'N/A')}
매집가격대: {acc.get('하단', 'N/A')}~{acc.get('상단', 'N/A')}원 (추정: {acc.get('추정여부', False)})
목표가: 단기 {tgt.get('단기', 'N/A')}원 / 중기 {tgt.get('중기', 'N/A')}원

[최근 공시]
{', '.join(cached.get('recent_disclosures', ['없음']))}""")
        else:
            ctx_parts.append(f"[안내] {context.get('name', ticker)} ({ticker}) 종목의 분석 데이터가 없습니다. 사용자에게 'AI분석 탭에서 해당 종목을 검색하고 분석 버튼을 눌러주세요'라고 안내하세요.")

    ctx_str = "\n".join(ctx_parts)
    if ctx_str:
        ctx_str = ctx_str + "\n\n"

    messages.append({"role": "user", "content": ctx_str + question})

    system = """당신은 한국 주식 시장 전문 AI 애널리스트입니다.
세력 매매, 수급 분석, 기술적 분석, 공시 해석, 장기 매집/분산 패턴 분석을 전문으로 합니다.

★ 절대 규칙:
1. 메시지에 종목 데이터(현재가, 세력점수, 매집가격대, 목표가, 수급 등)가 포함되어 있으면 그 데이터를 즉시 사용하여 답변하라. 절대로 "데이터가 없다", "차트를 보내달라", "추가 정보가 필요하다" 같은 말을 하지 마라.
2. 매수 여부, 목표가, 손절가, 진입 시점을 물으면 제공된 데이터를 근거로 명확한 견해를 제시하라. 회피하지 마라.
3. 이미 분석된 데이터가 있으면 "정확한 분석을 위해 정보가 필요합니다" 같은 말은 절대 하지 마라.
4. 데이터가 전혀 없을 때(no data 상황)만 "AI분석 탭에서 해당 종목을 먼저 분석해주세요"라고 안내하라.

한국어로 명확하고 직접적으로 답변하라. 단락 구분을 명확히 하고 줄바꿈을 적절히 사용하라."""

    try:
        client = _get_anthropic_client()
        async with client.messages.stream(
            model=MODEL,
            max_tokens=1200,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text
    except asyncio.TimeoutError:
        yield "\n[AI 응답 타임아웃이 발생했습니다. 다시 시도해 주세요.]"
    except Exception as e:
        logger.error("채팅 스트리밍 오류: %s", e)
        yield f"\n[AI 응답 오류: {str(e)}]"


async def chat_with_analyst(
    question: str,
    context: Optional[dict] = None,
    history: Optional[list] = None,
) -> str:
    messages = []
    if history:
        for h in history[-6:]:
            if h.get("role") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})

    ctx_parts = []
    if context:
        ticker = str(context.get("ticker", "")).zfill(6)
        cached = _analysis_cache.get(ticker)
        if cached:
            score = cached.get("score", {})
            ai = cached.get("ai_analysis", {})
            acc = ai.get("매집가격대", {})
            tgt = ai.get("목표가격대", {})
            ctx_parts.append(f"""[현재 분석 종목]
종목: {ticker} ({cached.get('name', ticker)})
현재가: {cached.get('price', 0):,}원 ({cached.get('change_rate', 0):+.2f}%)
52주 범위: {cached.get('low52', 0):,} ~ {cached.get('high52', 0):,}원
분석 데이터: {cached.get('ohlcv_len', 0)}거래일

[세력 점수]
종합: {score.get('total', 0):.1f}/10 (등급: {score.get('grade', 'N/A')})
거래량이상: {score.get('scores', {}).get('A', 0):.1f} — {score.get('reasons', {}).get('A', '')}
가격/거래량: {score.get('scores', {}).get('B', 0):.1f} — {score.get('reasons', {}).get('B', '')}
수급신뢰도: {score.get('scores', {}).get('C', 0):.1f} — {score.get('reasons', {}).get('C', '')}
기술지표: {score.get('scores', {}).get('D', 0):.1f} — {score.get('reasons', {}).get('D', '')}
거래량프로파일: {score.get('scores', {}).get('E', 0):.1f} — {score.get('reasons', {}).get('E', '')}
박스권돌파: {score.get('scores', {}).get('F', 0):.1f} — {score.get('reasons', {}).get('F', '')}
업종모멘텀: {score.get('scores', {}).get('G', 0):.1f} — {score.get('reasons', {}).get('G', '')}
고래신호: {score.get('scores', {}).get('H', 0):.1f} — {score.get('reasons', {}).get('H', '')}

[AI 분석 결과]
세력단계: {ai.get('세력단계', 'N/A')}
장기패턴: {ai.get('장기패턴', 'N/A')}
신뢰도: {ai.get('신뢰도', 'N/A')}
펌핑가능성: {ai.get('펌핑가능성', False)}
진입추천: {ai.get('진입추천', False)}
핵심근거: {ai.get('핵심근거', 'N/A')}
매집가격대: {acc.get('하단', 'N/A')}~{acc.get('상단', 'N/A')}원 (추정: {acc.get('추정여부', False)})
목표가: 단기 {tgt.get('단기', 'N/A')}원 / 중기 {tgt.get('중기', 'N/A')}원

[최근 공시]
{', '.join(cached.get('recent_disclosures', ['없음']))}""")
        else:
            ctx_parts.append(f"[안내] {context.get('name', ticker)} ({ticker}) 종목의 분석 데이터가 없습니다. 사용자에게 'AI분석 탭에서 해당 종목을 검색하고 분석 버튼을 눌러주세요'라고 안내하세요.")

    ctx_str = "\n".join(ctx_parts)
    if ctx_str:
        ctx_str = ctx_str + "\n\n"

    messages.append({"role": "user", "content": ctx_str + question})

    system = """당신은 한국 주식 시장 전문 AI 애널리스트입니다.
세력 매매, 수급 분석, 기술적 분석, 공시 해석, 장기 매집/분산 패턴 분석을 전문으로 합니다.

★ 절대 규칙:
1. 메시지에 종목 데이터(현재가, 세력점수, 매집가격대, 목표가, 수급 등)가 포함되어 있으면 그 데이터를 즉시 사용하여 답변하라. 절대로 "데이터가 없다", "차트를 보내달라", "추가 정보가 필요하다" 같은 말을 하지 마라.
2. 매수 여부, 목표가, 손절가, 진입 시점을 물으면 제공된 데이터를 근거로 명확한 견해를 제시하라. 회피하지 마라.
3. 이미 분석된 데이터가 있으면 "정확한 분석을 위해 정보가 필요합니다" 같은 말은 절대 하지 마라.
4. 데이터가 전혀 없을 때(no data 상황)만 "AI분석 탭에서 해당 종목을 먼저 분석해주세요"라고 안내하라.

한국어로 명확하고 직접적으로 답변하라. 단락 구분을 명확히 하고 줄바꿈을 적절히 사용하라."""

    try:
        client = _get_anthropic_client()
        response = await asyncio.wait_for(
            client.messages.create(
                model=MODEL,
                max_tokens=1200,
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


async def analyze_portfolio(positions: list[dict]) -> list[dict]:
    results = []

    for pos in positions:
        ticker = str(pos.get("ticker", "")).zfill(6)
        avg_price = float(pos.get("avg_price", 0))
        quantity = int(pos.get("quantity", 0))

        try:
            from screening import get_stock_detail

            detail = await get_stock_detail(ticker)
            if not detail:
                results.append({
                    "ticker": ticker,
                    "name": ticker,
                    "avg_price": avg_price,
                    "current_price": 0,
                    "quantity": quantity,
                    "pnl_rate": 0,
                    "pnl_amount": 0,
                    "score": 0,
                    "grade": "D",
                    "recommendation": "조회불가",
                    "reason": "데이터 조회 실패",
                    "score_detail": {},
                })
                continue

            current_price = detail["price"]
            score_data = detail["score"]
            score = score_data.get("total", 0)
            grade = score_data.get("grade", "D")

            pnl_rate = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
            pnl_amount = (current_price - avg_price) * quantity

            recommendation, reason = _portfolio_recommendation(
                pnl_rate=pnl_rate,
                score=score,
                score_data=score_data,
            )

            results.append({
                "ticker": ticker,
                "name": detail["name"],
                "avg_price": avg_price,
                "current_price": current_price,
                "quantity": quantity,
                "pnl_rate": round(pnl_rate, 2),
                "pnl_amount": int(pnl_amount),
                "score": score,
                "grade": grade,
                "recommendation": recommendation,
                "reason": reason,
                "score_detail": score_data,
            })

        except Exception as e:
            logger.error("포트폴리오 분석 오류 %s: %s", ticker, e)
            results.append({
                "ticker": ticker,
                "name": ticker,
                "avg_price": avg_price,
                "current_price": 0,
                "quantity": quantity,
                "pnl_rate": 0,
                "pnl_amount": 0,
                "score": 0,
                "grade": "D",
                "recommendation": "오류",
                "reason": str(e),
                "score_detail": {},
            })

    return results


async def stream_market_summary(market_data: dict, whale_list: list):
    """오늘 시장 흐름 AI 요약 — 스트리밍"""
    from typing import AsyncGenerator
    client = _get_anthropic_client()

    kospi = market_data.get("kospi", {}) or {}
    kosdaq = market_data.get("kosdaq", {}) or {}
    is_open = market_data.get("is_open", False)

    kospi_idx = kospi.get("index") or "N/A"
    kospi_cr = float(kospi.get("change_rate") or 0)
    kospi_up = int(kospi.get("up") or 0)
    kospi_dn = int(kospi.get("down") or 0)

    kosdaq_idx = kosdaq.get("index") or "N/A"
    kosdaq_cr = float(kosdaq.get("change_rate") or 0)
    kosdaq_up = int(kosdaq.get("up") or 0)
    kosdaq_dn = int(kosdaq.get("down") or 0)

    status = "장중" if is_open else "장마감"

    if whale_list:
        whale_lines = "\n".join(
            f"- {w.get('name', w['ticker'])} ({w['ticker']}): {w['total_amount']/1e8:.1f}억원 [{w.get('top_level','SMALL')}]"
            for w in whale_list[:5]
        )
    else:
        whale_lines = "- 금일 고래 감지 없음"

    prompt = f"""아래 데이터를 기반으로 오늘 한국 주식시장 현황을 간결하게 분석해주세요.

[시장 현황 — {status}]
• KOSPI {kospi_idx} ({kospi_cr:+.2f}%) — 상승 {kospi_up}/ 하락 {kospi_dn}종목
• KOSDAQ {kosdaq_idx} ({kosdaq_cr:+.2f}%) — 상승 {kosdaq_up}/ 하락 {kosdaq_dn}종목

[고래 출몰 현황 상위]
{whale_lines}

다음 세 가지를 포함하되 250자 이내로 자연스럽게 서술하세요. 불릿 포인트 없이 한 문단으로:
1. 현재 시장 전반의 분위기 (리스크온/오프, 방향성, 강도)
2. 고래 출몰 종목 중 주목할 만한 포인트 (있다면)
3. 오늘 남은 시간 또는 내일을 위한 한 줄 전략"""

    async with client.messages.stream(
        model=MODEL,
        max_tokens=400,
        system="당신은 한국 주식시장 전문 AI 애널리스트입니다. 데이터를 간결하고 실용적으로 분석하며 핵심만 전달합니다.",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


def _portfolio_recommendation(pnl_rate: float, score: float, score_data: dict) -> tuple[str, str]:
    if pnl_rate <= -15:
        if score >= 5:
            return "물타기 고려", f"손실 {pnl_rate:.1f}%이나 세력점수 {score:.1f}로 회복 가능성. 분할 매수 검토."
        else:
            return "손절 고려", f"손실 {pnl_rate:.1f}%에 세력점수 {score:.1f}로 추가 하락 위험."
    elif pnl_rate <= -8:
        if score >= 6:
            return "물타기 고려", f"손실 {pnl_rate:.1f}%이나 세력점수 {score:.1f}(매력적). 분할 물타기 적합."
        elif score >= 4:
            return "보유", f"손실 {pnl_rate:.1f}%, 세력점수 {score:.1f}. 추가 매수보다 보유 후 관망."
        else:
            return "손절 고려", f"손실 {pnl_rate:.1f}%에 세력점수 {score:.1f}로 회복 근거 부족."
    elif pnl_rate <= -3:
        if score >= 5:
            return "추가매수 고려", f"소폭 손실({pnl_rate:.1f}%)에 세력점수 {score:.1f}. 단가 낮추기 유리."
        else:
            return "보유", f"손실 {pnl_rate:.1f}%. 세력점수 {score:.1f}로 지켜보기."
    elif pnl_rate >= 25:
        return "익절 고려", f"수익 {pnl_rate:.1f}%. 1/3~1/2 분할 익절 후 나머지 홀딩 전략."
    elif pnl_rate >= 15:
        if score >= 7:
            return "홀딩 (추가 상승 기대)", f"수익 {pnl_rate:.1f}%에 세력점수 {score:.1f}(강함). 목표가까지 보유."
        else:
            return "일부 익절", f"수익 {pnl_rate:.1f}%. 세력점수 {score:.1f}로 보수적 익절 권장."
    else:
        if score >= 7:
            return "추가매수 고려", f"수익 {pnl_rate:.1f}%에 세력점수 {score:.1f}(강함). 비중 확대 검토."
        elif score >= 5:
            return "보유", f"수익 {pnl_rate:.1f}%, 세력점수 {score:.1f}. 현 포지션 유지."
        else:
            return "보유 (주의)", f"수익 {pnl_rate:.1f}%이나 세력점수 {score:.1f}로 추가 매수 자제."
