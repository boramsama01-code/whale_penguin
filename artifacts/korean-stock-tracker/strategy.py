import logging
from typing import Optional

logger = logging.getLogger(__name__)


def generate_strategy(
    ticker: str,
    name: str,
    current_price: float,
    score: float,
    high52: float,
    low52: float,
    atr: Optional[float] = None,
    ai_targets: Optional[dict] = None,
    ai_analysis: Optional[dict] = None,
) -> dict:
    """
    자동매매 전략 생성 — 모든 항목 AI 분석 기반, 항목별 근거 포함.
    ai_targets: AI 분석에서 나온 {'단기': price, '중기': price, '근거': str}
    ai_analysis: 전체 AI 분석 결과 (매집가격대, 세력단계 등)
    """
    try:
        price = float(current_price)
        ai = ai_analysis or {}
        acc = ai.get("매집가격대", {}) or {}
        stage = ai.get("세력단계", "")
        tgt_data = ai.get("목표가격대", {}) or {}

        # ── 손절가 계산 ─────────────────────────────────────────────────────
        # 우선순위: ATR 2배 → 매집가격대 하단 → 현재가 -5%
        if atr and atr > 0:
            stop_distance = atr * 2.0
            stop_method = f"ATR({atr:,.0f}) × 2배 변동성 기반"
        else:
            stop_distance = price * 0.05
            stop_method = "현재가 5% 하단 기준"

        stop_loss = round(price - stop_distance, 0)
        stop_loss = max(stop_loss, price * 0.90)  # 최대 10% 손실

        # 매집가격대 하단이 있으면 더 논리적인 손절선으로
        acc_lower = None
        acc_upper = None
        try:
            if acc.get("추정여부") and acc.get("하단"):
                acc_lower = float(acc["하단"])
            if acc.get("상단"):
                acc_upper = float(acc["상단"])
        except Exception:
            pass

        if acc_lower and acc_lower < price:
            # 매집하단 -2% 를 손절선으로
            acc_stop = round(acc_lower * 0.98, 0)
            if acc_stop > stop_loss:
                stop_loss = acc_stop
                stop_method = f"AI 추정 매집가격대 하단({acc_lower:,.0f}원) -2% 이탈 기준"

        risk = price - stop_loss
        if risk <= 0:
            risk = price * 0.05
            stop_loss = round(price - risk, 0)

        stop_pct = risk / price * 100

        # ── 진입가 결정 ──────────────────────────────────────────────────────
        # 매집가격대 상단 ~ 현재가 사이 진입 권장
        if acc_upper and acc_upper <= price * 1.02:
            entry_min = round(acc_upper * 0.99, 0)
            entry_max = round(price * 1.005, 0)
            entry_reason = (
                f"AI 추정 매집가격대 상단({acc_upper:,.0f}원) 돌파 후 눌림목 구간 진입. "
                f"현재 {stage or '관망'} 단계 — 분할 매수(1차 {entry_min:,.0f}원, 2차 현재가 부근) 권장"
            )
        elif acc_lower and acc_lower < price:
            entry_min = round(price * 0.99, 0)
            entry_max = round(price * 1.01, 0)
            entry_reason = (
                f"AI 추정 매집가격대({acc_lower:,.0f}~{acc_upper:,.0f}원) 상단 돌파 후 진입 적기. "
                f"현재가({price:,.0f}원) ±1% 구간에서 분할 매수. {stage or ''} 단계"
            )
        else:
            entry_min = round(price * 0.99, 0)
            entry_max = round(price * 1.005, 0)
            rsi_note = "RSI 과매수 구간이면 눌림목 대기 권장." if score >= 7 else ""
            entry_reason = (
                f"현재가({price:,.0f}원) 기준 ±1% 진입 구간. "
                f"세력점수 {score:.1f}/10 — {'적극 매수 가능' if score >= 7 else '소량 선진입 후 추이 확인'}. "
                f"{rsi_note}"
            )

        # ── 목표가 결정 ──────────────────────────────────────────────────────
        rr_ratio = 2.5 if score >= 7 else (2.0 if score >= 5 else 1.5)

        ai_short = None
        ai_mid = None
        ai_tgt_basis = tgt_data.get("근거", "")
        try:
            v = ai_targets.get("단기") if ai_targets else tgt_data.get("단기")
            if v and float(v) > price:
                ai_short = float(v)
        except Exception:
            pass
        try:
            v = ai_targets.get("중기") if ai_targets else tgt_data.get("중기")
            if v and float(v) > price:
                ai_mid = float(v)
        except Exception:
            pass

        if ai_short:
            target1 = round(ai_short, 0)
            t1_upside = (target1 - price) / price * 100
            t1_reason = (
                f"단기 목표 {target1:,.0f}원 (현재가 대비 +{t1_upside:.1f}%). "
                f"{ai_tgt_basis or '전고점·저항선 기술적 분석 기반'}. "
                "도달 시 1/3 익절 후 나머지 보유 전략"
            )
            if ai_mid:
                target2 = round(ai_mid, 0)
                t2_upside = (target2 - price) / price * 100
                t2_reason = (
                    f"중기 목표 {target2:,.0f}원 (현재가 대비 +{t2_upside:.1f}%). "
                    f"52주 고점({high52:,.0f}원) 및 장기 추세선 기반. "
                    "목표가1 달성 후 홀딩, 추세 유지 시 목표"
                )
            else:
                target2 = round(ai_short + (ai_short - price) * 0.6, 0)
                t2_upside = (target2 - price) / price * 100
                t2_reason = (
                    f"중기 목표 {target2:,.0f}원 (단기 목표 기반 +{t2_upside:.1f}% 연장). "
                    "세력 분산 완료 시점 예상 구간"
                )
            actual_rr = round((target1 - price) / risk, 2) if risk > 0 else rr_ratio
        else:
            target1 = round(price + risk * rr_ratio, 0)
            target2 = round(price + risk * rr_ratio * 1.7, 0)
            if high52 > 0 and target1 > high52 * 1.1:
                target1 = round(high52 * 1.05, 0)
            t1_upside = (target1 - price) / price * 100
            t2_upside = (target2 - price) / price * 100
            t1_reason = (
                f"손익비 {rr_ratio:.1f}:1 기반 1차 목표 {target1:,.0f}원 (+{t1_upside:.1f}%). "
                f"{'52주 고점(' + str(int(high52)) + '원) 하단 저항 고려.' if high52 > 0 else ''} "
                "도달 시 1/3 익절 권장"
            )
            t2_reason = (
                f"2차 목표 {target2:,.0f}원 (+{t2_upside:.1f}%). "
                "1차 목표 달성 후 추세 지속 시 홀딩, 강한 추세에서 목표"
            )
            actual_rr = rr_ratio

        # ── 포지션 사이징 ────────────────────────────────────────────────────
        account_risk_pct = 0.01  # 계좌 1% 리스크
        account_size = 100_000_000  # 1억 기준
        risk_amount = account_size * account_risk_pct
        position_size = int(risk_amount / risk) if risk > 0 else 0
        position_amount = int(position_size * price)

        confidence = "높음" if score >= 7.5 else "보통" if score >= 5 else "낮음"
        pos_reason = (
            f"총 자산 1억 기준 1% 리스크({risk_amount:,.0f}원). "
            f"주당 리스크 {risk:,.0f}원으로 {position_size:,}주 산출. "
            f"신뢰도 {confidence} — {'3회 분할 매수 권장 (33%씩)' if score >= 5 else '소량 선진입 후 확인 후 추가 매수'}. "
            f"총 투입 예상 {position_amount:,}원"
        )

        stop_reason_text = (
            f"{stop_method}. "
            f"손절 시 주당 손실 {risk:,.0f}원 (진입가 대비 -{stop_pct:.1f}%). "
            "이탈 즉시 감정 배제하고 전량 손절"
        )

        liquidation_rules = [
            f"목표가1({target1:,.0f}원) 도달 — 보유량 1/3 익절 (손익비 {actual_rr:.1f}:1)",
            f"목표가2({target2:,.0f}원) 도달 — 추가 1/3 익절, 나머지 트레일링 스톱",
            f"손절가({stop_loss:,.0f}원) 이탈 — 전량 즉시 손절 (계좌 -{account_risk_pct*100:.0f}%)",
            "진입 후 3거래일 이내 방향 미확정 시 포지션 절반 정리",
            f"{'세력 분산 신호(거래량 급감+고점) 포착 시 조기 청산 고려' if score >= 7 else '세력점수 3점 이하 하락 시 재검토'}",
        ]

        return {
            "ticker": ticker,
            "name": name,
            "score": score,
            "entry": {
                "min": entry_min,
                "max": entry_max,
                "current": price,
            },
            "stop_loss": stop_loss,
            "targets": [target1, target2],
            "rr_ratio": actual_rr,
            "position": {
                "size": position_size,
                "amount": position_amount,
                "account_risk_pct": account_risk_pct * 100,
            },
            "risk_per_share": round(risk, 0),
            "reasons": {
                "entry": entry_reason,
                "stop_loss": stop_reason_text,
                "target1": t1_reason,
                "target2": t2_reason,
                "position": pos_reason,
            },
            "liquidation_rules": liquidation_rules,
            "52w_high": high52,
            "52w_low": low52,
        }

    except Exception as e:
        logger.error("전략 생성 오류: %s", e)
        return {
            "ticker": ticker,
            "name": name,
            "error": str(e),
        }
