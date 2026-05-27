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
) -> dict:
    """
    자동매매 전략 생성.
    ai_targets: AI 분석에서 나온 {'단기': price, '중기': price} — 있으면 우선 사용.
    """
    try:
        price = float(current_price)

        if atr and atr > 0:
            stop_distance = atr * 2.0
        else:
            stop_distance = price * 0.05

        stop_loss = round(price - stop_distance, 0)
        stop_loss = max(stop_loss, price * 0.92)

        risk = price - stop_loss
        if risk <= 0:
            risk = price * 0.05

        # ── 목표가 결정: AI 분석 목표가 우선, 없으면 기술적 계산 ────────────
        ai_short = None
        ai_mid = None
        if ai_targets:
            try:
                v = ai_targets.get("단기")
                if v and float(v) > price:
                    ai_short = float(v)
            except Exception:
                pass
            try:
                v = ai_targets.get("중기")
                if v and float(v) > price:
                    ai_mid = float(v)
            except Exception:
                pass

        # RR 비율 계산 (AI 목표가 없을 때 사용)
        rr_ratio = 2.5 if score >= 7 else (2.0 if score >= 5 else 1.5)

        if ai_short:
            target1 = round(ai_short, 0)
            # 중기가 없으면 단기에서 추가 상승 여지 계산
            if ai_mid:
                target2 = round(ai_mid, 0)
            else:
                target2 = round(ai_short + (ai_short - price) * 0.5, 0)
            target3 = round(price + risk * rr_ratio * 2.5, 0)
            # target3이 target2보다 낮으면 target2 기준으로 재계산
            if target3 <= target2:
                target3 = round(target2 * 1.15, 0)
            actual_rr = round((target1 - price) / risk, 2) if risk > 0 else rr_ratio
        else:
            target1 = round(price + risk * rr_ratio, 0)
            target2 = round(price + risk * rr_ratio * 1.5, 0)
            target3 = round(price + risk * rr_ratio * 2.5, 0)
            if high52 > 0 and target1 > high52 * 1.1:
                target1 = round(high52 * 1.05, 0)
            actual_rr = rr_ratio

        account_risk_pct = 0.01
        account_size = 100_000_000

        risk_amount = account_size * account_risk_pct
        position_size = int(risk_amount / risk) if risk > 0 else 0
        position_amount = int(position_size * price)

        entry_min = round(price * 0.99, 0)
        entry_max = round(price * 1.005, 0)

        source_label = "AI 분석 기반" if ai_short else f"손익비 {actual_rr:.1f}:1 기반"
        liquidation_rules = [
            f"목표가1 ({target1:,.0f}원) 도달 시 1/3 익절 [{source_label}]",
            f"목표가2 ({target2:,.0f}원) 도달 시 1/3 익절",
            f"목표가3 ({target3:,.0f}원) 도달 시 잔여 전량 청산",
            f"손절가 ({stop_loss:,.0f}원) 이탈 시 전량 손절",
            "진입 후 3일 이내 방향 미확정 시 1/2 청산",
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
            "targets": [target1, target2, target3],
            "rr_ratio": actual_rr,
            "ai_target_used": bool(ai_short),
            "position": {
                "size": position_size,
                "amount": position_amount,
                "account_risk_pct": account_risk_pct * 100,
            },
            "risk_per_share": round(risk, 0),
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
