import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calc_macd(closes: np.ndarray) -> tuple[float, float]:
    if len(closes) < 26:
        return 0.0, 0.0
    def ema(data, span):
        k = 2 / (span + 1)
        result = [data[0]]
        for v in data[1:]:
            result.append(v * k + result[-1] * (1 - k))
        return np.array(result)
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line = ema12 - ema26
    if len(macd_line) < 9:
        return float(macd_line[-1]), 0.0
    signal = ema(macd_line, 9)
    return float(macd_line[-1]), float(signal[-1])


def calc_bollinger(closes: np.ndarray, period: int = 20) -> tuple[float, float, float]:
    if len(closes) < period:
        mid = float(closes[-1])
        return mid, mid + 1, mid - 1
    window = closes[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window))
    return mid, mid + 2 * std, mid - 2 * std


def score_volume_anomaly(ohlcv: pd.DataFrame) -> tuple[float, str]:
    try:
        vols = ohlcv["거래량"].values.astype(float)
        if len(vols) < 20:
            return 0.0, "거래량 데이터 부족"
        avg20 = np.mean(vols[-20:])
        today = vols[-1]
        ratio = today / avg20 if avg20 > 0 else 0
        if ratio >= 5:
            return 10.0, f"거래량 평균 {ratio:.1f}배 (매우 이상)"
        elif ratio >= 3:
            return 7.0, f"거래량 평균 {ratio:.1f}배 (이상)"
        elif ratio >= 2:
            return 4.0, f"거래량 평균 {ratio:.1f}배 (증가)"
        else:
            return 0.0, f"거래량 평균 {ratio:.1f}배 (정상)"
    except Exception as e:
        return 0.0, f"오류: {e}"


def score_price_volume_divergence(ohlcv: pd.DataFrame) -> tuple[float, str]:
    try:
        closes = ohlcv["종가"].values.astype(float)
        vols = ohlcv["거래량"].values.astype(float)
        if len(closes) < 5:
            return 0.0, "데이터 부족"
        price_change = (closes[-1] - closes[-5]) / closes[-5] * 100
        vol_change = (np.mean(vols[-3:]) - np.mean(vols[-8:-3])) / (np.mean(vols[-8:-3]) + 1) * 100
        if price_change < 2 and vol_change > 50:
            return 8.0, f"횡보 중 거래량 급증 (가격변화 {price_change:.1f}%, 거래량 {vol_change:.1f}%)"
        elif price_change > 0 and vol_change > 30:
            return 5.0, f"가격-거래량 동반 상승 ({price_change:.1f}%, {vol_change:.1f}%)"
        else:
            return 0.0, f"패턴 미감지 (가격 {price_change:.1f}%, 거래량 {vol_change:.1f}%)"
    except Exception as e:
        return 0.0, f"오류: {e}"


def score_supply_reliability(supply_df: Optional[pd.DataFrame]) -> tuple[float, str]:
    if supply_df is None or supply_df.empty:
        return 0.0, "수급 데이터 없음"
    try:
        if "기관합계" not in supply_df.columns and "외국인합계" not in supply_df.columns:
            return 0.0, "수급 컬럼 없음"
        inst = supply_df.get("기관합계", pd.Series([0] * len(supply_df))).values.astype(float)
        foreign = supply_df.get("외국인합계", pd.Series([0] * len(supply_df))).values.astype(float)
        inst_sum = np.sum(inst[-10:])
        foreign_sum = np.sum(foreign[-10:])
        score = 0.0
        reasons = []
        if inst_sum > 0:
            score += 4.0
            reasons.append(f"기관 순매수 {inst_sum/1e4:.0f}만주")
        if foreign_sum > 0:
            score += 4.0
            reasons.append(f"외국인 순매수 {foreign_sum/1e4:.0f}만주")
        if inst_sum > 0 and foreign_sum > 0:
            score += 2.0
            reasons.append("동반 매수")
        return min(score, 10.0), " | ".join(reasons) if reasons else "수급 중립"
    except Exception as e:
        return 0.0, f"오류: {e}"


def score_technical(ohlcv: pd.DataFrame) -> tuple[float, str]:
    try:
        closes = ohlcv["종가"].values.astype(float)
        score = 0.0
        reasons = []

        rsi = calc_rsi(closes)
        if 40 <= rsi <= 65:
            score += 3.0
            reasons.append(f"RSI {rsi:.1f} (적정 구간)")
        elif rsi < 35:
            score += 1.0
            reasons.append(f"RSI {rsi:.1f} (과매도 근접)")

        macd, signal = calc_macd(closes)
        if macd > signal and macd > 0:
            score += 3.0
            reasons.append("MACD 골든크로스")
        elif macd > signal:
            score += 1.5
            reasons.append("MACD 상향 크로스")

        if len(closes) >= 20:
            ma5 = np.mean(closes[-5:])
            ma20 = np.mean(closes[-20:])
            if closes[-1] > ma5 > ma20:
                score += 2.0
                reasons.append("정배열 (5>20MA)")
            elif closes[-1] > ma20:
                score += 1.0
                reasons.append("MA20 위")

        _, upper, lower = calc_bollinger(closes)
        bw = (upper - lower) / ((upper + lower) / 2) * 100
        if bw < 5:
            score += 2.0
            reasons.append(f"볼린저 수축 {bw:.1f}% (폭발 예고)")

        return min(score, 10.0), " | ".join(reasons) if reasons else "기술지표 중립"
    except Exception as e:
        return 0.0, f"오류: {e}"


def score_volume_profile(ohlcv: pd.DataFrame) -> tuple[float, str]:
    try:
        closes = ohlcv["종가"].values.astype(float)
        vols = ohlcv["거래량"].values.astype(float)
        if len(closes) < 20:
            return 0.0, "데이터 부족"
        recent_avg_vol = np.mean(vols[-5:])
        past_avg_vol = np.mean(vols[-20:-5])
        ratio = recent_avg_vol / (past_avg_vol + 1)
        if ratio > 2:
            return 8.0, f"최근 거래량 집중 ({ratio:.1f}x)"
        elif ratio > 1.5:
            return 5.0, f"거래량 증가 추세 ({ratio:.1f}x)"
        return 0.0, "거래량 집중 없음"
    except Exception as e:
        return 0.0, f"오류: {e}"


def score_box_breakout(ohlcv: pd.DataFrame) -> tuple[float, str]:
    try:
        closes = ohlcv["종가"].values.astype(float)
        highs = ohlcv["고가"].values.astype(float)
        if len(closes) < 20:
            return 0.0, "데이터 부족"
        box_high = np.max(closes[-20:-1])
        current = closes[-1]
        if current > box_high * 1.02:
            gain = (current / box_high - 1) * 100
            return 10.0, f"박스권 돌파 +{gain:.1f}%"
        elif current > box_high * 0.99:
            return 5.0, "박스권 저항 테스트 중"
        return 0.0, "박스권 내 횡보"
    except Exception as e:
        return 0.0, f"오류: {e}"


def score_sector_momentum(sector: str, sector_change: float) -> tuple[float, str]:
    if sector_change > 3:
        return 8.0, f"업종 강세 +{sector_change:.1f}%"
    elif sector_change > 1:
        return 4.0, f"업종 상승 +{sector_change:.1f}%"
    elif sector_change < -2:
        return -2.0, f"업종 약세 {sector_change:.1f}%"
    return 0.0, f"업종 중립 {sector_change:.1f}%"


def score_whale_signal(whale_data: Optional[dict]) -> tuple[float, str]:
    if not whale_data:
        return 0.0, "고래 신호 없음"
    level = whale_data.get("level", "")
    amount = whale_data.get("total_amount", 0)
    if level == "EMERGENCY":
        return 10.0, f"긴급 고래 신호 {amount/1e8:.1f}억"
    elif level == "LARGE":
        return 8.0, f"대형 고래 {amount/1e8:.1f}억"
    elif level == "MEDIUM":
        return 5.0, f"중형 고래 {amount/1e8:.1f}억"
    elif level == "SMALL":
        return 2.0, f"소형 고래 {amount/1e8:.1f}억"
    return 0.0, "고래 신호 없음"


def calculate_score(
    ohlcv: pd.DataFrame,
    supply_df: Optional[pd.DataFrame] = None,
    sector: str = "",
    sector_change: float = 0.0,
    whale_data: Optional[dict] = None,
) -> dict:
    scores = {}
    reasons = {}

    scores["A"], reasons["A"] = score_volume_anomaly(ohlcv)
    scores["B"], reasons["B"] = score_price_volume_divergence(ohlcv)
    scores["C"], reasons["C"] = score_supply_reliability(supply_df)
    scores["D"], reasons["D"] = score_technical(ohlcv)
    scores["E"], reasons["E"] = score_volume_profile(ohlcv)
    scores["F"], reasons["F"] = score_box_breakout(ohlcv)
    scores["G"], reasons["G"] = score_sector_momentum(sector, sector_change)
    scores["H"], reasons["H"] = score_whale_signal(whale_data)

    weights = {"A": 2, "B": 1.5, "C": 1.5, "D": 1, "E": 1, "F": 2, "G": 0.5, "H": 2}
    total_weight = sum(weights.values())
    weighted_sum = sum(scores[k] * weights[k] for k in scores)
    total_score = round(weighted_sum / total_weight, 2)

    return {
        "total": total_score,
        "max": 10.0,
        "scores": scores,
        "reasons": reasons,
        "weights": weights,
        "grade": _grade(total_score),
    }


def _grade(score: float) -> str:
    if score >= 8:
        return "S"
    elif score >= 6:
        return "A"
    elif score >= 4:
        return "B"
    elif score >= 2:
        return "C"
    return "D"
