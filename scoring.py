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
        vol_col = _get_col(ohlcv, "거래량", "Volume")
        vols = ohlcv[vol_col].values.astype(float)
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
        close_col = _get_col(ohlcv, "종가", "Close")
        vol_col = _get_col(ohlcv, "거래량", "Volume")
        closes = ohlcv[close_col].values.astype(float)
        vols = ohlcv[vol_col].values.astype(float)
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
        # pykrx 컬럼명 매핑 (여러 버전 대응)
        inst_col = None
        foreign_col = None
        for c in supply_df.columns:
            if "기관" in str(c):
                inst_col = c
            if "외국인" in str(c):
                foreign_col = c

        if inst_col is None and foreign_col is None:
            return 0.0, f"수급 컬럼 없음 ({list(supply_df.columns)[:3]})"

        inst = supply_df[inst_col].values.astype(float) if inst_col else np.zeros(len(supply_df))
        foreign = supply_df[foreign_col].values.astype(float) if foreign_col else np.zeros(len(supply_df))

        # 최근 20일 합산
        n = min(20, len(inst))
        inst_sum = float(np.sum(inst[-n:]))
        foreign_sum = float(np.sum(foreign[-n:]))

        # 매집 강도 — 연속 순매수일 수 계산
        inst_days = int(np.sum(inst[-n:] > 0))
        foreign_days = int(np.sum(foreign[-n:] > 0))

        score = 0.0
        reasons = []
        if inst_sum > 0:
            score += 3.0
            reasons.append(f"기관 순매수 {inst_days}일/{n}일")
        if foreign_sum > 0:
            score += 3.0
            reasons.append(f"외국인 순매수 {foreign_days}일/{n}일")
        if inst_sum > 0 and foreign_sum > 0:
            score += 4.0
            reasons.append("기관+외국인 동반")

        # 장기(60일) 추세 보강
        if len(inst) >= 60:
            inst_60 = float(np.sum(inst[-60:]))
            if inst_60 > 0:
                score = min(score + 1.0, 10.0)
                reasons.append("60일 기관 누적 순매수")

        return min(score, 10.0), " | ".join(reasons) if reasons else "수급 중립"
    except Exception as e:
        return 0.0, f"오류: {e}"


def score_technical(ohlcv: pd.DataFrame) -> tuple[float, str]:
    try:
        close_col = _get_col(ohlcv, "종가", "Close")
        closes = ohlcv[close_col].values.astype(float)
        score = 0.0
        reasons = []

        rsi = calc_rsi(closes)
        if 40 <= rsi <= 65:
            score += 3.0
            reasons.append(f"RSI {rsi:.1f} (적정)")
        elif rsi < 35:
            score += 1.0
            reasons.append(f"RSI {rsi:.1f} (과매도)")

        macd, signal = calc_macd(closes)
        if macd > signal and macd > 0:
            score += 3.0
            reasons.append("MACD 골든크로스")
        elif macd > signal:
            score += 1.5
            reasons.append("MACD 상향")

        if len(closes) >= 20:
            ma5 = np.mean(closes[-5:])
            ma20 = np.mean(closes[-20:])
            if closes[-1] > ma5 > ma20:
                score += 2.0
                reasons.append("5>20MA 정배열")
            elif closes[-1] > ma20:
                score += 1.0
                reasons.append("MA20 위")

        _, upper, lower = calc_bollinger(closes)
        bw = (upper - lower) / ((upper + lower) / 2) * 100
        if bw < 5:
            score += 2.0
            reasons.append(f"볼린저 수축 {bw:.1f}%")

        return min(score, 10.0), " | ".join(reasons) if reasons else "기술지표 중립"
    except Exception as e:
        return 0.0, f"오류: {e}"


def score_volume_profile(ohlcv: pd.DataFrame) -> tuple[float, str]:
    try:
        vol_col = _get_col(ohlcv, "거래량", "Volume")
        close_col = _get_col(ohlcv, "종가", "Close")
        closes = ohlcv[close_col].values.astype(float)
        vols = ohlcv[vol_col].values.astype(float)
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
        close_col = _get_col(ohlcv, "종가", "Close")
        closes = ohlcv[close_col].values.astype(float)
        if len(closes) < 20:
            return 0.0, "데이터 부족"
        box_high = np.max(closes[-20:-1])
        current = closes[-1]
        if current > box_high * 1.02:
            gain = (current / box_high - 1) * 100
            return 10.0, f"박스권 돌파 +{gain:.1f}%"
        elif current > box_high * 0.99:
            return 5.0, "박스권 저항 테스트"
        return 0.0, "박스권 내 횡보"
    except Exception as e:
        return 0.0, f"오류: {e}"


def score_relative_momentum(ohlcv: pd.DataFrame, market_change_20d: float = 0.0) -> tuple[float, str]:
    """
    시장(코스피) 대비 상대 모멘텀으로 업종 모멘텀(G) 대체.
    20일 상대 성과 + 60일 트렌드 반영.
    """
    try:
        close_col = _get_col(ohlcv, "종가", "Close")
        closes = ohlcv[close_col].values.astype(float)

        if len(closes) < 20:
            return 0.0, "데이터 부족"

        stock_20d = (closes[-1] / closes[-20] - 1) * 100
        relative_20d = stock_20d - market_change_20d

        score = 0.0
        reasons = []

        if relative_20d > 15:
            score += 8.0
            reasons.append(f"시장 대비 +{relative_20d:.1f}% 강세")
        elif relative_20d > 5:
            score += 5.0
            reasons.append(f"시장 대비 +{relative_20d:.1f}% 상회")
        elif relative_20d > 0:
            score += 2.0
            reasons.append(f"시장 대비 +{relative_20d:.1f}%")
        elif relative_20d < -10:
            score += 0.0
            reasons.append(f"시장 대비 {relative_20d:.1f}% 약세")
        else:
            score += 0.0
            reasons.append(f"시장 대비 {relative_20d:.1f}%")

        # 60일 트렌드 보강
        if len(closes) >= 60:
            stock_60d = (closes[-1] / closes[-60] - 1) * 100
            if stock_60d > 20:
                score = min(score + 2.0, 10.0)
                reasons.append(f"60일 +{stock_60d:.0f}%")

        return min(score, 10.0), " | ".join(reasons) if reasons else "모멘텀 중립"
    except Exception as e:
        return 0.0, f"오류: {e}"


def score_whale_signal(whale_data: Optional[dict]) -> tuple[float, str]:
    if not whale_data:
        return 0.0, "고래 신호 없음"
    level = whale_data.get("level", whale_data.get("top_level", ""))
    amount = whale_data.get("total_amount", whale_data.get("accumulated_5m", 0))
    if level == "EMERGENCY":
        return 10.0, f"긴급 고래 {amount/1e8:.1f}억"
    elif level == "LARGE":
        return 8.0, f"대형 고래 {amount/1e8:.1f}억"
    elif level == "MEDIUM":
        return 5.0, f"중형 고래 {amount/1e7:.0f}천만"
    elif level == "SMALL":
        return 2.0, f"소형 고래 감지"
    return 0.0, "고래 신호 없음"


def _get_col(df: pd.DataFrame, *names) -> str:
    for n in names:
        if n in df.columns:
            return n
    return df.columns[0]


def calculate_score(
    ohlcv: pd.DataFrame,
    supply_df: Optional[pd.DataFrame] = None,
    market_change_20d: float = 0.0,
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
    scores["G"], reasons["G"] = score_relative_momentum(ohlcv, market_change_20d)
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
