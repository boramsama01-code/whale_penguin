import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── 보조 계산 함수 ─────────────────────────────────────────────────────────────

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


def calc_obv(ohlcv: pd.DataFrame) -> np.ndarray:
    """On-Balance Volume 계산."""
    close_col = _get_col(ohlcv, "종가", "Close")
    vol_col = _get_col(ohlcv, "거래량", "Volume")
    closes = ohlcv[close_col].values.astype(float)
    vols = ohlcv[vol_col].values.astype(float)
    obv = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + vols[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - vols[i])
        else:
            obv.append(obv[-1])
    return np.array(obv)


def _get_col(df: pd.DataFrame, *names) -> str:
    for n in names:
        if n in df.columns:
            return n
    return df.columns[0]


# ── 개별 점수 함수 ─────────────────────────────────────────────────────────────

def score_volume_anomaly(ohlcv: pd.DataFrame) -> tuple[float, str]:
    """A: 거래량 이상 탐지."""
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
    """B: 가격/거래량 괴리 — 횡보 중 거래량 급증 = 세력 진입 신호."""
    try:
        close_col = _get_col(ohlcv, "종가", "Close")
        vol_col = _get_col(ohlcv, "거래량", "Volume")
        closes = ohlcv[close_col].values.astype(float)
        vols = ohlcv[vol_col].values.astype(float)
        if len(closes) < 10:
            return 0.0, "데이터 부족"

        # 5거래일 가격 변화율
        price_change_5d = (closes[-1] - closes[-5]) / (closes[-5] + 1) * 100

        # 거래량 기준: 최근 3일 평균 vs 20일(또는 가용) 기준선
        baseline_len = min(len(vols) - 3, 20)
        recent_vol = np.mean(vols[-3:])
        if baseline_len > 0:
            baseline_vol = np.mean(vols[-(baseline_len + 3):-3])
        else:
            baseline_vol = np.mean(vols[:-3]) if len(vols) > 3 else 1.0
        if baseline_vol <= 0:
            baseline_vol = 1.0
        vol_ratio = recent_vol / baseline_vol  # 기준선 대비 몇 배

        # 거래량 단기 추세 (최근 5일 선형 기울기)
        vol_trend_up = False
        if len(vols) >= 8:
            x = np.arange(5, dtype=float)
            coef = np.polyfit(x, vols[-5:], 1)
            vol_trend_up = bool(coef[0] > 0)

        # 패턴 1: 횡보 + 거래량 급증 (가장 강한 매집 신호) — 최대 8점
        if abs(price_change_5d) <= 3.0 and vol_ratio >= 1.5:
            score = min(8.0, 5.0 + (vol_ratio - 1.5) * 2.0)
            return round(score, 1), f"횡보 중 거래량 급증 (가격변화 {price_change_5d:+.1f}%, 거래량 기준선 대비 {vol_ratio:.1f}배)"

        # 패턴 2: 가격-거래량 동반 상승 (돌파 신호) — 최대 6점
        if price_change_5d > 0.5 and vol_ratio >= 1.3:
            score = min(6.0, 3.0 + (vol_ratio - 1.0) * 1.5)
            return round(score, 1), f"가격-거래량 동반 상승 ({price_change_5d:+.1f}%, 거래량 {vol_ratio:.1f}배)"

        # 패턴 3: 거래량 추세 상승 (점진 매집) — 3점
        if vol_trend_up and vol_ratio >= 1.15:
            return 3.0, f"거래량 상승 추세 형성 (5일 기울기 상향, 기준선 {vol_ratio:.1f}배)"

        # 패턴 4: 거래량 소폭 상회 — 1.5점
        if vol_ratio >= 1.2:
            return 1.5, f"거래량 기준선 상회 ({vol_ratio:.1f}배, 가격 {price_change_5d:+.1f}%)"

        return 0.0, f"정상 거래량 (기준선 {vol_ratio:.1f}배, 가격 {price_change_5d:+.1f}%)"
    except Exception as e:
        return 0.0, f"오류: {e}"


def _score_supply_obv(ohlcv: pd.DataFrame, cap_tier: str = "SMALL") -> tuple[float, str]:
    """OBV 기반 수급 신뢰도 (소형/중형주 대체 지표)."""
    try:
        close_col = _get_col(ohlcv, "종가", "Close")
        vol_col = _get_col(ohlcv, "거래량", "Volume")

        closes = ohlcv[close_col].values.astype(float)
        vols = ohlcv[vol_col].values.astype(float)

        if len(closes) < 20:
            return 0.0, "OBV 데이터 부족"

        obv = calc_obv(ohlcv)
        score = 0.0
        reasons = []

        # 1. OBV 20일 선형 추세
        obv_20 = obv[-20:]
        obv_slope = np.polyfit(range(len(obv_20)), obv_20, 1)[0]
        vol_mean = np.mean(vols[-20:])
        obv_slope_norm = obv_slope / (vol_mean + 1)

        if obv_slope_norm > 0.3:
            score += 5.0
            reasons.append("OBV 강한 상승")
        elif obv_slope_norm > 0.1:
            score += 3.0
            reasons.append("OBV 상승 추세")
        elif obv_slope_norm > 0:
            score += 1.0
            reasons.append("OBV 완만한 상승")

        # 2. 상승일 vs 하락일 평균 거래량 비율 (U/D Ratio)
        up_vols = [vols[i] for i in range(1, len(closes)) if closes[i] >= closes[i - 1]]
        dn_vols = [vols[i] for i in range(1, len(closes)) if closes[i] < closes[i - 1]]
        if up_vols and dn_vols:
            up_avg = float(np.mean(up_vols[-20:] if len(up_vols) >= 20 else up_vols))
            dn_avg = float(np.mean(dn_vols[-20:] if len(dn_vols) >= 20 else dn_vols))
            ud_ratio = up_avg / (dn_avg + 1)
            if ud_ratio > 2.0:
                score += 3.0
                reasons.append(f"매수세 강함 U/D {ud_ratio:.1f}x")
            elif ud_ratio > 1.3:
                score += 1.5
                reasons.append(f"매수세 우위 U/D {ud_ratio:.1f}x")

        # 3. 최근 5일 양봉 비율
        if len(closes) >= 5 and ("시가" in ohlcv.columns or "Open" in ohlcv.columns):
            open_col = _get_col(ohlcv, "시가", "Open")
            opens = ohlcv[open_col].values.astype(float)
            recent_c = closes[-5:]
            recent_o = opens[-5:]
            bullish = int(np.sum(recent_c > recent_o))
            if bullish >= 4:
                score += 2.0
                reasons.append(f"최근 5일 양봉 {bullish}일")
            elif bullish >= 3:
                score += 1.0
                reasons.append(f"최근 5일 양봉 {bullish}일")

        prefix = f"[{cap_tier}]"
        label = f"{prefix} " + " | ".join(reasons) if reasons else f"{prefix} OBV 중립"
        return min(score, 10.0), label
    except Exception as e:
        return 0.0, f"OBV 오류: {e}"


def score_supply_reliability(
    supply_df: Optional[pd.DataFrame],
    ohlcv: Optional[pd.DataFrame] = None,
    mktcap: float = 0.0,
) -> tuple[float, str]:
    """
    C: 수급 신뢰도 — 시가총액 등급에 따라 분기.
    - LARGE (>1조): 기관/외국인 20일 순매수 분석
    - MID  (1000억~1조): 기관 중심 + OBV 보완
    - SMALL (<1000억): OBV + 연속양봉 패턴 (기관/외국인 미미)
    """
    LARGE_THRESH = 1_000_000_000_000
    MID_THRESH = 100_000_000_000

    cap_tier = (
        "LARGE" if mktcap >= LARGE_THRESH
        else "MID" if mktcap >= MID_THRESH
        else "SMALL"
    )

    # SMALL cap 또는 수급 데이터 없으면 OBV 기반
    if cap_tier == "SMALL" or supply_df is None or supply_df.empty:
        if ohlcv is not None and not ohlcv.empty and len(ohlcv) >= 20:
            return _score_supply_obv(ohlcv, cap_tier)
        return 0.0, f"수급 데이터 없음 [{cap_tier}]"

    # LARGE / MID: 기관/외국인 분석
    try:
        inst_col = None
        foreign_col = None
        for c in supply_df.columns:
            if "기관" in str(c):
                inst_col = c
            if "외국인" in str(c):
                foreign_col = c

        if inst_col is None and foreign_col is None:
            if ohlcv is not None and not ohlcv.empty:
                return _score_supply_obv(ohlcv, cap_tier)
            return 0.0, f"수급 컬럼 없음 ({list(supply_df.columns)[:3]})"

        inst = supply_df[inst_col].values.astype(float) if inst_col else np.zeros(len(supply_df))
        foreign = supply_df[foreign_col].values.astype(float) if foreign_col else np.zeros(len(supply_df))

        n = min(20, len(inst))
        inst_sum = float(np.sum(inst[-n:]))
        foreign_sum = float(np.sum(foreign[-n:]))
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

        if len(inst) >= 60:
            inst_60 = float(np.sum(inst[-60:]))
            if inst_60 > 0:
                score = min(score + 1.0, 10.0)
                reasons.append("60일 기관 누적")

        # MID cap은 OBV로 보완
        if cap_tier == "MID" and ohlcv is not None and not ohlcv.empty:
            obv_score, obv_reason = _score_supply_obv(ohlcv, "MID")
            score = min((score + obv_score) / 2.0 * 1.2, 10.0)
            if "OBV" in obv_reason or "U/D" in obv_reason:
                reasons.append(obv_reason)

        return min(score, 10.0), " | ".join(reasons) if reasons else "수급 중립"
    except Exception as e:
        return 0.0, f"오류: {e}"


def score_technical(ohlcv: pd.DataFrame) -> tuple[float, str]:
    """D: 기술지표 (RSI, MACD, MA, 볼린저)."""
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
    """E: 거래량 프로파일 — 최근 거래량 집중도."""
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
    """F: 박스권 돌파 감지."""
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
    """G: 코스피 대비 상대 모멘텀 (20일 + 60일 트렌드)."""
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
        else:
            reasons.append(f"시장 대비 {relative_20d:.1f}%")

        if len(closes) >= 60:
            stock_60d = (closes[-1] / closes[-60] - 1) * 100
            if stock_60d > 20:
                score = min(score + 2.0, 10.0)
                reasons.append(f"60일 +{stock_60d:.0f}%")

        return min(score, 10.0), " | ".join(reasons) if reasons else "모멘텀 중립"
    except Exception as e:
        return 0.0, f"오류: {e}"


def score_whale_signal(whale_data: Optional[dict]) -> tuple[float, str]:
    """H: 실시간 고래(세력) 신호."""
    if not whale_data:
        return 0.0, "고래 신호 없음"
    level = whale_data.get("level", whale_data.get("top_level", ""))
    amount = whale_data.get("total_amount", whale_data.get("accumulated_5m", 0))
    if level == "EMERGENCY":
        return 10.0, f"긴급 고래 {amount / 1e8:.1f}억"
    elif level == "LARGE":
        return 8.0, f"대형 고래 {amount / 1e8:.1f}억"
    elif level == "MEDIUM":
        return 5.0, f"중형 고래 감지 {amount / 1e7:.0f}천만"
    elif level == "SMALL":
        return 2.0, "소형 고래 감지"
    return 0.0, "고래 신호 없음"


def score_early_accumulation(
    ohlcv: pd.DataFrame,
    low52: float = 0.0,
    high52: float = 0.0,
) -> tuple[float, str]:
    """
    I: 초기 매집 구간 탐지 (사전 포착 신호).
    - 52주 저가 근처 횡보
    - 거래량 소량 지속 증가 (스텔스 매집)
    - 저점 상승 패턴 (Higher Lows)
    - 볼린저 수축 (에너지 응축)
    """
    try:
        close_col = _get_col(ohlcv, "종가", "Close")
        vol_col = _get_col(ohlcv, "거래량", "Volume")
        low_col = _get_col(ohlcv, "저가", "Low")

        closes = ohlcv[close_col].values.astype(float)
        vols = ohlcv[vol_col].values.astype(float)
        lows = ohlcv[low_col].values.astype(float)

        if len(closes) < 20:
            return 0.0, "데이터 부족"

        current = closes[-1]
        score = 0.0
        reasons = []

        # 1. 52주 저가 근처 (포지션 비율)
        if low52 > 0 and high52 > low52:
            range_52 = high52 - low52
            pos_from_low = (current - low52) / range_52 * 100
            if pos_from_low <= 15:
                score += 4.0
                reasons.append(f"52주저가 근처 {pos_from_low:.0f}% 위치")
            elif pos_from_low <= 30:
                score += 2.0
                reasons.append(f"52주저가 부근 {pos_from_low:.0f}% 위치")

        # 2. 20일 가격 횡보 (변동폭 낮음 = 조용한 매집)
        if len(closes) >= 20:
            price_range_20 = (
                (np.max(closes[-20:]) - np.min(closes[-20:])) / (np.mean(closes[-20:]) + 1) * 100
            )
            if price_range_20 < 8:
                score += 2.0
                reasons.append(f"20일 횡보 변동폭 {price_range_20:.1f}%")
            elif price_range_20 < 15:
                score += 1.0
                reasons.append(f"20일 소폭 변동 {price_range_20:.1f}%")

        # 3. 거래량 점진적 증가 (스텔스 매집 패턴)
        #    조건: 전반부 대비 후반부 거래량 10~150% 증가 (폭발 제외)
        if len(vols) >= 20:
            early_vol = np.mean(vols[-20:-10])
            recent_vol = np.mean(vols[-10:])
            vol_trend = (recent_vol - early_vol) / (early_vol + 1) * 100
            if 10 <= vol_trend <= 150:
                score += 2.0
                reasons.append(f"거래량 점진 증가 +{vol_trend:.0f}%")

        # 4. 저점 상승 패턴 (Higher Lows — 5일 단위 4분위 비교)
        if len(lows) >= 20:
            lows_20 = lows[-20:]
            q1 = float(np.min(lows_20[:5]))
            q2 = float(np.min(lows_20[5:10]))
            q3 = float(np.min(lows_20[10:15]))
            q4 = float(np.min(lows_20[15:]))
            if q4 > q3 > q2 > q1:
                score += 3.0
                reasons.append("연속 저점 상승 패턴")
            elif q4 > q3 and q4 > q2:
                score += 1.5
                reasons.append("저점 부분 상승")

        # 5. 볼린저 수축 (에너지 응축)
        _, upper, lower = calc_bollinger(closes)
        bw = (upper - lower) / ((upper + lower) / 2 + 1) * 100
        if bw < 4:
            score += 1.0
            reasons.append(f"볼린저 극도 수축 {bw:.1f}%")

        return min(score, 10.0), " | ".join(reasons) if reasons else "초기 매집 신호 없음"
    except Exception as e:
        return 0.0, f"오류: {e}"


# ── 종합 점수 계산 ─────────────────────────────────────────────────────────────

def calculate_score(
    ohlcv: pd.DataFrame,
    supply_df: Optional[pd.DataFrame] = None,
    market_change_20d: float = 0.0,
    whale_data: Optional[dict] = None,
    mktcap: float = 0.0,
    low52: float = 0.0,
    high52: float = 0.0,
) -> dict:
    """
    종합 세력+퀀트 점수 계산.

    가중치 설계 (설계가이드 기반):
      A 거래량이상  × 2.0   — 가장 직접적인 세력 진입 신호
      B 가격/거래량 × 1.5   — 횡보 중 거래량 = 조용한 매집
      C 수급신뢰도  × 2.0   — 기관/외국인 or OBV (시총 계층 분기)
      D 기술지표    × 1.0   — RSI·MACD·MA 정배열
      E 거래량프로파일× 1.0  — 최근 거래량 집중도
      F 박스돌파    × 2.0   — 돌파 시 폭발적 수익 가능성
      G 상대모멘텀  × 0.5   — 시장 대비 강도
      H 고래신호    × 2.0   — 실시간 대형 매수 감지
      I 초기매집    × 1.5   — 사전 매집 구간 포착 (신규)
    """
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}

    scores["A"], reasons["A"] = score_volume_anomaly(ohlcv)
    scores["B"], reasons["B"] = score_price_volume_divergence(ohlcv)
    scores["C"], reasons["C"] = score_supply_reliability(supply_df, ohlcv, mktcap)
    scores["D"], reasons["D"] = score_technical(ohlcv)
    scores["E"], reasons["E"] = score_volume_profile(ohlcv)
    scores["F"], reasons["F"] = score_box_breakout(ohlcv)
    scores["G"], reasons["G"] = score_relative_momentum(ohlcv, market_change_20d)
    scores["H"], reasons["H"] = score_whale_signal(whale_data)
    scores["I"], reasons["I"] = score_early_accumulation(ohlcv, low52, high52)

    weights = {
        "A": 2.0,
        "B": 1.5,
        "C": 2.0,
        "D": 1.0,
        "E": 1.0,
        "F": 2.0,
        "G": 0.5,
        "H": 2.0,
        "I": 1.5,
    }

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
    if score >= 8.0:
        return "S"
    elif score >= 6.0:
        return "A"
    elif score >= 4.0:
        return "B"
    elif score >= 2.0:
        return "C"
    return "D"
