import pandas as pd
import numpy as np
import talib

def calculate_cpr(df_prev_day):
    """Calculates Central Pivot Range (CPR) and standard pivots."""
    if df_prev_day.empty:
        return {}
    high = df_prev_day['high'].iloc[-1]
    low = df_prev_day['low'].iloc[-1]
    close = df_prev_day['close'].iloc[-1]

    pivot = (high + low + close) / 3
    bc = (high + low) / 2
    tc = (pivot - bc) + pivot

    r1 = (2 * pivot) - low; s1 = (2 * pivot) - high
    r2 = pivot + (high - low); s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low); s3 = low - 2 * (high - pivot)

    pivots = {'pivot': pivot, 'bc': bc, 'tc': tc, 'r1': r1, 'r2': r2, 'r3': r3, 's1': s1, 's2': s2, 's3': s3}
    if pivots['tc'] < pivots['bc']:
        pivots['tc'], pivots['bc'] = pivots['bc'], pivots['tc']
    return pivots

def calculate_ema(prices, period):
    """Calculates the Exponential Moving Average (EMA)."""
    return talib.EMA(prices, timeperiod=period)

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    return talib.RSI(prices, timeperiod=period)

def check_ema_crossover(df, current_candle, last_candle, period):
    """Checks for a bullish or bearish EMA crossover for two consecutive candles."""
    ema_col = f'ema_{period}'
    price = current_candle['close']
    last_price = last_candle['close']
    ema_val = current_candle[ema_col]
    last_ema_val = last_candle[ema_col]
    
    # Bullish Crossover: Price crossed above EMA and stayed above
    if price > ema_val and last_price > last_ema_val:
        return "Bullish"
    # Bearish Crossover: Price crossed below EMA and stayed below
    if price < ema_val and last_price < last_ema_val:
        return "Bearish"
    
    return "None"

def check_rsi_divergence(price_df, rsi_series):
    """Simplified check for bullish/bearish RSI divergence."""
    period = -30
    low_prices = price_df['low'][period:]; high_prices = price_df['high'][period:]
    rsi_values = rsi_series[period:]

    if rsi_values.empty or len(rsi_values) < 2: return "None"

    if low_prices.iloc[-1] < low_prices.iloc[:-1].min() and rsi_values.iloc[-1] > rsi_values.iloc[:-1].min():
        return "Bullish"
    if high_prices.iloc[-1] > high_prices.iloc[:-1].max() and rsi_values.iloc[-1] < rsi_values.iloc[:-1].max():
        return "Bearish"
    return "None"

def check_cpr_breakout(current_candle, cpr_pivots, last_candle):
    """Checks for a bullish or bearish breakout from the CPR."""
    if not cpr_pivots: return "None"
    price = current_candle['close']; last_price = last_candle['close']
    tc = cpr_pivots['tc']; bc = cpr_pivots['bc']

    if price > tc and last_price > tc: return "Bullish"
    if price < bc and last_price < bc: return "Bearish"
    return "None"

def lex_algo_supply_demand(df):
    """PLACEHOLDER for the 'Lex Algo Supply & Demand' indicator."""
    return "None"

def _find_extrema(series: pd.Series, window: int = 5):
    """
    Find SIGNIFICANT local peaks and troughs.

    Uses STRICT inequality: a bar is a peak only if it is the UNIQUE maximum of
    its ±window neighbourhood (every other bar strictly lower), and a trough only
    if the unique minimum. The old version used `max() == value`, which on flat
    or quiet bars marked EVERY point as both a peak and a trough — so the
    divergence check ended up comparing spurious flat-region points instead of
    the real swing highs, silently missing on-screen divergences.
    """
    extrema = []
    n = len(series)
    if n < (2 * window + 1):
        return extrema
    vals = series.values
    for i in range(window, n - window):
        seg = vals[i - window:i + window + 1]
        c = vals[i]
        if c == seg.max() and int((seg < c).sum()) == len(seg) - 1:
            extrema.append((i, 'peak'))
        elif c == seg.min() and int((seg > c).sum()) == len(seg) - 1:
            extrema.append((i, 'trough'))
    return extrema

def check_momentum_divergence(price_series: pd.Series, oscillator_series: pd.Series, lookback: int = 45):
    """
    Checks for Class A Regular Divergence over the lookback period.
    Returns 'Bullish', 'Bearish', or 'None'.
    """
    if len(price_series) < lookback or len(oscillator_series) < lookback:
        return "None"
        
    price_slice = price_series.tail(lookback)
    osc_slice = oscillator_series.tail(lookback)

    price_extrema = _find_extrema(price_slice)
    osc_extrema = _find_extrema(osc_slice)

    price_peaks = [p for p in price_extrema if p[1] == 'peak']
    price_troughs = [p for p in price_extrema if p[1] == 'trough']
    osc_peaks = [p for p in osc_extrema if p[1] == 'peak']
    osc_troughs = [p for p in osc_extrema if p[1] == 'trough']

    # Bearish Divergence: Higher high in price, lower high in oscillator
    if len(price_peaks) >= 2 and len(osc_peaks) >= 2:
        last_price_peak_val = price_slice.iloc[price_peaks[-1][0]]
        prev_price_peak_val = price_slice.iloc[price_peaks[-2][0]]
        last_osc_peak_val = osc_slice.iloc[osc_peaks[-1][0]]
        prev_osc_peak_val = osc_slice.iloc[osc_peaks[-2][0]]

        if last_price_peak_val > prev_price_peak_val and last_osc_peak_val < prev_osc_peak_val:
            return "Bearish"

    # Bullish Divergence: Lower low in price, higher low in oscillator
    if len(price_troughs) >= 2 and len(osc_troughs) >= 2:
        last_price_trough_val = price_slice.iloc[price_troughs[-1][0]]
        prev_price_trough_val = price_slice.iloc[price_troughs[-2][0]]
        last_osc_trough_val = osc_slice.iloc[osc_troughs[-1][0]]
        prev_osc_trough_val = osc_slice.iloc[osc_troughs[-2][0]]
        
        if last_price_trough_val < prev_price_trough_val and last_osc_trough_val > prev_osc_trough_val:
            return "Bullish"

    return "None"

def check_candle_confirmation(df: pd.DataFrame, signal: str,
                              support: float = None, resistance: float = None,
                              proximity: float = 10.0):
    """
    Candle-quality read of the LAST bar, used as entry timing at structure levels.
    Playbook semantics (5-min chart):
      • Doji / indecision (tiny body)                  → WAIT       (0.0)
      • Strong directional candle agreeing with signal → CONFIRM    (1.0)
        (extra-strong when it CLOSES through the nearby level)
      • Long lower wick AT support  + BUY  signal      → CONFIRM    (1.0)  [bounce]
      • Long upper wick AT resistance + SELL signal    → CONFIRM    (1.0)  [rejection]
      • Strong candle AGAINST the signal               → CONTRA     (0.0)
      • Anything else                                  → NEUTRAL    (0.5)

    Returns (value, reason) where value ∈ {0.0, 0.5, 1.0}; returns (None, reason)
    when the bar can't be read (missing columns / zero range) so the caller can
    EXCLUDE the factor rather than penalise.
    """
    try:
        bar = df.iloc[-1]
        o, h, l, c = float(bar['open']), float(bar['high']), float(bar['low']), float(bar['close'])
    except Exception:
        return None, "bar unreadable (missing OHLC)"
    rng = h - l
    if rng <= 0:
        return None, "zero-range bar"

    body = abs(c - o)
    body_frac = body / rng
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    is_green = c > o

    # 1. Doji / indecision: tiny body relative to range → wait.
    if body_frac < 0.25:
        # ... unless the wick itself is the signal: a hammer at support / a
        # shooting star at resistance is a CONFIRMING rejection, not indecision.
        near_support = support is not None and abs(l - float(support)) <= proximity
        near_resistance = resistance is not None and abs(h - float(resistance)) <= proximity
        if signal == 'BUY' and near_support and lower_wick >= 0.5 * rng:
            return 1.0, f"hammer: long lower wick rejecting support {support:.0f}"
        if signal == 'SELL' and near_resistance and upper_wick >= 0.5 * rng:
            return 1.0, f"shooting star: upper wick rejecting resistance {resistance:.0f}"
        return 0.0, "doji/indecision bar — wait"

    # 2. Wick rejection at a structure level (body need not be tiny).
    if signal == 'BUY' and support is not None:
        if abs(l - float(support)) <= proximity and lower_wick >= 2.0 * body and lower_wick >= 0.4 * rng:
            return 1.0, f"long lower wick bounce off support {support:.0f}"
    if signal == 'SELL' and resistance is not None:
        if abs(h - float(resistance)) <= proximity and upper_wick >= 2.0 * body and upper_wick >= 0.4 * rng:
            return 1.0, f"long upper wick rejection at resistance {resistance:.0f}"

    # 3. Strong directional candle (body dominates the range).
    if body_frac >= 0.6:
        agrees = (signal == 'BUY' and is_green) or (signal == 'SELL' and not is_green)
        if agrees:
            # Closing THROUGH the nearby level = breakout confirmation.
            if signal == 'BUY' and resistance is not None and c > float(resistance) >= l:
                return 1.0, f"strong green close above resistance {resistance:.0f}"
            if signal == 'SELL' and support is not None and c < float(support) <= h:
                return 1.0, f"strong red close below support {support:.0f}"
            return 1.0, "strong directional candle agrees"
        return 0.0, "strong candle AGAINST the signal"

    # 4. Ordinary bar: neither confirming nor contradicting.
    return 0.5, "no decisive candle"


def is_trend_overextended(day_df: pd.DataFrame, lookback: int = 20, percent_move: float = 0.01, rsi_high: int = 70, rsi_low: int = 30):
    """Quantitatively defines an overextended trend."""
    if len(day_df) < lookback:
        return "None"
    price_slice = day_df['close'][-lookback:]
    max_price, min_price = price_slice.max(), price_slice.min()
    current_price = price_slice.iloc[-1]
    rsi = day_df['rsi'].iloc[-1]
    
    # Check for overextended uptrend
    if (current_price / min_price - 1) > percent_move and rsi > rsi_high:
        return "Uptrend"
    # Check for overextended downtrend
    if (max_price / current_price - 1) > percent_move and rsi < rsi_low:
        return "Downtrend"
        
    return "None"