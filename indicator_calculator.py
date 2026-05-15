import logging
import pandas_ta_classic as ta
import pandas as pd


# Indicator columns that must be valid on the latest bar for strategies to
# trade. After all rolling/EMA computations are done, these columns are
# forward-filled so a transient NaN in the middle of the series (e.g., a Kite
# bar with bad data) doesn't corrupt the latest-bar value the bot acts on.
# OHLCV are deliberately NOT in this list — never overwrite price/volume.
_INDICATOR_COLS_TO_FFILL = (
    "rsi", "ema_9", "ema_15", "ema_20", "ema_21", "ema_50",
    "supertrend_direction", "macd", "macd_signal",
    "atr", "atr_ma", "spread", "volume_ma",
    "vwap",
    "bb_upper", "bb_lower", "bb_mid", "bb_bandwidth", "bb_bandwidth_ma",
    "psar_long", "psar_short",
)


def _safe_rolling_mean(series: pd.Series, window: int, min_periods: int = 5) -> pd.Series:
    """
    Rolling mean that survives sparse data. `min_periods` is the minimum
    number of non-NaN values required to produce a result — without this,
    pandas needs the FULL window populated, so the first (window-1) bars
    of any session return NaN, breaking strategies that touch this MA.

    With min_periods=5 the MA stabilises from bar 5 onwards, slightly noisier
    for the first 15 bars but never NaN on the latest bar of a real dataset.
    """
    return series.rolling(window=window, min_periods=min(min_periods, window)).mean()


def _compute_vwap_daily(df: pd.DataFrame) -> pd.Series:
    """
    Daily-anchored VWAP, computed manually without pandas_ta's `ta.vwap()` —
    which silently emits NaN when the DatetimeIndex is timezone-aware and was
    causing every VWAP-based strategy to return HOLD on every bar.

    VWAP for bar i within a day:
        VWAP_i = sum( typical_price_k * volume_k ) / sum( volume_k )   for k=0..i

    where typical_price = (high + low + close) / 3, and the sums reset at each
    calendar-date boundary.

    Returns a Series aligned with df.index. The first bar of each day equals
    its own typical price (well-defined; no NaN). If a row has volume=0, the
    cumulative denominator carries forward — no division by zero.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64", index=df.index if df is not None else None)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_x_vol = typical_price * df["volume"]

    # Derive a daily-grouping key from the index (or 'date' column).
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        # Strip timezone so date extraction is consistent across pandas versions.
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        day_key = idx.normalize()  # truncates to midnight of the same day
    else:
        day_key = pd.to_datetime(df.get("date", df.index)).dt.normalize()

    # Cumulative-within-day numerator / denominator.
    cum_tpv = tp_x_vol.groupby(day_key).cumsum()
    cum_vol = df["volume"].groupby(day_key).cumsum()

    # Avoid div-by-zero on a hypothetical zero-volume bar; replace with NaN.
    cum_vol_safe = cum_vol.where(cum_vol > 0)
    vwap = cum_tpv / cum_vol_safe
    vwap.index = df.index  # ensure alignment (groupby may reset the index name)
    return vwap


def calculate_all_indicators(df: pd.DataFrame, config: dict):
    """
    Calculates and attaches all required technical indicators to the dataframe.
    This is called once per day to avoid redundant calculations.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

    # Gemini Default & RSI Divergence Strategy Indicators
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ema_9'] = ta.ema(df['close'], length=9)
    df['ema_15'] = ta.ema(df['close'], length=15)
    df['ema_20'] = ta.ema(df['close'], length=20)
    df['ema_21'] = ta.ema(df['close'], length=21)
    df['ema_50'] = ta.ema(df['close'], length=50)

    # Supertrend & MACD Strategy Indicators
    supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
    if supertrend is not None and not supertrend.empty:
        df['supertrend_direction'] = supertrend['SUPERTd_10_3.0']
        
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']

    # Volatility & VSA Strategy Indicators
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_ma'] = _safe_rolling_mean(df['atr'], window=20)
    df['spread'] = df['high'] - df['low']
    df['volume_ma'] = _safe_rolling_mean(df['volume'], window=20)

    # Daily-anchored VWAP — computed manually to avoid pandas_ta's NaN-on-TZ
    # quirk. See _compute_vwap_daily docstring for the math.
    df['vwap'] = _compute_vwap_daily(df)

    # Bollinger Band Squeeze Strategy Indicators
    bbands = ta.bbands(df['close'], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_mid'] = bbands['BBM_20_2.0']
        df['bb_bandwidth'] = bbands['BBB_20_2.0']
        df['bb_bandwidth_ma'] = _safe_rolling_mean(df['bb_bandwidth'], window=20)

    # PSAR for indicator-based exits (long_psar = trailing stop for long positions).
    psar = ta.psar(df['high'], df['low'], df['close'], af=0.02, max_af=0.2)
    if psar is not None and not psar.empty:
        psar_long_col = next((c for c in psar.columns if c.startswith('PSARl_')), None)
        psar_short_col = next((c for c in psar.columns if c.startswith('PSARs_')), None)
        if psar_long_col:
            df['psar_long'] = psar[psar_long_col]
        if psar_short_col:
            df['psar_short'] = psar[psar_short_col]

    # Forward-fill indicator columns so a single bad bar in the middle of the
    # rolling window doesn't corrupt the latest-bar value strategies act on.
    # OHLCV is deliberately NOT touched — never overwrite price/volume.
    cols_to_fill = [c for c in _INDICATOR_COLS_TO_FFILL if c in df.columns]
    if cols_to_fill:
        df[cols_to_fill] = df[cols_to_fill].ffill()

    # Diagnostic: warn (once per call) if the latest bar still has any NaN in
    # the critical indicator set after the forward-fill. If this fires, the
    # dataset is too small or genuinely malformed — strategies will skip.
    if not df.empty:
        last = df.iloc[-1]
        missing = [c for c in cols_to_fill if pd.isna(last.get(c))]
        if missing:
            logging.warning(
                f"calculate_all_indicators: latest bar still has NaN for "
                f"{missing} after ffill (df has {len(df)} bars). Strategies "
                f"depending on these columns will skip."
            )

    return df
