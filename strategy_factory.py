import logging
import pandas as pd
import pandas_ta_classic as ta
import datetime
from indicators import (
    calculate_cpr, calculate_rsi, check_rsi_divergence, 
    check_cpr_breakout, calculate_ema, check_ema_crossover,
    check_momentum_divergence, is_trend_overextended
)

class BaseStrategy:
    """Base class for all trading strategies."""

    def __init__(self, kite, config):
        self.kite = kite
        self.config = config
        self.name = "Base"
        self.is_reversal_trade = False # Default behavior requires sentiment confirmation


    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        """
        Generates 'BUY', 'SELL', or 'HOLD' signals.
        The 'index' argument is now optional. If None, it defaults to the latest candle.
        """
        raise NotImplementedError

    def get_status_message(self, day_df, sentiment, **kwargs):
        """Returns a human-readable status message."""
        return "Awaiting signal: Generic strategy waiting for conditions."

    def _log_hold(self, reason: str):
        """
        Emit a HOLD-reason log when the operator has opted in via
        `trading_flags.log_hold_reasons: true`. Lets you read the logs
        and see exactly which strategy condition isn't being met instead
        of guessing why nothing is firing.
        """
        if (self.config.get("trading_flags") or {}).get("log_hold_reasons", False):
            logging.info(f"[{self.name}] HOLD: {reason}")

class Gemini_Default_Strategy(BaseStrategy):
    """The original Gemini strategy based on CPR, EMA, and RSI."""
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "Gemini_Default"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 1:
            self._log_hold("insufficient bars (index < 1)")
            return 'HOLD'
        if 'ema_50' not in day_df.columns:
            day_df['ema_50'] = calculate_ema(day_df['close'], 50)
        if 'rsi' not in day_df.columns:
            day_df['rsi'] = calculate_rsi(day_df['close'], 14)

        cpr_pivots = kwargs.get('cpr_pivots', {})
        current_candle = day_df.iloc[index]

        primary_signal_met = False
        confirmation_signals_met = 0

        cpr_breakout_signal = check_cpr_breakout(current_candle, cpr_pivots, day_df.iloc[index-1])
        if cpr_breakout_signal == sentiment:
            primary_signal_met = True

        if primary_signal_met:
            if sentiment == 'Bullish':
                if current_candle['close'] > current_candle['ema_50']: confirmation_signals_met += 1
                if current_candle['rsi'] > 55: confirmation_signals_met += 1
            elif sentiment == 'Bearish':
                if current_candle['close'] < current_candle['ema_50']: confirmation_signals_met += 1
                if current_candle['rsi'] < 45: confirmation_signals_met += 1

        logging.debug(f"[{self.name}] Check on {current_candle.name}: Primary Met={primary_signal_met}, Confirmations Met={confirmation_signals_met}")

        if primary_signal_met and confirmation_signals_met >= 1:
            logging.info(f"[{self.name}] Signal confirmed: Primary condition and {confirmation_signals_met} confirmation(s) met.")
            return 'BUY' if sentiment == 'Bullish' else 'SELL'

        if not primary_signal_met:
            self._log_hold(
                f"CPR breakout direction ({cpr_breakout_signal!r}) does not match "
                f"sentiment ({sentiment!r}). "
                f"close={current_candle.get('close', 0):.2f}, "
                f"tc={cpr_pivots.get('tc', 'N/A')}, bc={cpr_pivots.get('bc', 'N/A')}"
            )
        else:
            self._log_hold(
                f"CPR breakout matched ({sentiment}) but 0 confirmations met "
                f"(need >=1 from EMA50/RSI). "
                f"close={current_candle.get('close', 0):.2f} vs "
                f"ema_50={current_candle.get('ema_50', float('nan')):.2f}, "
                f"rsi={current_candle.get('rsi', float('nan')):.1f}"
            )
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        cpr = kwargs.get('cpr_pivots', {})
        if not cpr or 'tc' not in cpr or 'bc' not in cpr:
            return f"Awaiting signal for {self.name}: CPR pivots not yet calculated."
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Price to cross above CPR Top ({cpr['tc']:.2f}) and be confirmed by EMA(50) & RSI > 55."
        else:
            return f"Awaiting SELL signal: Price to cross below CPR Bottom ({cpr['bc']:.2f}) and be confirmed by EMA(50) & RSI < 45."

class Supertrend_MACD_Strategy(BaseStrategy):
    """A trend-following strategy based on Supertrend and MACD."""
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "Supertrend_MACD"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 1: return 'HOLD'
        
        if 'supertrend_direction' not in day_df.columns:
            supertrend = ta.supertrend(day_df['high'], day_df['low'], day_df['close'])
            if supertrend is not None and not supertrend.empty:
                day_df['supertrend_direction'] = supertrend.get('SUPERTd_7_3.0')
        if 'macd' not in day_df.columns:
            macd = ta.macd(day_df['close'])
            if macd is not None and not macd.empty:
                day_df[['macd', 'macd_signal']] = macd[['MACD_12_26_9', 'MACDs_12_26_9']]

        current = day_df.iloc[index]
        st_dir = current.get('supertrend_direction')
        macd = current.get('macd')
        macd_sig = current.get('macd_signal')
        is_bullish_signal = st_dir == 1 and (macd or 0) > (macd_sig or 0)
        is_bearish_signal = st_dir == -1 and (macd or 0) < (macd_sig or 0)

        if is_bullish_signal:
            logging.info(f"[{self.name}] BUY Signal condition met.")
            return 'BUY'
        if is_bearish_signal:
            logging.info(f"[{self.name}] SELL Signal condition met.")
            return 'SELL'

        self._log_hold(
            f"supertrend_dir={st_dir}, macd={macd}, macd_signal={macd_sig} "
            f"(need ST=+1 AND macd>signal for BUY, or ST=-1 AND macd<signal for SELL)"
        )
        return 'HOLD'
    
    def get_status_message(self, day_df, sentiment, **kwargs):
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Supertrend must be bullish AND the MACD line must cross above its signal line."
        else:
            return f"Awaiting SELL signal: Supertrend must be bearish AND the MACD line must cross below its signal line."

class VolatilityClusterStrategy(BaseStrategy):
    """A reversal strategy based on the concept of Volatility Clustering."""
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "Volatility_Cluster_Reversal"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 20: return 'HOLD'

        if 'atr' not in day_df.columns:
            day_df['atr'] = ta.atr(day_df['high'], day_df['low'], day_df['close'], length=14)
        if 'atr_ma' not in day_df.columns:
            day_df['atr_ma'] = day_df['atr'].rolling(window=20).mean()
            
        last_completed_candle = day_df.iloc[index - 1]

        if pd.isna(last_completed_candle['atr']) or pd.isna(last_completed_candle['atr_ma']): return 'HOLD'

        is_high_volatility = last_completed_candle['atr'] > last_completed_candle['atr_ma']
        avg_candle_size = day_df['atr'].iloc[index-1]
        last_candle_size = abs(last_completed_candle['open'] - last_completed_candle['close'])
        is_large_move = last_candle_size > (avg_candle_size * 1.5)

        if sentiment in ['Bullish', 'Very Bullish']:
            is_reversal_candle = last_completed_candle['close'] < last_completed_candle['open']
            if is_high_volatility and is_large_move and is_reversal_candle:
                logging.info(f"[{self.name}] Reversal BUY signal: High volatility detected after a large down move.")
                return 'BUY'
        elif sentiment in ['Bearish', 'Very Bearish']:
            is_reversal_candle = last_completed_candle['close'] > last_completed_candle['open']
            if is_high_volatility and is_large_move and is_reversal_candle:
                logging.info(f"[{self.name}] Reversal SELL signal: High volatility detected after a large up move.")
                return 'SELL'
            
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Looking for a large downward candle during high volatility to signal a potential reversal up."
        else:
            return f"Awaiting SELL signal: Looking for a large upward candle during high volatility to signal a potential reversal down."

class VSA_Strategy(BaseStrategy):
    """A strategy based on Volume Spread Analysis (VSA)."""
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "Volume_Spread_Analysis"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 20: return 'HOLD'
        
        if 'volume_ma' not in day_df.columns:
            day_df['volume_ma'] = day_df['volume'].rolling(window=20).mean()
        if 'spread' not in day_df.columns:
            day_df['spread'] = day_df['high'] - day_df['low']
        
        last_candle = day_df.iloc[index - 1]
        
        is_high_volume = last_candle.get('volume', 0) > (last_candle.get('volume_ma', 0) * 1.3)
        is_wide_spread = last_candle.get('spread', 0) > day_df['spread'].rolling(window=20).mean().iloc[index - 1]
        
        if sentiment in ['Bullish', 'Very Bullish']:
            is_down_bar = last_candle['close'] < last_candle['open']
            is_high_close = last_candle['close'] > (last_candle['low'] + last_candle['spread'] * 0.5)
            if is_down_bar and is_high_volume and is_wide_spread and is_high_close:
                logging.info(f"[{self.name}] Signal confirmed: Sign of Strength detected."); return 'BUY'
        
        if sentiment in ['Bearish', 'Very Bearish']:
            is_up_bar = last_candle['close'] > last_candle['open']
            is_low_close = last_candle['close'] < (last_candle['low'] + last_candle['spread'] * 0.5)
            if is_up_bar and is_high_volume and is_wide_spread and is_low_close:
                logging.info(f"[{self.name}] Signal confirmed: Sign of Weakness detected."); return 'SELL'
            
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Looking for a 'Sign of Strength' - a down-bar with high volume closing in its upper half."
        else:
            return f"Awaiting SELL signal: Looking for a 'Sign of Weakness' - an up-bar with high volume closing in its lower half."

class Momentum_VWAP_RSI_Strategy(BaseStrategy):
    def __init__(self, kite, config): super().__init__(kite, config); self.name = "Momentum_VWAP_RSI"
    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 1:
            self._log_hold("insufficient bars (index < 1)")
            return 'HOLD'

        current = day_df.iloc[index]
        close = current.get('close', float('nan'))
        vwap = current.get('vwap', float('nan'))
        rsi = current.get('rsi', float('nan'))

        if sentiment in ['Bullish', 'Very Bullish'] and close > vwap and rsi > 55:
            return 'BUY'
        if sentiment in ['Bearish', 'Very Bearish'] and close < vwap and rsi < 45:
            return 'SELL'

        if sentiment in ['Bullish', 'Very Bullish']:
            self._log_hold(
                f"need close>vwap AND rsi>55. got close={close:.2f}, vwap={vwap:.2f} "
                f"({'above' if close > vwap else 'below'}), rsi={rsi:.1f}"
            )
        else:
            self._log_hold(
                f"need close<vwap AND rsi<45. got close={close:.2f}, vwap={vwap:.2f} "
                f"({'above' if close > vwap else 'below'}), rsi={rsi:.1f}"
            )
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        vwap = day_df.iloc[-1].get('vwap', 0)
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Price needs to be above VWAP ({vwap:.2f}) with RSI > 55."
        else:
            return f"Awaiting SELL signal: Price needs to be below VWAP ({vwap:.2f}) with RSI < 45."

class Breakout_Prev_Day_HL_Strategy(BaseStrategy):
    def __init__(self, kite, config): super().__init__(kite, config); self.name = "Breakout_Prev_Day_HL"
    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 1:
            self._log_hold("insufficient bars (index < 1)")
            return 'HOLD'

        cpr = kwargs.get('cpr_pivots', {})
        pdh, pdl = cpr.get('prev_high'), cpr.get('prev_low')
        if not pdh or not pdl:
            self._log_hold("prev_day high/low pivots missing")
            return 'HOLD'
        current, last = day_df.iloc[index], day_df.iloc[index - 1]
        close = current.get('close', float('nan'))
        vol = current.get('volume', 0)
        vol_ma = current.get('volume_ma', 0)
        vol_ok = vol > (vol_ma * 1.2) if vol_ma else False

        if sentiment in ['Bullish', 'Very Bullish'] and last['close'] < pdh and close > pdh and vol_ok:
            return 'BUY'
        if sentiment in ['Bearish', 'Very Bearish'] and last['close'] > pdl and close < pdl and vol_ok:
            return 'SELL'

        if sentiment in ['Bullish', 'Very Bullish']:
            self._log_hold(
                f"need break above PDH({pdh:.2f}) with vol>1.2*MA. "
                f"close={close:.2f}, prev_close={last['close']:.2f}, "
                f"vol={vol:.0f}, vol_ma={vol_ma:.0f}, vol_ok={vol_ok}"
            )
        else:
            self._log_hold(
                f"need break below PDL({pdl:.2f}) with vol>1.2*MA. "
                f"close={close:.2f}, prev_close={last['close']:.2f}, "
                f"vol={vol:.0f}, vol_ma={vol_ma:.0f}, vol_ok={vol_ok}"
            )
        return 'HOLD'
    
    def get_status_message(self, day_df, sentiment, **kwargs):
        cpr = kwargs.get('cpr_pivots', {})
        pdh, pdl = cpr.get('prev_high'), cpr.get('prev_low')
        if not pdh or not pdl:
            return f"Awaiting signal for {self.name}: Previous day's high/low not available."
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Price needs to break above previous day's high ({pdh:.2f}) on high volume."
        else:
            return f"Awaiting SELL signal: Price needs to break below previous day's low ({pdl:.2f}) on high volume."

class Opening_Range_Breakout_Strategy(BaseStrategy):
    def __init__(self, kite, config):
        super().__init__(kite, config); self.name = "Opening_Range_Breakout"
        self.orb_high = None; self.orb_low = None; self.orb_period_set = False
    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        orb_minutes = self.config['trading_flags'].get('orb_minutes', 30)
        
        current_time = day_df.index[index].time()
        market_open_time = datetime.time(9, 15)
        orb_end_time = (datetime.datetime.combine(datetime.date.today(), market_open_time) + datetime.timedelta(minutes=orb_minutes)).time()
        
        if not self.orb_period_set and current_time >= orb_end_time:
            orb_df = day_df.between_time(market_open_time.strftime("%H:%M"), orb_end_time.strftime("%H:%M"))
            if not orb_df.empty:
                self.orb_high, self.orb_low = orb_df['high'].max(), orb_df['low'].min()
                self.orb_period_set = True
                logging.info(f"[{self.name}] ORB Set: High={self.orb_high:.2f}, Low={self.orb_low:.2f}, Range={(self.orb_high - self.orb_low):.2f}")
        
        if not self.orb_period_set: return 'HOLD'
        
        if (self.orb_high - self.orb_low) < 10:
            logging.debug(f"[{self.name}] ORB range is too narrow ({self.orb_high - self.orb_low:.2f} points). No trades will be taken.")
            return 'HOLD'

        current, last = day_df.iloc[index], day_df.iloc[index - 1]
        if 'volume_ma' not in day_df.columns: day_df['volume_ma'] = day_df['volume'].rolling(window=20).mean()
        
        if sentiment in ['Bullish', 'Very Bullish'] and last['close'] < self.orb_high and current['close'] > self.orb_high and current['volume'] > (current.get('volume_ma', 0) * 1.5):
            logging.info(f"[{self.name}] BUY Signal on ORB High breakout.")
            return 'BUY'
        if sentiment in ['Bearish', 'Very Bearish'] and last['close'] > self.orb_low and current['close'] < self.orb_low and current['volume'] > (current.get('volume_ma', 0) * 1.5):
            logging.info(f"[{self.name}] SELL Signal on ORB Low breakdown.")
            return 'SELL'
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if not self.orb_period_set:
            return f"Awaiting signal for {self.name}: Waiting for the opening range to be established."
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Price needs to break above the ORB high of {self.orb_high:.2f} on high volume."
        else:
            return f"Awaiting SELL signal: Price needs to break below the ORB low of {self.orb_low:.2f} on high volume."

class Bollinger_Band_Squeeze_Strategy(BaseStrategy):
    def __init__(self, kite, config): super().__init__(kite, config); self.name = "BB_Squeeze_Breakout"
    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 1:
            self._log_hold("insufficient bars (index < 1)")
            return 'HOLD'

        current, last = day_df.iloc[index], day_df.iloc[index - 1]
        bw, bw_ma = current.get('bb_bandwidth'), current.get('bb_bandwidth_ma')
        in_squeeze = bw is not None and bw_ma is not None and bw < bw_ma

        if in_squeeze:
            close = current.get('close', float('nan'))
            upper = current.get('bb_upper', float('nan'))
            lower = current.get('bb_lower', float('nan'))
            last_close = last.get('close', float('nan'))
            last_upper = last.get('bb_upper', float('nan'))
            last_lower = last.get('bb_lower', float('nan'))
            if sentiment in ['Bullish', 'Very Bullish'] and last_close < last_upper and close > upper:
                return 'BUY'
            if sentiment in ['Bearish', 'Very Bearish'] and last_close > last_lower and close < lower:
                return 'SELL'
            self._log_hold(
                f"in squeeze (bw={bw:.3f}<bw_ma={bw_ma:.3f}) but no breakout. "
                f"close={close:.2f}, upper={upper:.2f}, lower={lower:.2f}"
            )
        else:
            self._log_hold(
                f"not in squeeze: bb_bandwidth={bw} >= bb_bandwidth_ma={bw_ma} "
                f"(need bandwidth below its MA)"
            )
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        current = day_df.iloc[-1]
        if current['bb_bandwidth'] > current['bb_bandwidth_ma']:
            return f"Awaiting signal for {self.name}: Waiting for Bollinger Bands to tighten into a squeeze."
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: In a BB Squeeze. Waiting for price to break above the upper band ({current['bb_upper']:.2f})."
        else:
            return f"Awaiting SELL signal: In a BB Squeeze. Waiting for price to break below the lower band ({current['bb_lower']:.2f})."

class MA_Crossover_Strategy(BaseStrategy):
    def __init__(self, kite, config): super().__init__(kite, config); self.name = "MA_Crossover"
    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        if index < 1: return 'HOLD'

        current, last = day_df.iloc[index], day_df.iloc[index - 1]
        if sentiment in ['Bullish', 'Very Bullish'] and last['ema_9'] <= last['ema_21'] and current['ema_9'] > current['ema_21']: return 'BUY'
        if sentiment in ['Bearish', 'Very Bearish'] and last['ema_9'] >= last['ema_21'] and current['ema_9'] < current['ema_21']: return 'SELL'
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Waiting for the 9-period EMA to cross above the 21-period EMA."
        else:
            return f"Awaiting SELL signal: Waiting for the 9-period EMA to cross below the 21-period EMA."

class RSI_Divergence_Strategy(BaseStrategy):
    def __init__(self, kite, config): super().__init__(kite, config); self.name = "RSI_Divergence"
    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None: index = len(day_df) - 1
        
        divergence = check_rsi_divergence(day_df.iloc[:index + 1], day_df['rsi'].iloc[:index + 1])
        if sentiment in ['Bullish', 'Very Bullish'] and divergence == 'Bullish': return 'BUY'
        if sentiment in ['Bearish', 'Very Bearish'] and divergence == 'Bearish': return 'SELL'
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Waiting for price to make a new low while RSI makes a higher low (Bullish Divergence)."
        else:
            return f"Awaiting SELL signal: Waiting for price to make a new high while RSI makes a lower high (Bearish Divergence)."

class EMACrossRSIStrategy(BaseStrategy):
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "EMA_Cross_RSI"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        """
        Generates a signal if the EMAs are in a trending state and a crossover
        has occurred within a recent lookback period.
        """
        if index is None:
            index = len(day_df) - 1
        
        # New configurable lookback period. Default to 5 candles if not set.
        lookback_period = self.config['trading_flags'].get('ema_cross_lookback', 5)

        if index < lookback_period + 1: # Ensure we have enough data for the lookback
            return 'HOLD'

        # Ensure indicators are present
        if 'ema_9' not in day_df.columns: day_df['ema_9'] = calculate_ema(day_df['close'], 9)
        if 'ema_15' not in day_df.columns: day_df['ema_15'] = calculate_ema(day_df['close'], 15)
        if 'rsi' not in day_df.columns: day_df['rsi'] = calculate_rsi(day_df['close'], 14)
        
        current_candle = day_df.iloc[index]

        # --- MODIFIED BULLISH (BUY) SIGNAL LOGIC ---
        # 1. Check current state: 9-EMA is above 15-EMA now.
        is_trending_up = current_candle['ema_9'] > current_candle['ema_15']
        # 2. Check confirmation conditions: RSI and price are favorable now.
        is_confirmed_up = current_candle['rsi'] > 50 and current_candle['close'] > current_candle['ema_9']
        
        if is_trending_up and is_confirmed_up:
            # 3. Verify a "Golden Cross" happened recently
            recent_golden_cross = False
            for i in range(index - lookback_period, index + 1):
                prev_candle = day_df.iloc[i - 1]
                signal_candle = day_df.iloc[i]
                if prev_candle['ema_9'] < prev_candle['ema_15'] and signal_candle['ema_9'] > signal_candle['ema_15']:
                    recent_golden_cross = True
                    break  # Found the recent cross, no need to look further
            
            if recent_golden_cross:
                logging.info(f"[{self.name}] BUY Signal: 9/15 EMA in bullish state post-crossover with RSI > 50.")
                return 'BUY'

        # --- MODIFIED BEARISH (SELL) SIGNAL LOGIC ---
        # 1. Check current state: 9-EMA is below 15-EMA now.
        is_trending_down = current_candle['ema_9'] < current_candle['ema_15']
        # 2. Check confirmation conditions: RSI and price are favorable now.
        is_confirmed_down = current_candle['rsi'] < 50 and current_candle['close'] < current_candle['ema_9']

        if is_trending_down and is_confirmed_down:
            # 3. Verify a "Death Cross" happened recently
            recent_death_cross = False
            for i in range(index - lookback_period, index + 1):
                prev_candle = day_df.iloc[i - 1]
                signal_candle = day_df.iloc[i]
                if prev_candle['ema_9'] > prev_candle['ema_15'] and signal_candle['ema_9'] < signal_candle['ema_15']:
                    recent_death_cross = True
                    break
            
            if recent_death_cross:
                logging.info(f"[{self.name}] SELL Signal: 9/15 EMA in bearish state post-crossover with RSI < 50.")
                return 'SELL'

        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if sentiment in ['Bullish', 'Very Bullish']:
            return f"Awaiting BUY signal: Waiting for 9-EMA to cross above 15-EMA, with confirmation from RSI > 50."
        else:
            return f"Awaiting SELL signal: Waiting for 9-EMA to cross below 15-EMA, with confirmation from RSI < 50."


class Reversal_Detector_Strategy(BaseStrategy):
    """
    A robust strategy that trades reversals based on a confluence of signals:
    1. Pre-Condition: An overextended trend.
    2. Primary Signal: RSI momentum divergence.
    3. Confirmation: A break of price structure (close over/under a fast EMA).
    """
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "Reversal_Detector"
        self.is_reversal_trade = True # This flag bypasses the daily sentiment check

    def _is_trend_overextended(self, day_df, lookback=20):
        """Quantitatively defines an overextended trend."""
        price_slice = day_df['close'][-lookback:]
        max_price, min_price = price_slice.max(), price_slice.min()
        current_price = price_slice.iloc[-1]
        rsi = day_df['rsi'].iloc[-1]
        
        # Check for overextended uptrend 
        if (current_price / min_price - 1) > 0.015 and rsi > 70:
            return "Uptrend"
        # Check for overextended downtrend 
        if (max_price / current_price - 1) > 0.015 and rsi < 30:
            return "Downtrend"
            
        return "None"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        trend_status = is_trend_overextended(day_df)
        if trend_status == "None":
            return 'HOLD'

        rsi_divergence = check_momentum_divergence(day_df['close'], day_df['rsi'])
        
        current_candle = day_df.iloc[-1]

        # Look for a Bearish Reversal signal
        if trend_status == "Uptrend" and rsi_divergence == "Bearish":
            if current_candle['close'] < current_candle['ema_9']:
                logging.info(f"[{self.name}] Bearish Reversal Signal: Overextended uptrend with RSI divergence confirmed by close below 9-EMA.")
                return 'SELL'

        # Look for a Bullish Reversal signal
        if trend_status == "Downtrend" and rsi_divergence == "Bullish":
            if current_candle['close'] > current_candle['ema_9']:
                logging.info(f"[{self.name}] Bullish Reversal Signal: Overextended downtrend with RSI divergence confirmed by close above 9-EMA.")
                return 'BUY'
        
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        trend_status = is_trend_overextended(day_df)
        if trend_status == "Uptrend":
            return f"Awaiting SELL signal: Overextended uptrend detected. Looking for bearish RSI divergence and a confirmation break below 9-EMA."
        if trend_status == "Downtrend":
            return f"Awaiting BUY signal: Overextended downtrend detected. Looking for bullish RSI divergence and a confirmation break above 9-EMA."
        return f"Awaiting signal for {self.name}: Waiting for a sustained, overextended trend to form."

class VWAP_Reversion_Strategy(BaseStrategy):
    """
    Intraday VWAP-pullback play, designed to fire MULTIPLE times per day in a
    trending session.

    Bullish day:
      - BUY when the prior bar closed at or below VWAP AND the current bar
        closes above VWAP (a "reclaim").
      - Momentum filter: RSI > 45 (not deep oversold).

    Bearish day: mirror image (VWAP loss + RSI < 55).

    Avoids buying breakouts at the high — instead buys the dip back to the
    institutional anchor and reclaim. Pairs well with the trailing-SL setup.
    """
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "VWAP_Reversion"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None:
            index = len(day_df) - 1
        if index < 2:
            self._log_hold("insufficient bars (index < 2)")
            return 'HOLD'
        if 'vwap' not in day_df.columns or 'rsi' not in day_df.columns:
            self._log_hold("vwap or rsi column missing from bars")
            return 'HOLD'

        current = day_df.iloc[index]
        prev = day_df.iloc[index - 1]

        if pd.isna(current.get('vwap')) or pd.isna(prev.get('vwap')):
            self._log_hold("vwap is NaN on current or prev bar")
            return 'HOLD'
        if pd.isna(current.get('rsi')):
            self._log_hold("rsi is NaN on current bar")
            return 'HOLD'

        cur_close, cur_vwap, cur_rsi = current['close'], current['vwap'], current['rsi']
        prev_close, prev_vwap = prev['close'], prev['vwap']

        # Bullish reclaim: prior bar at/under VWAP, current bar closes above it.
        if sentiment in ['Bullish', 'Very Bullish']:
            reclaimed = (prev_close <= prev_vwap) and (cur_close > cur_vwap)
            momentum_ok = cur_rsi > 45
            if reclaimed and momentum_ok:
                logging.info(f"[{self.name}] BUY: VWAP reclaim with RSI {cur_rsi:.1f} > 45.")
                return 'BUY'
            self._log_hold(
                f"need VWAP-reclaim (prev close<=VWAP AND curr close>VWAP) AND RSI>45. "
                f"prev_close={prev_close:.2f} vs prev_vwap={prev_vwap:.2f} "
                f"({'<=' if prev_close <= prev_vwap else '>'}), "
                f"curr_close={cur_close:.2f} vs curr_vwap={cur_vwap:.2f} "
                f"({'>' if cur_close > cur_vwap else '<='}), "
                f"rsi={cur_rsi:.1f} (momentum_ok={momentum_ok})"
            )
            return 'HOLD'

        # Bearish loss: prior bar at/above VWAP, current bar closes below it.
        if sentiment in ['Bearish', 'Very Bearish']:
            lost = (prev_close >= prev_vwap) and (cur_close < cur_vwap)
            momentum_ok = cur_rsi < 55
            if lost and momentum_ok:
                logging.info(f"[{self.name}] SELL: VWAP loss with RSI {cur_rsi:.1f} < 55.")
                return 'SELL'
            self._log_hold(
                f"need VWAP-loss (prev close>=VWAP AND curr close<VWAP) AND RSI<55. "
                f"prev_close={prev_close:.2f} vs prev_vwap={prev_vwap:.2f}, "
                f"curr_close={cur_close:.2f} vs curr_vwap={cur_vwap:.2f}, "
                f"rsi={cur_rsi:.1f} (momentum_ok={momentum_ok})"
            )
            return 'HOLD'

        self._log_hold(f"sentiment {sentiment!r} is Neutral — no directional setup")
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        vwap = day_df.iloc[-1].get('vwap', 0) if len(day_df) else 0
        if sentiment in ['Bullish', 'Very Bullish']:
            return (f"Awaiting BUY: price to pull back to/under VWAP ({vwap:.2f}) "
                    f"and reclaim it on the next bar, RSI > 45.")
        return (f"Awaiting SELL: price to lift above VWAP ({vwap:.2f}) and lose "
                f"it on the next bar, RSI < 55.")


class NR7_Compression_Breakout_Strategy(BaseStrategy):
    """
    Compression-then-expansion play. Fires when:

      1. The narrowest of the last 7 completed bars (NR7) appeared within the
         last 3 bars (fresh compression — not stale).
      2. The current bar closes ABOVE the NR7 bar's high (BUY) or BELOW its
         low (SELL).
      3. The breakout is confirmed by volume > 1.2 × 20-bar volume MA.
      4. Sentiment direction matches.

    Compression-day breakouts have higher expectancy than chop-day breakouts.
    """
    def __init__(self, kite, config):
        super().__init__(kite, config)
        self.name = "NR7_Compression"

    def generate_signals(self, day_df, sentiment, index=None, **kwargs):
        if index is None:
            index = len(day_df) - 1
        if index < 8:
            self._log_hold("insufficient bars (need >= 8)")
            return 'HOLD'
        if 'volume_ma' not in day_df.columns:
            self._log_hold("volume_ma column missing")
            return 'HOLD'

        window = day_df.iloc[index - 7:index]
        if window.empty or len(window) < 7:
            self._log_hold("window too short for NR7 lookup")
            return 'HOLD'
        ranges = window['high'] - window['low']
        if ranges.isna().any():
            self._log_hold("range column has NaN in lookback window")
            return 'HOLD'
        nr7_idx_in_window = int(ranges.values.argmin())
        bars_since_nr7 = len(window) - 1 - nr7_idx_in_window
        if bars_since_nr7 > 3:
            self._log_hold(
                f"NR7 too stale: narrowest bar was {bars_since_nr7} bars ago "
                f"(need <= 3 to count as 'fresh' compression)"
            )
            return 'HOLD'

        nr7_bar = window.iloc[nr7_idx_in_window]
        current = day_df.iloc[index]
        cur_close = current.get('close', float('nan'))
        cur_vol = current.get('volume', 0)
        cur_vol_ma = current.get('volume_ma', 0)
        if pd.isna(cur_vol_ma) or cur_vol_ma <= 0:
            self._log_hold("volume_ma is NaN/zero")
            return 'HOLD'
        volume_confirm = cur_vol > (cur_vol_ma * 1.2)
        if not volume_confirm:
            self._log_hold(
                f"NR7 fresh ({bars_since_nr7} bars ago) but volume not confirming: "
                f"vol={cur_vol:.0f} <= 1.2*MA={1.2 * cur_vol_ma:.0f}"
            )
            return 'HOLD'

        nr7_high, nr7_low = nr7_bar['high'], nr7_bar['low']
        if sentiment in ['Bullish', 'Very Bullish'] and cur_close > nr7_high:
            logging.info(
                f"[{self.name}] BUY: close {cur_close:.2f} > NR7 high "
                f"{nr7_high:.2f} on volume {cur_vol:.0f} vs MA {cur_vol_ma:.0f}."
            )
            return 'BUY'
        if sentiment in ['Bearish', 'Very Bearish'] and cur_close < nr7_low:
            logging.info(
                f"[{self.name}] SELL: close {cur_close:.2f} < NR7 low "
                f"{nr7_low:.2f} on volume {cur_vol:.0f} vs MA {cur_vol_ma:.0f}."
            )
            return 'SELL'

        if sentiment in ['Bullish', 'Very Bullish']:
            self._log_hold(
                f"NR7 fresh + volume OK, but close {cur_close:.2f} <= NR7 high "
                f"{nr7_high:.2f} (need breakout above the NR7 bar's high)"
            )
        elif sentiment in ['Bearish', 'Very Bearish']:
            self._log_hold(
                f"NR7 fresh + volume OK, but close {cur_close:.2f} >= NR7 low "
                f"{nr7_low:.2f} (need breakdown below the NR7 bar's low)"
            )
        else:
            self._log_hold(f"sentiment {sentiment!r} is Neutral — no directional check")
        return 'HOLD'

    def get_status_message(self, day_df, sentiment, **kwargs):
        if len(day_df) < 8:
            return f"Awaiting signal for {self.name}: need at least 8 bars of history."
        # Locate the NR7 bar in the most recent 7-bar window for the status line
        window = day_df.iloc[-8:-1]
        ranges = window['high'] - window['low']
        if ranges.empty or ranges.isna().any():
            return f"Awaiting signal for {self.name}: data incomplete."
        nr7_bar = window.iloc[int(ranges.values.argmin())]
        if sentiment in ['Bullish', 'Very Bullish']:
            return (f"Awaiting BUY: price to close above NR7-bar high "
                    f"{nr7_bar['high']:.2f} on volume > 1.2x MA.")
        return (f"Awaiting SELL: price to close below NR7-bar low "
                f"{nr7_bar['low']:.2f} on volume > 1.2x MA.")


def get_strategy(name, kite, config):
    """Factory function to get a strategy instance by name."""
    strategies = {
        "Gemini_Default": Gemini_Default_Strategy,
        "Supertrend_MACD": Supertrend_MACD_Strategy,
        "Volatility_Cluster_Reversal": VolatilityClusterStrategy,
        "Volume_Spread_Analysis": VSA_Strategy,
        "Momentum_VWAP_RSI": Momentum_VWAP_RSI_Strategy,
        "Breakout_Prev_Day_HL": Breakout_Prev_Day_HL_Strategy,
        "Opening_Range_Breakout": Opening_Range_Breakout_Strategy,
        "BB_Squeeze_Breakout": Bollinger_Band_Squeeze_Strategy,
        "MA_Crossover": MA_Crossover_Strategy,
        "RSI_Divergence": RSI_Divergence_Strategy,
        "EMA_Cross_RSI": EMACrossRSIStrategy,
        "Reversal_Detector": Reversal_Detector_Strategy,
        "VWAP_Reversion": VWAP_Reversion_Strategy,
        "NR7_Compression": NR7_Compression_Breakout_Strategy,
    }
    strategy_class = strategies.get(name)
    if not strategy_class: raise ValueError(f"Strategy '{name}' not found.")
    return strategy_class(kite, config)