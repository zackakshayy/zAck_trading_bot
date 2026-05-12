import logging
import os
import re
import select
import sys
import yaml
import time
import datetime
import calendar
import pandas as pd
import asyncio
from kiteconnect import KiteConnect, exceptions
from agents import OrderExecutionAgent, PositionManagementAgent
from sentiment_agent import SentimentAgent
from langgraph_agent import LangGraphAgent
from strategy_factory import get_strategy
from backtester import run_backtest
from reporting import send_daily_report, initialize_trade_log, log_trade, send_monthly_report
from indicators import calculate_cpr, is_trend_overextended, check_momentum_divergence
from indicator_calculator import calculate_all_indicators
from market_context import MarketConditionIdentifier
from rag_service import RAGService
from infra import (
    is_nse_holiday,
    load_daily_pnl,
    safe_ltp,
    save_daily_pnl,
)
import multiprocessing
import warnings

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed. Falling back to OS environment only.")

# Suppress the repeated UserWarning from pandas_ta for cleaner logs
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Converting to PeriodArray/Index representation will drop timezone information.*"
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _substitute_env(value):
    """Recursively replace ${VAR} placeholders with values from os.environ."""
    if isinstance(value, str):
        def repl(match):
            return os.environ.get(match.group(1), match.group(0))
        return _ENV_VAR_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def load_config():
    """Loads config.yaml and substitutes ${VAR} placeholders from the environment."""
    with open('config.yaml', 'r') as file:
        raw = yaml.safe_load(file)
    return _substitute_env(raw)


def persist_access_token(token: str, env_path: str = '.env'):
    """Persists the daily Zerodha access token to .env (not config.yaml)."""
    lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith('ZERODHA_ACCESS_TOKEN='):
            lines[i] = f'ZERODHA_ACCESS_TOKEN={token}\n'
            found = True
            break
    if not found:
        lines.append(f'ZERODHA_ACCESS_TOKEN={token}\n')
    with open(env_path, 'w') as f:
        f.writelines(lines)
    os.environ['ZERODHA_ACCESS_TOKEN'] = token

class TradingBotOrchestrator:
    """
    The central orchestrator for the trading bot. Manages state, coordinates agents,
    and runs the main trading loop.
    """
    def __init__(self, config):
        self.config = config
        self.kite = KiteConnect(api_key=config['zerodha']['api_key'], timeout=120, debug=True)
        self.active_strategy_name = "None"
        self.active_strategy = None

        # Initialize core services
        self.rag_service = RAGService(config)
        self.langgraph_agent = LangGraphAgent(config, self.rag_service)
        self.sentiment_agent = SentimentAgent(config)

        # Defer initialization of session-dependent agents until after authentication
        self.market_condition_identifier = None
        self.order_agent = None
        self.position_agent = None

        # State variables
        self.day_sentiment = ""
        self.trades_today_count = 0
        self.no_trade_reason = None
        self.bot_state = "STARTING"
        self.last_processed_timestamp = None
        self.awaiting_signal_since = None
        # Realized P&L tracking for daily-loss circuit breaker — persisted per date.
        self._today_str = datetime.date.today().isoformat()
        self.realized_pnl_today = load_daily_pnl(self._today_str)
        if self.realized_pnl_today != 0.0:
            logging.info(
                f"Resuming with persisted realized P&L for {self._today_str}: "
                f"{self.realized_pnl_today:,.2f}"
            )
        self.starting_capital = None
        # Effective entry-start time for today (None until _compute_effective_entry_start runs).
        self.effective_entry_start_time = None
        # Cache for the underlying intraday bar data (refreshed only when a new bar closes).
        self._bars_cache = None
        self._bars_cached_at_bar = None
        # Track whether the bot should fail-stop on the next iteration (e.g. token expiry).
        self._abort = False
        # Sentiment + NL-prompt cache. Captured ONCE on first setup; reused on
        # every reassessment unless the market has materially shifted (drift in
        # spot / VIX / automated news-sentiment regime).
        self._cached_sentiment = None
        self._cached_nl_prompt = None
        self._sentiment_baseline_spot = None
        self._sentiment_baseline_vix = None
        self._sentiment_baseline_auto = None
        # Daily-report idempotency flag — set the first time the report is sent
        # (whether by normal market-close or by an early shutdown).
        self._report_sent = False

    def authenticate(self, request_token_override=None):
        """
        Handles user authentication. It can accept a token for API-driven flows
        or prompt the user in console mode.
        """
        logging.info("Attempting fresh authentication...")
        if not request_token_override:
            logging.info(f"Login URL: {self.kite.login_url()}")
            request_token = input("Enter request_token: ")
        else:
            request_token = request_token_override
            
        try:
            data = self.kite.generate_session(request_token, api_secret=self.config['zerodha']['api_secret'])
            access_token = data['access_token']
            
            # Set token on the main Kite instance and persist to .env
            self.kite.set_access_token(access_token)
            self.config['zerodha']['access_token'] = access_token
            persist_access_token(access_token)
            
            profile = self.kite.profile()
            logging.info(f"Authentication successful. Connected as {profile.get('user_name', 'user')}.")
            
            # Initialize agents now that we have a valid session
            logging.info("Initializing session-dependent agents...")
            self.market_condition_identifier = MarketConditionIdentifier(self.kite, self.config)
            self.order_agent = OrderExecutionAgent(self.kite, self.config)
            self.position_agent = PositionManagementAgent(self.kite, self.config, self.rag_service)
            logging.info("Agents initialized successfully.")
            
            return True
        except Exception as e:
            logging.error(f"Authentication failed: {e}", exc_info=True)
            return False

    async def _capture_starting_capital(self):
        """Snapshot capital at session start for the daily-loss circuit breaker."""
        try:
            margins = await asyncio.to_thread(self.kite.margins)
            equity = margins.get('equity', {}).get('available', {})
            cap = equity.get('live_balance') or equity.get('cash') or equity.get('net') or 0
            self.starting_capital = float(cap or 0)
            logging.info(f"Starting capital snapshot: {self.starting_capital:,.2f}")
        except Exception as e:
            logging.warning(f"Could not snapshot starting capital: {e}")
            self.starting_capital = 0.0

    async def _refresh_starting_capital(self) -> None:
        """
        Re-snapshot available capital so the daily-loss limit tracks any mid-day
        deposits/withdrawals. Called before each signal-evaluation cycle and at
        the start of each strategy-reassessment.

        Logs only on material changes (>= ₹100) to avoid polluting the loop with
        identical "capital baseline = X" lines every 5 seconds. The actual
        self.starting_capital field is always updated to the latest value so the
        daily-loss-limit math uses fresh data.
        """
        try:
            margins = await asyncio.to_thread(self.kite.margins)
            equity = (margins or {}).get('equity', {}).get('available', {})
            cap = (
                equity.get('live_balance')
                or equity.get('cash')
                or equity.get('net')
                or 0
            )
            new_capital = float(cap or 0)
            if new_capital <= 0:
                return  # transient broker glitch; keep the previous baseline
            prev = self.starting_capital or 0.0
            if abs(new_capital - prev) >= 100.0:
                rm = self.config.get('risk_management', {})
                limit = (
                    new_capital
                    * float(rm.get('max_daily_loss_percent', 2.5))
                    / 100.0
                )
                logging.info(
                    f"Capital baseline refreshed: {prev:,.2f} -> {new_capital:,.2f} "
                    f"(new daily-loss limit: {limit:,.2f})"
                )
            self.starting_capital = new_capital
        except Exception as e:
            logging.debug(f"Capital refresh skipped (non-fatal): {e}")

    # ----- Sentiment-refresh policy -----------------------------------------

    async def _should_refresh_sentiment(self):
        """
        Returns (should_refresh: bool, reason: str). Decides whether the
        operator should be re-prompted for sentiment + NL-prompt by comparing
        current spot / VIX / automated-news-sentiment against the baseline
        captured the last time the operator confirmed.
        """
        # No prior capture? First time through — must capture.
        if self._cached_sentiment is None:
            return True, "first capture"

        cfg = self.config.get('sentiment_refresh', {}) or {}
        if not cfg.get('enable', True):
            return False, ""

        reasons = []

        # 1. Underlying-spot drift
        spot_thresh = float(cfg.get('spot_change_pct', 1.0))
        if self._sentiment_baseline_spot and self.order_agent is not None:
            try:
                token = self.order_agent.underlying_token
                data = await asyncio.to_thread(self.kite.ltp, str(token))
                spot_now = float((data or {}).get(str(token), {}).get('last_price', 0))
                if spot_now > 0:
                    pct = abs(spot_now - self._sentiment_baseline_spot) / self._sentiment_baseline_spot * 100.0
                    if pct >= spot_thresh:
                        reasons.append(
                            f"underlying moved {pct:.2f}% "
                            f"({self._sentiment_baseline_spot:.2f} -> {spot_now:.2f}, "
                            f"threshold {spot_thresh}%)"
                        )
            except Exception as e:
                logging.debug(f"Spot drift check failed: {e}")

        # 2. VIX drift
        vix_thresh = float(cfg.get('vix_change_pct', 25.0))
        if self._sentiment_baseline_vix and self.market_condition_identifier:
            try:
                vix_token = self.market_condition_identifier.vix_token
                data = await asyncio.to_thread(self.kite.ltp, str(vix_token))
                vix_now = float((data or {}).get(str(vix_token), {}).get('last_price', 0))
                if vix_now > 0:
                    pct = abs(vix_now - self._sentiment_baseline_vix) / self._sentiment_baseline_vix * 100.0
                    if pct >= vix_thresh:
                        reasons.append(
                            f"VIX moved {pct:.2f}% "
                            f"({self._sentiment_baseline_vix:.2f} -> {vix_now:.2f}, "
                            f"threshold {vix_thresh}%)"
                        )
            except Exception as e:
                logging.debug(f"VIX drift check failed: {e}")

        # 3. Automated news-sentiment regime flip (BULL <-> BEAR)
        if cfg.get('on_sentiment_flip', True) and self._sentiment_baseline_auto:
            try:
                new_auto = self.sentiment_agent.get_market_sentiment()
                def _regime(s):
                    if s in ('Bullish', 'Very Bullish'): return 'BULL'
                    if s in ('Bearish', 'Very Bearish'): return 'BEAR'
                    return 'NEUTRAL'
                old_r, new_r = _regime(self._sentiment_baseline_auto), _regime(new_auto)
                if old_r != new_r and old_r != 'NEUTRAL' and new_r != 'NEUTRAL':
                    reasons.append(
                        f"automated news-sentiment regime flipped "
                        f"({self._sentiment_baseline_auto} -> {new_auto})"
                    )
            except Exception as e:
                logging.debug(f"Auto-sentiment flip check failed: {e}")

        if reasons:
            return True, "; ".join(reasons)
        return False, ""

    async def _snapshot_sentiment_context(self):
        """Records spot / VIX / auto-sentiment baseline for future drift comparisons."""
        try:
            if self.order_agent is not None:
                token = self.order_agent.underlying_token
                data = await asyncio.to_thread(self.kite.ltp, str(token))
                self._sentiment_baseline_spot = float(
                    (data or {}).get(str(token), {}).get('last_price', 0) or 0
                ) or None
        except Exception as e:
            logging.debug(f"Could not snapshot baseline spot: {e}")
            self._sentiment_baseline_spot = None
        try:
            if self.market_condition_identifier is not None:
                vix_token = self.market_condition_identifier.vix_token
                data = await asyncio.to_thread(self.kite.ltp, str(vix_token))
                self._sentiment_baseline_vix = float(
                    (data or {}).get(str(vix_token), {}).get('last_price', 0) or 0
                ) or None
        except Exception as e:
            logging.debug(f"Could not snapshot baseline VIX: {e}")
            self._sentiment_baseline_vix = None
        try:
            self._sentiment_baseline_auto = self.sentiment_agent.get_market_sentiment()
        except Exception as e:
            logging.debug(f"Could not snapshot baseline auto-sentiment: {e}")
            self._sentiment_baseline_auto = None

    async def _is_daily_loss_breached(self) -> bool:
        rm = self.config.get('risk_management', {})
        if not rm.get('enable_daily_loss_limit', False):
            return False
        if not self.starting_capital or self.starting_capital <= 0:
            return False
        max_loss_pct = float(rm.get('max_daily_loss_percent', 2.5))
        max_loss_amt = self.starting_capital * (max_loss_pct / 100.0)
        if self.realized_pnl_today <= -abs(max_loss_amt):
            logging.error(
                f"DAILY LOSS LIMIT BREACHED: realized={self.realized_pnl_today:,.2f} "
                f"limit={-max_loss_amt:,.2f}. Halting new entries."
            )
            return True
        return False

    async def _is_vix_too_high(self) -> bool:
        max_vix = float(self.config['trading_flags'].get('max_vix_level', 0) or 0)
        if max_vix <= 0:
            return False
        try:
            vix_token = self.market_condition_identifier.vix_token
            ltp_data = await asyncio.to_thread(self.kite.ltp, str(vix_token))
            vix = ltp_data[str(vix_token)]['last_price']
            if vix > max_vix:
                logging.warning(f"VIX gate: {vix:.2f} > max {max_vix}. Blocking new entries.")
                return True
        except Exception as e:
            logging.debug(f"VIX gate check failed (non-fatal): {e}")
        return False

    @staticmethod
    def _parse_hhmm(value):
        if not value:
            return None
        try:
            hh, mm = [int(x) for x in str(value).split(':')]
            return datetime.time(hh, mm)
        except Exception:
            return None

    def _no_trade_window_reason(self):
        """Returns a string explaining why we're in a no-trade window, or None.

        Uses self.effective_entry_start_time which may have been pulled forward to
        09:15 by _compute_effective_entry_start (event-day or open-gap override).
        """
        flags = self.config['trading_flags']
        now = datetime.datetime.now().time()

        start = self.effective_entry_start_time or self._parse_hhmm(flags.get('entry_start_time'))
        if start and now < start:
            return f"before effective entry start {start.strftime('%H:%M')}"

        cutoff = self._parse_hhmm(flags.get('entry_cutoff_time'))
        # On expiry day, use the tighter expiry-day cutoff if configured.
        if getattr(self, "is_expiry_day", False):
            exp_cfg = self.config.get('expiry_day_overrides', {}) or {}
            if exp_cfg.get('enable', True):
                exp_cutoff = self._parse_hhmm(exp_cfg.get('entry_cutoff_time'))
                if exp_cutoff and (cutoff is None or exp_cutoff < cutoff):
                    cutoff = exp_cutoff
        if cutoff and now >= cutoff:
            return f"past entry_cutoff_time {cutoff.strftime('%H:%M')}"

        lunch_start = self._parse_hhmm(flags.get('lunch_pause_start'))
        lunch_end = self._parse_hhmm(flags.get('lunch_pause_end'))
        if lunch_start and lunch_end and lunch_start <= now < lunch_end:
            return f"lunch pause {lunch_start.strftime('%H:%M')}-{lunch_end.strftime('%H:%M')}"
        return None

    async def _compute_effective_entry_start(self):
        """
        Computes today's effective entry-start time. Defaults to entry_start_time
        in config (typically 09:30), but pulls forward to 09:15 when:
          - today is a known macro event day (FED / RBI / etc.) per EconomicCalendar, OR
          - the open-gap vs prior-day close is >= early_entry_gap_threshold_percent.
        Cached in self.effective_entry_start_time so it's a one-shot per session.
        """
        flags = self.config['trading_flags']
        default_start = self._parse_hhmm(flags.get('entry_start_time')) or datetime.time(9, 30)
        early_start = datetime.time(9, 15)

        # Already decided this session.
        if self.effective_entry_start_time is not None:
            return self.effective_entry_start_time

        today = datetime.date.today()
        chosen = default_start
        reason = None

        # 1. Event-day override.
        try:
            calendar = getattr(self.market_condition_identifier, 'calendar', None)
            event = calendar.get_event_for_date(today) if calendar else None
            if event:
                chosen = early_start
                reason = f"event-day ({event})"
        except Exception as e:
            logging.debug(f"Event-day check failed: {e}")

        # 2. Open-gap override (only checked if event-day didn't already trigger).
        if chosen != early_start:
            threshold = float(flags.get('early_entry_gap_threshold_percent', 0) or 0)
            if threshold > 0 and self.order_agent is not None:
                try:
                    token = self.order_agent.underlying_token
                    hist = await asyncio.to_thread(
                        self.kite.historical_data, token,
                        today - datetime.timedelta(days=10), today, "day",
                    )
                    df = pd.DataFrame(hist)
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date']).dt.date
                        prev = df[df['date'] < today].tail(1)
                        if not prev.empty:
                            prev_close = float(prev.iloc[0]['close'])
                            ltp_data = await asyncio.to_thread(self.kite.ltp, str(token))
                            ltp = ltp_data[str(token)]['last_price']
                            gap_pct = abs(ltp - prev_close) / prev_close * 100.0
                            if gap_pct >= threshold:
                                chosen = early_start
                                reason = (f"open-gap {gap_pct:.2f}% >= threshold {threshold}% "
                                          f"(prev_close={prev_close:.2f}, ltp={ltp:.2f})")
                except Exception as e:
                    logging.debug(f"Open-gap check failed (using default start): {e}")

        self.effective_entry_start_time = chosen
        if chosen == early_start and reason:
            logging.info(f"Early-entry override active: {reason}. Allowing entries from 09:15.")
        else:
            logging.info(f"Effective entry start: {chosen.strftime('%H:%M')} (default).")
        return chosen

    def is_market_open(self):
        """Checks if the current time is within Indian market trading hours and not a holiday."""
        now_dt = datetime.datetime.now()
        now = now_dt.time()
        market_open = datetime.time(9, 15)
        market_close = datetime.time(15, 30)
        if now_dt.weekday() >= 5:
            return False
        if is_nse_holiday(now_dt.date()):
            return False
        return market_open <= now <= market_close

    def is_pre_market_window(self) -> bool:
        """
        True when we're in the pre-market warm-up window on a trading day
        (default 08:50 IST -> 09:15 IST). Lets the bot do auth, sentiment
        capture, strategy selection, etc. in advance so it's ready to fire
        at the open instead of starting cold at 09:15.
        """
        now_dt = datetime.datetime.now()
        if now_dt.weekday() >= 5:
            return False
        if is_nse_holiday(now_dt.date()):
            return False
        flags = self.config.get('trading_flags', {}) or {}
        start = self._parse_hhmm(flags.get('pre_market_start_time', '08:50')) \
            or datetime.time(8, 50)
        return start <= now_dt.time() < datetime.time(9, 15)

    async def _wait_for_market_open(self):
        """
        Async sleep until 09:15:30 IST (a few seconds past open so the LTP
        feed has actually printed at least one tick before any gap-based
        decisions run). Logs once per minute so the wait is visible in the
        terminal; sleeps in 5-second chunks so Ctrl+C remains responsive.
        """
        target = datetime.datetime.combine(
            datetime.date.today(), datetime.time(9, 15, 30)
        )
        print("\n" + "=" * 78)
        print("  Pre-market setup complete. Holding until market opens at 09:15 IST...")
        print("=" * 78)
        last_announced_min = -1
        while True:
            now = datetime.datetime.now()
            remaining = (target - now).total_seconds()
            if remaining <= 0:
                break
            mins_left = int(remaining // 60)
            if mins_left != last_announced_min:
                secs = int(remaining - mins_left * 60)
                logging.info(f"Pre-market hold: market opens in {mins_left}m {secs}s.")
                last_announced_min = mins_left
            await asyncio.sleep(min(5.0, max(1.0, remaining)))
        print("\n  Market is now open. Entering trading loop.\n")
        logging.info("Pre-market hold released; market is open.")

    def get_next_trading_day(self):
        """Calculates the next NSE trading day (skips weekends and holidays)."""
        today = datetime.date.today()
        next_day = today + datetime.timedelta(days=1)
        while next_day.weekday() >= 5 or is_nse_holiday(next_day):
            next_day += datetime.timedelta(days=1)
        return next_day

    def _next_market_open_str(self) -> str:
        """
        Returns 'today at 9:15 AM' if today is a trading day and we're still
        before 9:15, else 'next trading day at 9:15 AM' phrasing for the
        closed-info banner.
        """
        now_dt = datetime.datetime.now()
        today = now_dt.date()
        if (now_dt.weekday() < 5
                and not is_nse_holiday(today)
                and now_dt.time() < datetime.time(9, 15)):
            return "today at 9:15 AM"
        next_day = self.get_next_trading_day()
        return f"{next_day.strftime('%A, %d %B')} at 9:15 AM"

    async def setup(self):
        """
        Sets up the bot for the trading day, including sentiment analysis, RAG context,
        and strategy selection. This can also be called to re-assess the strategy.
        """
        self.bot_state = "SETUP"
        logging.info("--- Running Bot Setup & Strategy Assessment ---")

        # On reassessment runs (when a starting baseline already exists), refresh
        # capital so the strategy decision and any downstream gates work off
        # current account state, not the morning snapshot.
        if self.starting_capital is not None and self.starting_capital > 0:
            await self._refresh_starting_capital()

        try:
            today = datetime.date.today()
            # 1. Get Market Conditions
            todays_conditions = self.market_condition_identifier.get_conditions_for_date(today)
            if 'UNKNOWN' in todays_conditions:
                self.no_trade_reason = "Could not determine market conditions."; return False

            # 2. Determine Sentiment — capture ONCE, reuse on reassessment unless
            # the market has materially shifted (spot/VIX drift or news-sentiment flip).
            should_refresh, refresh_reason = await self._should_refresh_sentiment()

            if should_refresh:
                if self._cached_sentiment is not None:
                    # We're re-prompting (not first capture). Loud notice so the
                    # operator knows why the bot is asking again.
                    logging.warning(
                        f"Re-prompting operator: significant market shift since last "
                        f"sentiment capture — {refresh_reason}"
                    )
                    if self._is_interactive_tty():
                        print("\n" + "!" * 78)
                        print("  Significant market shift detected:")
                        print(f"    {refresh_reason}")
                        print("  Please reconfirm market sentiment.")
                        print("!" * 78)

                self.day_sentiment = await self._resolve_sentiment()
                self._cached_sentiment = self.day_sentiment
                await self._snapshot_sentiment_context()
            else:
                logging.info(
                    f"Reusing cached sentiment '{self._cached_sentiment}' — "
                    f"no material market shift since last capture."
                )
                self.day_sentiment = self._cached_sentiment

            if self.day_sentiment == "Neutral":
                self.no_trade_reason = "Market sentiment is Neutral. No new entries today."
                return False

            logging.info(f"Today's Market Conditions: {todays_conditions} | Final Sentiment: {self.day_sentiment}")

            # 3. Get User Prompt — captured once when sentiment is captured. Reused
            # on subsequent reassessments unless we're refreshing (i.e. there was a
            # market shift big enough to ask for both inputs again).
            if self.config['trading_flags'].get('enable_natural_language_prompt', False):
                if should_refresh:
                    user_prompt = (
                        (self.config.get("daily_overrides", {}) or {}).get("nl_prompt", "")
                        or os.environ.get("DAILY_NL_PROMPT", "")
                    )
                    if not user_prompt and self._is_interactive_tty():
                        timeout = float(self.config['trading_flags'].get(
                            'operator_input_timeout_seconds', 20))
                        result = await self._input_with_timeout(
                            "Enter trading observation or preference (or press Enter): ",
                            timeout=timeout,
                        )
                        # None (timeout) and "" (empty) both mean "no observation".
                        user_prompt = result if result else ""
                    self._cached_nl_prompt = user_prompt
                else:
                    user_prompt = self._cached_nl_prompt or ""
            else:
                user_prompt = ""

            # 4. Conditional RAG Context
            rag_context = None
            use_rag_flag = self.config['trading_flags'].get('use_rag', False)
            rag_min_days = self.config['trading_flags'].get('rag_min_trading_days', 5)
            
            if use_rag_flag:
                trade_log_df = self.rag_service._load_data(self.rag_service.trade_log_path)
                if trade_log_df is not None and not trade_log_df.empty:
                    trading_days = pd.to_datetime(trade_log_df['Timestamp']).dt.date.nunique()
                    if trading_days >= rag_min_days:
                        logging.info(f"Sufficient historical data found ({trading_days} days). Activating RAG.")
                        rag_context = self.rag_service.retrieve_context_for_strategy_selection(todays_conditions)
                    else:
                        logging.warning(f"RAG disabled: Insufficient data. Found {trading_days}, need {rag_min_days}.")
                else:
                    logging.warning("RAG disabled: No trade log found.")
            else:
                logging.info("RAG is disabled in config.yaml.")
            
            # 5. Select Strategy
            best_strategy_name = await self.langgraph_agent.get_recommended_strategy(todays_conditions, user_prompt, rag_context)
            self.active_strategy_name = best_strategy_name
            self.active_strategy = get_strategy(best_strategy_name, self.kite, self.config)
            
            # 6. Finalize Setup
            initialize_trade_log()

            # CPR from the most recent trading day strictly before `today`.
            # Fails gracefully on network blips: a transient outage during this
            # one fetch must not kill the whole startup, because most strategies
            # don't even use CPR (Momentum_VWAP_RSI, Supertrend_MACD, EMA_Cross_RSI,
            # Reversal_Detector, etc.). Strategies that DO use CPR will simply
            # skip when pivots is empty.
            self.position_agent.cpr_pivots = {}
            try:
                token = self.order_agent.underlying_token
                hist = None
                for attempt in range(2):
                    try:
                        hist = await asyncio.to_thread(
                            self.kite.historical_data,
                            token, today - datetime.timedelta(days=10), today, "day",
                        )
                        break
                    except Exception as fetch_err:
                        if attempt == 0:
                            logging.warning(
                                f"CPR fetch attempt 1 failed ({fetch_err}); retrying in 3s."
                            )
                            await asyncio.sleep(3)
                        else:
                            raise

                day_df = pd.DataFrame(hist or [])
                if not day_df.empty:
                    day_df["date_only"] = pd.to_datetime(day_df["date"]).dt.date
                    prior = day_df[day_df["date_only"] < today].tail(1)
                    if not prior.empty:
                        self.position_agent.cpr_pivots = calculate_cpr(prior)
                        logging.info("CPR pivots calculated for the day.")
                    else:
                        logging.warning(
                            "No prior-day data; CPR-dependent strategies will skip."
                        )
                else:
                    logging.warning("Empty daily history; CPR pivots unavailable.")
            except Exception as e:
                logging.error(
                    f"CPR pivot fetch failed (non-fatal): {e}. "
                    f"Continuing without CPR — strategies that need pivots will skip."
                )
            
            self.bot_state = "AWAITING_SIGNAL"
            self.awaiting_signal_since = datetime.datetime.now() # Reset the reassessment timer
            logging.info(f"Setup complete. Active strategy: '{self.active_strategy.name}'.")
            return True
        except Exception as e:
            logging.error(f"Setup failed: {e}", exc_info=True)
            self.no_trade_reason = str(e)
            return False

    @staticmethod
    def _is_interactive_tty() -> bool:
        """True only if stdin is connected to a real terminal — never block on input() in headless runs."""
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except Exception:
            return False

    async def _input_with_timeout(self, prompt: str, timeout: float = 20.0):
        """
        Reads a line from stdin with a timeout. Returns the stripped input
        line, "" if the user just pressed Enter, or None if no input arrived
        within `timeout` seconds (the bot will then take its own decision).

        Implementation note: uses POSIX select() in a worker thread so the
        asyncio event loop stays responsive and no thread is leaked on timeout
        (select doesn't consume any data — readline() is only called if input
        is actually ready).
        """
        if not self._is_interactive_tty():
            return None  # headless: caller falls back to its default

        print(prompt, end='', flush=True)

        def _wait_for_line():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], timeout)
            except Exception as e:
                logging.debug(f"select() on stdin failed: {e}")
                return None
            if not ready:
                return None
            try:
                line = sys.stdin.readline()
            except Exception:
                return None
            return line if line else None  # readline returns "" on EOF

        line = await asyncio.to_thread(_wait_for_line)
        if line is None:
            print(f"\n  [no response in {int(timeout)}s — bot will take its own decision]")
            logging.info(f"Operator input timeout ({int(timeout)}s); bot using its own default.")
            return None
        return line.rstrip("\n").strip()

    def _send_shutdown_report_once(self):
        """
        Sends the daily P&L report, idempotently. Fires at end-of-day, on
        Ctrl+C, on early-stop conditions (token expiry, daily-loss breach),
        or on any unhandled exception. If a report has already been sent
        this session, this is a no-op.
        """
        if self._report_sent:
            return
        self._report_sent = True
        try:
            send_daily_report(self.config, str(datetime.date.today()))
            logging.info("Daily report sent.")
        except Exception as e:
            logging.error(f"Failed to send shutdown report: {e}", exc_info=True)

    async def _resolve_sentiment(self):
        """
        Compute today's sentiment via a hybrid flow:

          1. Hard override from config['daily_overrides']['sentiment'] or
             DAILY_SENTIMENT env var — used directly, no prompt. (For unattended runs.)
          2. Run automated sentiment via SentimentAgent (NewsAPI + TextBlob).
          3. If trading_flags.manual_sentiment_override is True AND we have a TTY:
             show the automated read, ask the operator to confirm or override.
             Press Enter -> accept automated. Type a sentiment -> use that. Invalid
             input re-prompts.
          4. If no TTY (headless run) or manual override is disabled in config:
             use the automated read silently.
        """
        valid = {"Very Bullish", "Bullish", "Bearish", "Very Bearish", "Neutral"}

        # 1. Hard config/env override — wins over everything (used for unattended runs).
        hard_override = (
            (self.config.get("daily_overrides", {}) or {}).get("sentiment")
            or os.environ.get("DAILY_SENTIMENT")
        )
        if hard_override and hard_override in valid:
            logging.info(f"Sentiment hard-override from config/env: {hard_override}")
            return hard_override

        # 2. Automated read.
        try:
            automated = self.sentiment_agent.get_market_sentiment()
        except Exception as e:
            logging.warning(f"Automated sentiment failed ({e}); defaulting to 'Neutral'.")
            automated = "Neutral"
        logging.info(f"Automated sentiment read: {automated}")

        manual_enabled = self.config['trading_flags'].get('manual_sentiment_override', True)
        if not manual_enabled:
            logging.info(f"manual_sentiment_override=false → using automated sentiment '{automated}'.")
            return automated

        # 3. Interactive confirm/override (only on a real TTY).
        if not self._is_interactive_tty():
            logging.info(f"No TTY → using automated sentiment '{automated}' without confirmation.")
            return automated

        # Surface the headlines that drove the automated read so the operator can
        # sanity-check it. Polarity is per-article; the bot's overall verdict is a
        # recency-weighted average of these.
        try:
            headlines = self.sentiment_agent.get_top_headlines(n=10)
        except Exception as e:
            logging.debug(f"Could not fetch top headlines (non-fatal): {e}")
            headlines = []

        if headlines:
            print("\n" + "=" * 78)
            print("Top headlines driving the automated sentiment read:")
            print("=" * 78)
            n_pos = n_neg = n_neu = 0
            for h in headlines:
                p = h["polarity"]
                if p > 0.05:
                    marker = "[+]"
                    n_pos += 1
                elif p < -0.05:
                    marker = "[-]"
                    n_neg += 1
                else:
                    marker = "[~]"
                    n_neu += 1
                title = h["title"]
                if len(title) > 100:
                    title = title[:97] + "..."
                src = f"  ({h['source']})" if h.get("source") else ""
                print(f"  {marker} {p:+.3f}  {title}{src}")
            print("-" * 78)
            print(f"  {len(headlines)} headlines analysed:  +{n_pos} bullish  {n_neg} bearish  ~{n_neu} neutral")
            print("=" * 78)

        prompt_options = sorted(valid)
        # Case-insensitive + whitespace-tolerant lookup. Maps "bullish",
        # "BULLISH", "very  bullish" etc. all to the canonical form.
        canonical_by_norm = {" ".join(s.split()).lower(): s for s in valid}
        timeout = float(self.config['trading_flags'].get('operator_input_timeout_seconds', 20))

        while True:
            user_input = await self._input_with_timeout(
                f"\nAutomated sentiment: {automated}\n"
                f"Press Enter to accept, or type one of {prompt_options} to override: ",
                timeout=timeout,
            )

            if user_input is None:
                # Timeout — bot accepts its own automated read.
                logging.info(f"Sentiment auto-resolved on timeout: {automated}")
                return automated
            if user_input == "":
                logging.info(f"Operator accepted automated sentiment: {automated}")
                return automated

            normalized = " ".join(user_input.split()).lower()
            canonical = canonical_by_norm.get(normalized)
            if canonical:
                logging.info(f"Operator overrode {automated} → {canonical}")
                return canonical

            logging.warning(
                f"Invalid input '{user_input}'. Choose from {prompt_options} "
                f"(case-insensitive) or press Enter to accept '{automated}'."
            )

    async def display_market_closed_info(self):
        """Fetches and displays EOD info when the bot is run outside trading hours."""
        logging.warning("Market is currently closed.")
        try:
            token = self.order_agent.underlying_token
            to_date = datetime.date.today()
            from_date = to_date - datetime.timedelta(days=7)
            hist_data = await asyncio.to_thread(self.kite.historical_data, token, from_date, to_date, "day")
            
            if hist_data:
                last_day = hist_data[-1]
                print("\n--- Last Trading Day Summary ---")
                print(f"Date:   {last_day['date'].strftime('%A, %d %B %Y')}")
                print(f"Open:   {last_day['open']:.2f}")
                print(f"Close:  {last_day['close']:.2f}")
                print("---------------------------------")

            news = self.sentiment_agent._get_news_articles()
            if news and news.get('articles'):
                print("\n--- Latest News Headlines ---")
                for article in news['articles'][:5]:
                    print(f"- {article['title']}")
                print("---------------------------------")
        except Exception as e:
            logging.error(f"Could not fetch post-market data: {e}")
        
        print(
            f"\nMarket is closed right now, enjoy your day and come back "
            f"{self._next_market_open_str()} to trade like a Warrior!\n"
        )

    @staticmethod
    def _timeframe_minutes(timeframe: str) -> int:
        """Parse a Kite timeframe string like '5minute' / '15minute' / 'minute' into minutes."""
        if not timeframe:
            return 5
        if timeframe == "minute":
            return 1
        if timeframe.endswith("minute"):
            try:
                return int(timeframe.replace("minute", ""))
            except ValueError:
                return 5
        return 5

    def _current_bar_index(self, timeframe: str) -> int:
        """Index of the bar that *just closed* — increments when a new bar boundary is crossed."""
        m = self._timeframe_minutes(timeframe)
        now = datetime.datetime.now()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return int((now - midnight).total_seconds() // (m * 60))

    async def _get_underlying_bars(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Returns the underlying intraday bar history. Cached until a new bar boundary
        is crossed — eliminates the 2-calls/min historical_data hammer-on of the
        previous version.
        """
        timeframe = self.config["trading_flags"]["chart_timeframe"]
        bar_idx = self._current_bar_index(timeframe)
        if (not force_refresh
                and self._bars_cache is not None
                and self._bars_cached_at_bar == bar_idx):
            return self._bars_cache

        hist = await asyncio.to_thread(
            self.kite.historical_data,
            self.order_agent.underlying_token,
            datetime.datetime.now() - datetime.timedelta(days=5),
            datetime.datetime.now(),
            timeframe,
        )
        df = pd.DataFrame(hist)
        if not df.empty:
            df = calculate_all_indicators(df, self.config)
        self._bars_cache = df
        self._bars_cached_at_bar = bar_idx
        return df

    async def _aligned_sleep(self, max_seconds: float = 30.0) -> None:
        """
        Sleep until the next 5s tick (bounded by max_seconds). Cheap to call,
        responsive to broker SL-M fills, and avoids drifting against the bar clock.
        """
        sleep_for = max(1.0, min(max_seconds, 5.0))
        await asyncio.sleep(sleep_for)

    def _record_realized_pnl(self, delta_pnl: float) -> None:
        self.realized_pnl_today += float(delta_pnl or 0)
        try:
            save_daily_pnl(self._today_str, self.realized_pnl_today)
        except Exception as e:
            logging.warning(f"Could not persist daily P&L: {e}")

    async def run(self):
        """The main event loop for the trading bot."""
        # Accept startup during market hours OR the pre-market warm-up window
        # (default 08:50 -> 09:15 IST). Anything else -> closed-info banner.
        if not (self.is_market_open() or self.is_pre_market_window()):
            await self.display_market_closed_info()
            return  # No report — bot never attempted trading.

        if self.is_pre_market_window() and not self.is_market_open():
            now_str = datetime.datetime.now().strftime('%H:%M:%S')
            print("\n" + "=" * 78)
            print(f"  Pre-market start at {now_str} — running setup ahead of 09:15 open.")
            print("=" * 78)
            logging.info("Started during pre-market window; running setup in advance.")

        try:
            await self._run_inner()
        except (KeyboardInterrupt, asyncio.CancelledError, SystemExit) as e:
            logging.info(f"Bot shutdown signal: {type(e).__name__}.")
        except Exception as e:
            logging.error(f"Unhandled exception in run(): {e}", exc_info=True)
        finally:
            # Always send the daily report on exit — Ctrl+C, token expiry,
            # daily-loss breach, crash, normal market close. Whatever trades
            # happened today get emailed; if none, a "no trades" report goes out.
            self._send_shutdown_report_once()

    async def _run_inner(self):
        """The actual trading orchestration — wrapped by run() in a try/finally."""
        # Reconcile any persisted open position from a previous session before setup.
        try:
            resumed = await self.position_agent.reconcile_open_position()
        except Exception as e:
            logging.warning(f"Reconciliation skipped: {e}")
            resumed = False

        if not await self.setup():
            logging.warning(f"Setup failed. Reason: {self.no_trade_reason or 'Unknown'}. Bot will exit.")
            return  # Finally block in run() will send the report.

        await self._capture_starting_capital()

        # ---------- Expiry-day detection + banner (no LTP needed; safe pre-market) ----------
        self.is_expiry_day = False
        try:
            self.is_expiry_day = self.order_agent.is_weekly_expiry_today()
        except Exception as e:
            logging.debug(f"Expiry-day check failed (non-fatal): {e}")
        if self.is_expiry_day:
            exp_cfg = self.config.get('expiry_day_overrides', {}) or {}
            print("\n" + "=" * 78)
            print("  Today's the weekly expiry day, so consider trading safely.")
            print("=" * 78)
            if exp_cfg.get('enable', True):
                print("  Expiry-day overrides ACTIVE:")
                print(f"    Risk per trade x {float(exp_cfg.get('risk_reduction_factor', 0.5))}")
                print(f"    Max trades today: {int(exp_cfg.get('max_trades', 1))}")
                print(f"    Entry cutoff:    {exp_cfg.get('entry_cutoff_time', '13:00')}")
            else:
                print("  Expiry-day overrides are disabled in config.yaml.")
            print("=" * 78 + "\n")
            logging.warning("Today is weekly expiry day; expiry-day overrides applied.")

        # If we started in pre-market, hold here until the actual open. All
        # setup (auth, sentiment, strategy pick, expiry detection) is done;
        # we just sleep until 09:15:30 IST so the open-gap check below has
        # a real LTP to compare against.
        if not self.is_market_open() and self.is_pre_market_window():
            await self._wait_for_market_open()

        # Compute effective entry start AFTER the market is open so the open
        # gap check uses real LTP (gap = 0 in pre-market — would be a no-op).
        await self._compute_effective_entry_start()

        is_paper = self.config['trading_flags']['paper_trading']
        logging.info(f"Bot running in {'PAPER TRADING' if is_paper else 'LIVE TRADING'} mode.")
        if resumed:
            self.bot_state = "IN_POSITION"
            logging.info("Resuming management of pre-existing position.")

        while self.is_market_open():
            try:
                if self.bot_state == "AWAITING_SIGNAL":
                    # Refresh capital baseline so the daily-loss limit and
                    # downstream sizing reflect any mid-day deposits/withdrawals.
                    await self._refresh_starting_capital()
                    # Hard gates that must pass before considering any new entry.
                    if await self._is_daily_loss_breached():
                        self.bot_state = "STOPPED"; continue
                    max_trades = int(self.config['trading_flags']['max_trades_per_day'])
                    if getattr(self, "is_expiry_day", False):
                        exp_cfg = self.config.get('expiry_day_overrides', {}) or {}
                        if exp_cfg.get('enable', True):
                            max_trades = min(max_trades, int(exp_cfg.get('max_trades', max_trades)))
                    if self.trades_today_count >= max_trades:
                        self.bot_state = "STOPPED"; continue

                    # Soft gate: in a no-trade window we just sleep and try again later.
                    no_trade = self._no_trade_window_reason()
                    if no_trade:
                        logging.debug(f"In no-trade window ({no_trade}); waiting.")
                        await asyncio.sleep(30)
                        continue

                    # Strategy reassessment timer.
                    reassessment_period = self.config['trading_flags'].get('strategy_reassessment_period_minutes', 60)
                    if self.awaiting_signal_since and (datetime.datetime.now() - self.awaiting_signal_since).total_seconds() > reassessment_period * 60:
                        logging.warning(f"No trade signal for over {reassessment_period} minutes. Re-assessing strategy...")
                        if not await self.setup():
                            self.bot_state = "STOPPED"; continue

                    day_df_for_signal = await self._get_underlying_bars()
                    if day_df_for_signal is None or day_df_for_signal.empty:
                        logging.debug("Underlying bars unavailable; skipping iteration.")
                        await self._aligned_sleep()
                        continue

                    signal = self.active_strategy.generate_signals(
                        day_df_for_signal, self.day_sentiment,
                        cpr_pivots=self.position_agent.cpr_pivots,
                    )

                    if signal != 'HOLD':
                        is_primary_signal = (signal == 'BUY' and self.day_sentiment in ['Bullish', 'Very Bullish']) or (signal == 'SELL' and self.day_sentiment in ['Bearish', 'Very Bearish'])
                        if getattr(self.active_strategy, 'is_reversal_trade', False) or is_primary_signal:
                            # VIX gate is intentionally checked at the moment of intent so it doesn't
                            # block the loop in a quiet phase.
                            if await self._is_vix_too_high():
                                logging.warning("Skipping entry due to VIX gate.")
                            else:
                                trade_details = (await self.order_agent.place_trade(signal)
                                                 if not is_paper
                                                 else await self.order_agent.get_paper_trade_details(signal))
                                if trade_details:
                                    trade_details['Strategy'] = self.active_strategy_name
                                    self.position_agent.start_trade(trade_details)
                                    if not is_paper:
                                        await self.position_agent.attach_broker_stop_loss(self.order_agent)
                                    self.trades_today_count += 1
                                    self.bot_state = "IN_POSITION"
                                    self.awaiting_signal_since = None
                        else:
                            logging.warning(f"COUNTER-SIGNAL DETECTED: '{signal}' vs sentiment '{self.day_sentiment}'.")

                elif self.bot_state == "IN_POSITION":
                    underlying_df_hist = await self._get_underlying_bars()
                    status = await self.position_agent.manage(
                        is_paper,
                        underlying_hist_df=underlying_df_hist,
                        sentiment_agent=self.sentiment_agent,
                        gemini_api_key=self.config.get('google_api', {}).get('api_key'),
                    )
                    if isinstance(status, dict):
                        log_trade(status)
                        self._record_realized_pnl(status.get('ProfitLoss', 0))
                        self.bot_state = "AWAITING_SIGNAL"
                        self.awaiting_signal_since = datetime.datetime.now()

                await self._aligned_sleep()
            except exceptions.TokenException as e:
                logging.error(f"Zerodha session expired or invalidated: {e}. Halting bot.")
                self._abort = True
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(15)
        
        logging.info("Market is now closed. Shutting down trading loop.")
        # Report is sent by the finally block in run() — no need to call here.


if __name__ == "__main__":
    multiprocessing.freeze_support()
    bot = TradingBotOrchestrator(load_config())
    if bot.authenticate():
        asyncio.run(bot.run())
