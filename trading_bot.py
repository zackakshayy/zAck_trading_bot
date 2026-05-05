import logging
import os
import re
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
        """Snapshot capital once per day for the daily-loss circuit breaker."""
        try:
            margins = await asyncio.to_thread(self.kite.margins)
            equity = margins.get('equity', {}).get('available', {})
            cap = equity.get('live_balance') or equity.get('cash') or equity.get('net') or 0
            self.starting_capital = float(cap or 0)
            logging.info(f"Starting capital snapshot: {self.starting_capital:,.2f}")
        except Exception as e:
            logging.warning(f"Could not snapshot starting capital: {e}")
            self.starting_capital = 0.0

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

    def get_next_trading_day(self):
        """Calculates the next NSE trading day (skips weekends and holidays)."""
        today = datetime.date.today()
        next_day = today + datetime.timedelta(days=1)
        while next_day.weekday() >= 5 or is_nse_holiday(next_day):
            next_day += datetime.timedelta(days=1)
        return next_day

    async def setup(self):
        """
        Sets up the bot for the trading day, including sentiment analysis, RAG context,
        and strategy selection. This can also be called to re-assess the strategy.
        """
        self.bot_state = "SETUP"
        logging.info("--- Running Bot Setup & Strategy Assessment ---")
        
        try:
            today = datetime.date.today()
            # 1. Get Market Conditions
            todays_conditions = self.market_condition_identifier.get_conditions_for_date(today)
            if 'UNKNOWN' in todays_conditions:
                self.no_trade_reason = "Could not determine market conditions."; return False

            # 2. Determine Sentiment — automated read first, optional human override.
            self.day_sentiment = self._resolve_sentiment()
            if self.day_sentiment == "Neutral":
                self.no_trade_reason = "Market sentiment is Neutral. No new entries today."
                return False
            
            logging.info(f"Today's Market Conditions: {todays_conditions} | Final Sentiment: {self.day_sentiment}")

            # 3. Get User Prompt — non-blocking: prefer config / env override, then TTY only.
            user_prompt = ""
            if self.config['trading_flags'].get('enable_natural_language_prompt', False):
                user_prompt = (
                    (self.config.get("daily_overrides", {}) or {}).get("nl_prompt", "")
                    or os.environ.get("DAILY_NL_PROMPT", "")
                )
                if not user_prompt and self._is_interactive_tty():
                    try:
                        user_prompt = input("Enter trading observation or preference (or press Enter): ")
                    except EOFError:
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

    def _resolve_sentiment(self):
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
        while True:
            try:
                user_input = input(
                    f"\nAutomated sentiment: {automated}\n"
                    f"Press Enter to accept, or type one of {prompt_options} to override: "
                ).strip()
            except EOFError:
                logging.info(f"EOF on stdin → using automated sentiment '{automated}'.")
                return automated

            if not user_input:
                logging.info(f"Operator accepted automated sentiment: {automated}")
                return automated
            if user_input in valid:
                logging.info(f"Operator overrode {automated} → {user_input}")
                return user_input
            logging.warning(
                f"Invalid input '{user_input}'. Choose from {prompt_options} "
                f"or press Enter to accept '{automated}'."
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
        
        next_day_str = self.get_next_trading_day().strftime('%A, %d %B')
        print(f"\nMarket is closed right now, enjoy your day and come back on {next_day_str} at 9:15 AM to trade like a Warrior!\n")

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
        if not self.is_market_open():
            await self.display_market_closed_info()
            return

        # Reconcile any persisted open position from a previous session before setup.
        try:
            resumed = await self.position_agent.reconcile_open_position()
        except Exception as e:
            logging.warning(f"Reconciliation skipped: {e}")
            resumed = False

        if not await self.setup():
            logging.warning(f"Setup failed. Reason: {self.no_trade_reason or 'Unknown'}. Bot will exit.")
            send_daily_report(self.config, str(datetime.date.today()))
            return

        await self._capture_starting_capital()
        await self._compute_effective_entry_start()

        # ---------- Expiry-day banner + active overrides ----------
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

        is_paper = self.config['trading_flags']['paper_trading']
        logging.info(f"Bot running in {'PAPER TRADING' if is_paper else 'LIVE TRADING'} mode.")
        if resumed:
            self.bot_state = "IN_POSITION"
            logging.info("Resuming management of pre-existing position.")

        while self.is_market_open():
            try:
                if self.bot_state == "AWAITING_SIGNAL":
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
        send_daily_report(self.config, str(datetime.date.today()))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    bot = TradingBotOrchestrator(load_config())
    if bot.authenticate():
        asyncio.run(bot.run())
