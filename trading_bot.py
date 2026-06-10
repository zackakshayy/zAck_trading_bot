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
from youtube_sentiment import YouTubeSentimentAgent
from langgraph_agent import LangGraphAgent
from market_trend import MarketTrendAnalyzer
from regime import RegimeClassifier, ALL_STRATEGIES
from fii import fetch_fii_bias
from strategy_factory import get_strategy
from backtester import run_backtest
from reporting import (
    send_daily_report, initialize_trade_log, log_trade, send_monthly_report,
    send_loss_analysis_email, send_token_expiry_alert, send_weekly_report,
)
from loss_analyzer import build_loss_report
from indicators import calculate_cpr, is_trend_overextended, check_momentum_divergence
from indicator_calculator import calculate_all_indicators
from market_context import MarketConditionIdentifier
from rag_service import RAGService
from pcr_feed import PCRFeed
from infra import (
    is_nse_holiday,
    NSE_HOLIDAY_NAMES,
    load_daily_pnl,
    load_weekly_pnl,
    safe_ltp,
    save_daily_pnl,
    save_weekly_pnl,
    state_path,
    atomic_write_json,
    read_json,
    load_week_trade_days,
    add_week_trade_day,
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

# Also persist logs to a rotating file so the dashboard can tail them. This is a
# FileHandler, so run()'s status-line handler swap leaves it in place.
try:
    import logging.handlers as _logging_handlers
    os.makedirs("output", exist_ok=True)
    _file_handler = _logging_handlers.RotatingFileHandler(
        "output/bot.log", maxBytes=5_000_000, backupCount=3
    )
    _file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(_file_handler)
except Exception as _log_exc:  # never let logging setup kill startup
    logging.warning(f"Could not attach rotating log file: {_log_exc}")

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

class _StatusLineAwareHandler(logging.StreamHandler):
    """
    A StreamHandler that clears the live status line before emitting any log
    record so log output never overlaps with the ticker. The status line is
    redrawn by the 1-second ticker task on its next tick.

    Only emits when the attached bot instance has an interactive TTY; otherwise
    it falls through to the default handler behaviour so headless / file logs
    are unaffected.
    """

    def __init__(self, bot: "TradingBotOrchestrator"):
        super().__init__(sys.stderr)
        self._bot = bot

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self._bot._is_interactive_tty():
                # Clear the status line first so the log starts at column 0.
                sys.stderr.write(f"\r{' ' * 119}\r")
                sys.stderr.flush()
        except Exception:
            pass
        super().emit(record)


class TradingBotOrchestrator:
    """
    The central orchestrator for the trading bot. Manages state, coordinates agents,
    and runs the main trading loop.
    """
    def __init__(self, config, manual_mode: bool = False):
        self.config = config
        self.kite = KiteConnect(api_key=config['zerodha']['api_key'], timeout=120, debug=True)
        self.active_strategy_name = "None"
        self.active_strategy = None
        # Manual strategy selection mode (--manual flag).
        # When True, the operator picks the strategy from a numbered menu instead
        # of the automated selector. All other logic (SL, targets, risk) is identical.
        self._manual_mode: bool = manual_mode
        self._manual_strategy_name: str | None = None  # operator-chosen strategy

        # Initialize core services
        self.rag_service = RAGService(config)
        self.langgraph_agent = LangGraphAgent(config, self.rag_service)
        # YouTube verdict agent (constructed first; the SentimentAgent blends
        # its cached verdicts into the daily sentiment score).
        self.youtube_agent = YouTubeSentimentAgent(
            config,
            gemini_api_key=(config.get('google_api') or {}).get('api_key', ''),
        )
        self.sentiment_agent = SentimentAgent(config, youtube_agent=self.youtube_agent)
        # Pre-market price-action / global-cues trend analyzer (CE/PE verdict).
        # Uses self.kite for NIFTY gap + price action; yfinance for global feeds.
        self.market_trend_analyzer = MarketTrendAnalyzer(self.kite, config)
        # Market-regime classifier (playbook Part 3) + per-session regime state.
        self.regime_classifier = RegimeClassifier(config)
        self.day_regime: str = "PENDING"
        self._regime_reason: str = ""
        self._regime_excluded: set = set()          # strategies excluded by regime
        self._regime_allowed: tuple = ()            # regime-preferred strategies
        self._regime_entry_not_before = None        # post-event entry-wait gate
        self._first_setup: bool = True              # day-level sit-out only on first setup
        self._fii_bias = None                       # best-effort FII positioning (informational)
        self._today_vix: float = 0.0                # cached for journaling
        self._trend_composite: float = 0.0          # market_trend directional score (-1..1)
        self.day_conviction: float = 0.0            # 0..1 conviction score
        self.day_conviction_level: str = "LOW"      # LOW / MEDIUM / HIGH
        self._expiry_gamma_active: bool = False      # 0-DTE expiry gamma-scalp mode
        # Setup-score funnel (gate-attribution): how many signals were scored,
        # how many passed, and the best score seen — turns "the bot sat idle"
        # into "the best setup scored 61 vs threshold 65 because PCR+15m failed".
        self._scored_today: int = 0
        self._score_passed_today: int = 0
        self._best_score_today: float = 0.0
        self._best_score_info: str = ""
        self._last_score_size: float = 1.0
        self._last_score = None
        self._last_score_block_key = None
        # Strategy whitelist (the cull) — populated each setup() from config.
        self._whitelist_excluded: set = set()

        # Defer initialization of session-dependent agents until after authentication
        self.market_condition_identifier = None
        self.order_agent = None
        self.position_agent = None

        # State variables
        self.day_sentiment = ""
        self.trades_today_count = 0
        self.no_trade_reason = None
        self.bot_state = "STARTING"
        self._trading_mode = "MODERATE"  # updated each setup() call
        self.last_processed_timestamp = None
        self.awaiting_signal_since = None
        # Realized P&L tracking for daily-loss and weekly-loss circuit breakers.
        # Both are persisted to disk so a same-day restart doesn't reset the caps.
        self._today_str = datetime.date.today().isoformat()
        self._today_week_str = datetime.date.today().strftime("%G-W%V")
        self.realized_pnl_today = load_daily_pnl(self._today_str)
        self.realized_pnl_week  = load_weekly_pnl(self._today_week_str)
        if self.realized_pnl_today != 0.0:
            logging.info(
                f"Resuming with persisted realized P&L for {self._today_str}: "
                f"{self.realized_pnl_today:,.2f}"
            )
        if self.realized_pnl_week != 0.0:
            logging.info(
                f"Resuming with persisted weekly P&L for {self._today_week_str}: "
                f"{self.realized_pnl_week:,.2f}"
            )
        # Consecutive-loss counter — resets each bot restart (session-scoped).
        # The circuit breaker fires once N losses occur in an unbroken streak
        # within this session; winning trades reset the streak to 0.
        self.consecutive_losses: int = 0
        # Session trade ledger for the live status line + end-of-day summary.
        self.wins_today: int = 0
        self.losses_today: int = 0
        self.scratch_today: int = 0
        self.completed_trades: list = []   # [{symbol, type, gross, costs, net, strategy}]
        self._session_summary_printed: bool = False
        # Tally of *why* entries were blocked, for the "why no trade" rationale.
        self._block_reasons: dict = {}
        # Restore today's trade ledger from disk so a same-day RESTART shows the
        # FULL day (trades, W/L, trade count) — not just trades since this run
        # started. The end-of-day report/summary then always covers the whole day.
        self._load_ledger()
        # Warn loudly at startup if the risk numbers can't coexist (risk > daily/2).
        self._validate_risk_coherence()
        self.starting_capital = None
        self.ending_capital = None   # captured at end-of-day for the reports
        # Effective entry-start time for today (None until _compute_effective_entry_start runs).
        self.effective_entry_start_time = None
        # Professional risk controls
        self._trade_size_multiplier = 1.0   # reduced after losses, restored on wins
        self._day_quality = 'UNKNOWN'       # set each setup() call
        self._last_reported_day_quality: str | None = None  # suppresses repeat DayQuality logs
        self._last_reported_scalp_state: bool = False       # suppresses repeat RangeScalp logs
        self._last_counter_signal: tuple | None = None       # suppresses repeat COUNTER-SIGNAL logs
        # Re-entry guard state: after an exit, block re-buying the same direction
        # for a cooldown so the bot doesn't get chopped in/out of the same setup.
        self._last_exit_direction: str | None = None
        self._last_exit_symbol: str | None = None
        self._last_exit_time = None
        self._ticker_paused: bool = False                   # paused while operator types input
        # Today's market-condition tags — stashed by setup() for the loss analyzer.
        self.todays_conditions = set()
        # Signed open-gap % vs prior close (set by _compute_effective_entry_start
        # once market is open; consumed by the strategy selector's Layer-2 override).
        self.open_gap_pct = None
        # Strategy cooldown bookkeeping. When a strategy produces 0 non-HOLD
        # signals during a reassessment window, it's "cooled" for
        # `strategy_cooldown_minutes` (default 60) — the selector skips it
        # until the cooldown expires. Time-based (not "rest of day") so
        # strategies can rotate back in as market conditions evolve.
        # Maps: strategy_name -> datetime when the cooldown expires.
        self._strategy_cooldown_until: dict = {}
        self._signals_seen_for_active_strategy: int = 0
        # Cache for the underlying intraday bar data (refreshed only when a new bar closes).
        self._bars_cache = None
        self._bars_cached_at_bar = None
        # Separate cache for 15-minute bars used by the confirmation gate.
        self._bars_15m_cache = None
        self._bars_15m_cached_at_bar = None
        # PCR feed — initialised after authentication.
        self.pcr_feed: PCRFeed | None = None
        # Last PCR result dict (tag, pcr, put_oi, call_oi …). Default to empty so
        # gate is bypassed until the first successful fetch.
        self._pcr_data: dict = {}
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
        # ONE-SHOT force-trade diagnostic flag. Read once at startup from
        # config.trading_flags.force_one_trade_today and auto-disarmed after
        # the first trade lands. Bypasses sentiment / VIX / IVR / IV-RV gates
        # for that one trade — liquidity, daily-loss, max-trades stay enforced.
        self._force_mode_armed = bool(
            (config.get('trading_flags') or {}).get('force_one_trade_today', False)
        )
        # Rate-limit the "invalid input" warning so a typo doesn't spam the log.
        self._last_stdin_override_warn: float = 0.0
        # Printed once after the first sentiment lock-in; not repeated on refreshes.
        self._override_hint_shown: bool = False

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
            self.pcr_feed = PCRFeed(self.kite, self.config)
            logging.info("Agents initialized successfully.")
            
            return True
        except Exception as e:
            logging.error(f"Authentication failed: {e}", exc_info=True)
            return False

    async def _validate_token(self) -> bool:
        """
        Confirms the current Zerodha access token is still valid by making a
        cheap API call (profile). Returns True if valid.

        On failure (TokenException / any exception) it:
          1. Logs the error with the Kite login URL.
          2. Sends an email alert via send_token_expiry_alert() so the operator
             knows to refresh the token even if they're not watching the terminal.
          3. Sets self._abort = True so the main loop exits cleanly.
          4. Returns False.

        Called once at the start of _run_inner(), before any setup work, so a
        stale token surfaces immediately rather than mid-session.
        """
        try:
            await asyncio.to_thread(self.kite.profile)
            return True
        except Exception as exc:
            login_url = self.kite.login_url() if self.kite else "https://kite.zerodha.com"
            logging.error(
                f"Token validation failed: {exc}\n"
                f"Refresh your token at: {login_url}"
            )
            if self._is_interactive_tty():
                print("\n" + "!" * 78)
                print("  Zerodha access token is INVALID or EXPIRED.")
                print(f"  Error: {exc}")
                print(f"  Refresh at: {login_url}")
                print("!" * 78 + "\n")
            try:
                send_token_expiry_alert(self.config, str(exc), login_url)
            except Exception as mail_exc:
                logging.debug(f"Token-expiry alert email failed: {mail_exc}")
            self._abort = True
            return False

    async def _run_startup_backtest(self) -> None:
        """
        Runs an optional warm-up backtest whose results are written to the trade
        log so the RAG service has real signal history to retrieve from.

        Gated by trading_flags.run_startup_backtest: true (default false).

        Date-range alignment fix
        ─────────────────────────
        The legacy `backtest_years: 2` setting generates 500+ rows of historical
        trades, but the RAG service only looks at the most recent
        `config.rag.recency_window_days` (default 30) trading days. Running 2 years
        of backtest to feed 30 days of RAG is wasteful and slow.

        This method aligns the two: it derives the from_date as
          today - max(recency_window_days + 20, backtest_years_days)
        where the +20-day buffer ensures enough context for RAG's
        min_trades_per_strategy check. If backtest_years is explicitly larger
        than the aligned window, a warning is logged so the operator knows.
        """
        flags = self.config.get('trading_flags', {})
        if not flags.get('run_startup_backtest', False):
            return

        rag_cfg    = self.config.get('rag', {}) or {}
        recency_d  = int(rag_cfg.get('recency_window_days', 30))
        bt_years   = float(flags.get('backtest_years', 2))
        bt_days    = int(bt_years * 365)

        # Aligned window: RAG recency + buffer.  Never shorter than 60 calendar days.
        aligned_days = max(recency_d + 20, 60)

        if bt_days > aligned_days * 2:
            logging.warning(
                f"[Backtest] backtest_years={bt_years} ({bt_days} days) is much larger "
                f"than the RAG recency_window_days={recency_d}. "
                f"Using aligned window of {aligned_days} days to avoid generating "
                f"stale data that RAG will ignore."
            )

        from_date = datetime.date.today() - datetime.timedelta(days=aligned_days)
        to_date   = datetime.date.today()

        active_strat = getattr(self, 'active_strategy_name', None)
        if not active_strat:
            logging.warning("[Backtest] No active strategy set; skipping startup backtest.")
            return

        logging.info(
            f"[Backtest] Running startup backtest for '{active_strat}' "
            f"from {from_date} to {to_date} ({aligned_days} days)."
        )
        try:
            await asyncio.to_thread(
                run_backtest,
                self.kite,
                self.config,
                active_strat,
                from_date,
                to_date,
            )
            logging.info("[Backtest] Startup backtest complete.")
        except Exception as e:
            logging.warning(f"[Backtest] Startup backtest failed (non-fatal): {e}")

    async def _capture_starting_capital(self):
        """Snapshot capital at session start for the daily-loss circuit breaker."""
        try:
            margins = await asyncio.to_thread(self.kite.margins)
            equity = margins.get('equity', {}).get('available', {})
            cap = equity.get('live_balance') or equity.get('cash') or equity.get('net') or 0
            self.starting_capital = float(cap or 0)
            logging.debug(f"Starting capital snapshot: {self.starting_capital:,.2f}")
        except Exception as e:
            logging.warning(f"Could not snapshot starting capital: {e}")
            self.starting_capital = 0.0

    async def _capture_ending_capital(self):
        """
        Snapshot capital at end-of-day for the daily/weekly reports. Best-effort:
        on early exits or monitor mode (no live session) it stays None.
        """
        try:
            margins = await asyncio.to_thread(self.kite.margins)
            equity = (margins or {}).get('equity', {}).get('available', {})
            cap = equity.get('live_balance') or equity.get('cash') or equity.get('net') or 0
            self.ending_capital = float(cap or 0)
            logging.debug(f"Ending capital snapshot: {self.ending_capital:,.2f}")
            self._persist_ledger()
        except Exception as e:
            logging.debug(f"Could not snapshot ending capital: {e}")

    @staticmethod
    def _dir_of(sentiment: str) -> str:
        s = (sentiment or "").lower()
        if s in ("bullish", "very bullish"):
            return "BULLISH"
        if s in ("bearish", "very bearish"):
            return "BEARISH"
        return "NEUTRAL"

    def _compute_conviction(self) -> tuple:
        """
        Combine regime + sentiment strength + global-cue alignment into a 0..1
        conviction score and a LOW/MEDIUM/HIGH level. High conviction = bigger
        size + hold-the-winner; weak = smaller / cut early. Pure scoring.
        """
        if not (self.config.get('conviction') or {}).get('enable', True):
            return 0.0, "LOW"
        score = 0.0
        # Regime quality.
        rgm = getattr(self, 'day_regime', 'PENDING')
        score += {'TRENDING': 0.35, 'POST_EVENT': 0.25, 'EXPIRY': 0.20,
                  'RANGE': 0.10}.get(rgm, 0.0)
        # Sentiment strength.
        sent = (self.day_sentiment or '')
        score += 0.30 if sent in ('Very Bullish', 'Very Bearish') else \
                 (0.15 if sent in ('Bullish', 'Bearish') else 0.0)
        # Global-cue magnitude (market_trend composite, |.| up to 0.25).
        score += min(0.25, abs(float(self._trend_composite or 0.0)) * 0.5)
        # Directional alignment: regime/trend direction == sentiment direction.
        sent_dir = self._dir_of(sent)
        trend_dir = "BULLISH" if self._trend_composite > 0 else ("BEARISH" if self._trend_composite < 0 else "NEUTRAL")
        fii_dir = (self._fii_bias or {}).get('bias') if self._fii_bias else None
        if sent_dir != "NEUTRAL" and sent_dir == trend_dir:
            score += 0.10
        if fii_dir and fii_dir == sent_dir:
            score += 0.05
        score = max(0.0, min(1.0, score))
        cfg = (self.config.get('conviction') or {})
        hi = float(cfg.get('high_threshold', 0.70))
        med = float(cfg.get('medium_threshold', 0.45))
        level = "HIGH" if score >= hi else ("MEDIUM" if score >= med else "LOW")
        return round(score, 3), level

    def _is_hold_to_close(self, signal: str) -> bool:
        """A trade qualifies to ride to ~15:15 (hold-the-winner) when conviction is
        HIGH, the regime is TRENDING, and the trade direction aligns with the day's
        sentiment direction (so we only let *aligned, high-conviction* trades run)."""
        if not (self.config.get('hold_to_close') or {}).get('enable', True):
            return False
        if self.day_conviction_level != "HIGH" or getattr(self, 'day_regime', '') != "TRENDING":
            return False
        sig_dir = "BULLISH" if signal == "BUY" else ("BEARISH" if signal == "SELL" else "NEUTRAL")
        return sig_dir != "NEUTRAL" and sig_dir == self._dir_of(self.day_sentiment)

    async def _current_vix(self) -> float:
        """Best-effort live India VIX (0.0 on failure)."""
        try:
            tok = self.market_condition_identifier.vix_token
            data = await asyncio.to_thread(self.kite.ltp, str(tok))
            return float((data or {}).get(str(tok), {}).get('last_price', 0) or 0)
        except Exception:
            return 0.0

    def _days_to_next_event(self):
        """Days until the next known macro event (FED/RBI/…) within the horizon,
        or None. Uses the EconomicCalendar already loaded by the condition agent."""
        try:
            cal = self.market_condition_identifier.calendar
            today = datetime.date.today()
            horizon = int((self.config.get('regime') or {}).get('pre_event_days', 3))
            for d in range(1, horizon + 1):
                if cal.get_event_for_date(today + datetime.timedelta(days=d)):
                    return d
        except Exception:
            pass
        return None

    async def _regime_governor(self) -> tuple:
        """
        Returns (sit_out: bool, reason: str). Implements the playbook's
        selectivity/survival governors (Phase A):
          • Weekly frequency cap — once `regime.max_trading_days_per_week`
            distinct trading days are used this ISO week, sit out new days.
          • VIX floor — below `regime.vix_no_trade_below`, the market is too
            calm for option buyers to overcome theta, so stand down.
        Disabled wholesale when regime.enable is false.
        """
        rcfg = (self.config.get('regime') or {})
        if not rcfg.get('enable', True):
            return False, ""

        # (a) Weekly trading-day allotment.
        max_days = int(rcfg.get('max_trading_days_per_week', 5))
        today_str = datetime.date.today().isoformat()
        week_str = datetime.date.today().strftime("%G-W%V")
        days = load_week_trade_days(week_str)
        if today_str not in days and len(days) >= max_days:
            return True, (
                f"Weekly trading-day cap reached ({len(days)}/{max_days} days traded "
                f"this week). Cash is a position — sitting out until next week."
            )

        # (b) VIX floor.
        floor = float(rcfg.get('vix_no_trade_below', 0) or 0)
        if floor > 0:
            try:
                vix_tok = self.market_condition_identifier.vix_token
                data = await asyncio.to_thread(self.kite.ltp, str(vix_tok))
                vix = float((data or {}).get(str(vix_tok), {}).get('last_price', 0))
                if 0 < vix < floor:
                    return True, (
                        f"India VIX {vix:.2f} is below the {floor:.0f} floor — market too "
                        f"calm for option buyers to beat theta. Standing down today."
                    )
            except Exception as e:
                logging.debug(f"VIX-floor check skipped (non-fatal): {e}")

        return False, ""

    # ----- Strategy cooldown bookkeeping ----------------------------------

    def _currently_cooled(self) -> set:
        """
        Returns the set of strategy names whose cooldown is still active. Side
        effect: expired entries are removed from `_strategy_cooldown_until` and
        an info line is logged so the operator sees strategies re-entering the
        eligible pool.
        """
        now = datetime.datetime.now()
        expired = [
            name for name, until in self._strategy_cooldown_until.items()
            if until <= now
        ]
        for name in expired:
            del self._strategy_cooldown_until[name]
            logging.info(f"Cooldown expired: '{name}' is eligible again.")
        return set(self._strategy_cooldown_until.keys())

    def _cool_strategy(self, name: str) -> None:
        """Adds `name` to the cooldown set with an expiry time."""
        if not name:
            return
        cooldown_minutes = int(
            self.config.get("trading_flags", {}).get("strategy_cooldown_minutes", 60)
        )
        expiry = datetime.datetime.now() + datetime.timedelta(minutes=cooldown_minutes)
        self._strategy_cooldown_until[name] = expiry
        logging.warning(
            f"Cooling down '{name}' until {expiry.strftime('%H:%M:%S')} "
            f"({cooldown_minutes}-minute cooldown). Currently cooled: "
            f"{sorted(self._strategy_cooldown_until.keys())}"
        )

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
                logging.debug(
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

    def _is_consecutive_loss_breached(self) -> bool:
        """
        True once the intra-session consecutive-loss streak reaches the
        configured limit. Resets to False after a winning trade.

        Reads from config.risk_management:
          enable_consecutive_loss_limit: true   (default true)
          max_consecutive_losses: 3             (default 3)
        """
        rm = self.config.get('risk_management', {})
        if not rm.get('enable_consecutive_loss_limit', True):
            return False
        max_streak = int(rm.get('max_consecutive_losses', 3))
        if self.consecutive_losses >= max_streak:
            logging.error(
                f"CONSECUTIVE LOSS LIMIT: {self.consecutive_losses} losses in a row "
                f"(limit={max_streak}). Halting new entries for the session."
            )
            return True
        return False

    async def _is_weekly_loss_breached(self) -> bool:
        """
        True when the cumulative realized P&L since Monday of the current
        ISO week hits the configured weekly-loss cap.

        Reads from config.risk_management:
          enable_weekly_loss_limit: true         (default false — opt-in)
          max_weekly_loss_percent: 5.0           (% of starting capital)
        """
        rm = self.config.get('risk_management', {})
        if not rm.get('enable_weekly_loss_limit', False):
            return False
        if not self.starting_capital or self.starting_capital <= 0:
            return False
        max_loss_pct = float(rm.get('max_weekly_loss_percent', 5.0))
        max_loss_amt = self.starting_capital * (max_loss_pct / 100.0)
        if self.realized_pnl_week <= -abs(max_loss_amt):
            logging.error(
                f"WEEKLY LOSS LIMIT BREACHED: realized={self.realized_pnl_week:,.2f} "
                f"limit={-max_loss_amt:,.2f}. Halting new entries for the week."
            )
            return True
        return False

    def _is_momentum_too_low(self, df) -> bool:
        """Return True when recent bars are too slow for options buying to be viable.

        Checks the rolling average ATR over the last `min_atr_lookback_bars` completed
        bars against `min_atr_per_bar`. A threshold of 0 (the default) disables the gate.
        """
        flags = self.config.get('trading_flags') or {}
        threshold = float(flags.get('min_atr_per_bar', 0) or 0)
        if threshold <= 0:
            return False
        try:
            lookback = int(flags.get('min_atr_lookback_bars', 3) or 3)
            if df is None or len(df) < lookback + 1:
                return False
            # Use completed bars only (exclude the forming bar at iloc[-1]).
            atr_vals = df['atr'].iloc[-(lookback + 1):-1]
            if atr_vals.isna().all():
                return False
            avg_atr = float(atr_vals.mean())
            if avg_atr < threshold:
                logging.warning(
                    f"ATR momentum gate: avg ATR over last {lookback} bars = {avg_atr:.2f} "
                    f"< min {threshold:.2f}. Market too slow for options buying — skipping entry."
                )
                return True
        except Exception as e:
            logging.debug(f"ATR momentum gate check failed (non-fatal): {e}")
        return False

    # ------------------------------------------------------------------
    # Professional entry gates (added by pro-strategy session)
    # ------------------------------------------------------------------

    async def _is_daily_profit_target_hit(self) -> bool:
        """Halt new entries when daily profit target is reached.
        Guards against giving back a good morning in a slow afternoon."""
        flags = self.config.get('trading_flags', {})
        target_pct = float(flags.get('max_daily_profit_percent', 0) or 0)
        if target_pct <= 0:
            return False
        if not self.starting_capital or self.starting_capital <= 0:
            return False
        target_amt = self.starting_capital * (target_pct / 100.0)
        if self.realized_pnl_today >= target_amt:
            logging.info(
                f"DAILY PROFIT TARGET HIT: realized={self.realized_pnl_today:,.2f} "
                f"target={target_amt:,.2f} ({target_pct}%). Stopping new entries."
            )
            return True
        return False

    def _classify_day_quality(self, df) -> str:
        """
        Classifies the intraday environment as TRENDING, RANGE, CHOPPY, or UNKNOWN.

        A professional sits out RANGE and CHOPPY days — options buying bleeds
        theta on every losing small move. Only TRENDING days have the directional
        persistence that makes long-options strategies profitable.

        Returns: 'TRENDING' | 'RANGE' | 'CHOPPY' | 'UNKNOWN'
        """
        flags = self.config.get('trading_flags', {})
        if not flags.get('enable_day_quality_filter', True):
            return 'TRENDING'  # filter disabled → always allow

        if df is None or len(df) < 8:
            return 'UNKNOWN'

        try:
            # ADX — primary trend-strength indicator
            adx = 0.0
            if 'adx' in df.columns:
                adx_val = df.iloc[-1].get('adx')
                adx = float(adx_val) if adx_val is not None and not pd.isna(adx_val) else 0.0

            # First-30-min range (first 6 bars × 5 min = 30 min from 9:15)
            today = datetime.date.today()
            if hasattr(df.index, 'date'):
                today_bars = df[df.index.date == today]
            else:
                today_bars = df.tail(30)

            first_30_range_pct = 0.0
            if len(today_bars) >= 6:
                h = float(today_bars.iloc[:6]['high'].max())
                l = float(today_bars.iloc[:6]['low'].min())
                o = float(today_bars.iloc[0]['open'])
                if o > 0:
                    first_30_range_pct = (h - l) / o * 100.0

            # Direction-change count in last 12 bars (1 hour)
            closes = df['close'].tail(13).values
            direction_changes = sum(
                1 for i in range(1, len(closes) - 1)
                if (closes[i] > closes[i - 1]) != (closes[i + 1] > closes[i])
            )

            # Classification.
            # ADX overrides direction-changes so a genuinely trending day
            # (institutional flow, strong impulse) isn't blocked just because
            # price oscillated during a consolidation phase.
            if adx > 20:
                quality = 'TRENDING'
            elif adx > 25 and first_30_range_pct > 0.3:
                quality = 'TRENDING'
            elif direction_changes >= 7 and adx < 20:
                # Only CHOPPY when trend strength is also weak — prevents
                # blocking a trending session with micro-oscillations.
                quality = 'CHOPPY'
            elif first_30_range_pct < 0.25 and adx < 18:
                quality = 'RANGE'
            else:
                quality = 'RANGE'

            logging.debug(
                f"[DayQuality] {quality} — ADX={adx:.1f} "
                f"first30_range={first_30_range_pct:.2f}% "
                f"direction_changes={direction_changes}"
            )
            return quality
        except Exception as e:
            logging.debug(f"Day quality classification failed (non-fatal): {e}")
            return 'UNKNOWN'

    def _is_false_breakout(self, signal: str, df) -> bool:
        """
        Detects trap breakouts: signal fires on bar N but bar N+1 immediately
        reverses back inside the prior bar's range — a classic stop-hunt pattern.

        BUY trap : breakout bar closed above prior high but current bar closed back below it.
        SELL trap: breakdown bar closed below prior low but current bar closed back above it.
        """
        flags = self.config.get('trading_flags', {})
        if not flags.get('enable_trap_detection', True):
            return False
        if df is None or len(df) < 4:
            return False
        try:
            prev_bar  = df.iloc[-3]
            break_bar = df.iloc[-2]
            curr_bar  = df.iloc[-1]
            if signal == 'BUY':
                if break_bar['close'] <= prev_bar['high']:
                    return False  # wasn't a breakout bar
                if curr_bar['close'] < prev_bar['high']:
                    logging.warning(
                        f"[TrapDetect] BUY trap — price reversed back below "
                        f"prior high {prev_bar['high']:.2f}. Skipping entry."
                    )
                    return True
            elif signal == 'SELL':
                if break_bar['close'] >= prev_bar['low']:
                    return False
                if curr_bar['close'] > prev_bar['low']:
                    logging.warning(
                        f"[TrapDetect] SELL trap — price reversed back above "
                        f"prior low {prev_bar['low']:.2f}. Skipping entry."
                    )
                    return True
        except Exception as e:
            logging.debug(f"Trap detection failed (non-fatal): {e}")
        return False

    # Strategies that work WITH a trend — unsuitable for range/mean-reversion days.
    _TREND_FOLLOWING_STRATEGIES = frozenset({
        'EMA_Cross_RSI', 'Supertrend_MACD', 'Breakout_Prev_Day_HL',
        'Opening_Range_Breakout', 'BB_Squeeze_Breakout', 'NR7_Compression',
        'Volume_Spread_Analysis', 'Gemini_Default', 'Expiry_Momentum_Scalp',
    })
    # Preferred order for range-day strategy selection (first available wins).
    _RANGE_SCALP_STRATEGIES = ('VWAP_Reversion', 'Reversal_Detector', 'RSI_Divergence')
    # Mean-reversion / counter-trend strategies. These trade AGAINST the short-
    # term move by design, so the 15-min trend-confirmation gate is skipped for
    # them — a higher-timeframe trend filter would reject every reversion entry.
    _MEAN_REVERSION_STRATEGIES = frozenset({
        'VWAP_Reversion', 'RSI_Divergence', 'Reversal_Detector',
        'Volatility_Cluster_Reversal',
    })

    def _enter_range_scalp_mode(self, day_df) -> bool:
        """
        Switches the bot into range-scalp mode for the current bar cycle.

        Actions:
          • Injects _scalp_* override keys into trading_flags so agents use
            tighter profit targets and a tighter trailing stop.
          • If the currently active strategy is trend-following, rotates to the
            first available range strategy (VWAP_Reversion preferred, then
            Reversal_Detector, then RSI_Divergence).
          • Applies a size haircut (range_scalp.size_factor, default 0.5) on top
            of whatever progressive-loss / time-of-day multiplier is already set.

        Returns True if scalp mode was engaged (a range strategy is available),
        False if setup should be skipped this tick (no range strategy available).
        """
        scalp_cfg = self.config.get('range_scalp') or {}
        if not scalp_cfg.get('enable', True):
            return False

        tf = self.config['trading_flags']

        # Inject tight scalp-specific overrides.
        tf['_scalp_mode']       = True
        tf['_scalp_trail_pct']  = float(scalp_cfg.get('trail_pct', 10.0))
        tf['_scalp_t1_gain_pct'] = float(scalp_cfg.get('t1_gain_pct', 15.0))
        tf['_scalp_t2_gain_pct'] = float(scalp_cfg.get('t2_gain_pct', 25.0))

        # Strategy rotation: only rotate if the active strategy is trend-following.
        needs_rotation = (
            not self.active_strategy
            or self.active_strategy_name in self._TREND_FOLLOWING_STRATEGIES
        )
        if needs_rotation:
            allowed = [s for s in self._RANGE_SCALP_STRATEGIES
                       if s not in self._currently_cooled()
                       and s not in getattr(self, '_whitelist_excluded', set())]
            if not allowed:
                logging.warning(
                    "[RangeScalp] All range strategies are cooled — skipping scalp tick."
                )
                tf.pop('_scalp_mode', None)
                return False

            new_name = allowed[0]
            if new_name != self.active_strategy_name:
                from strategy_factory import get_strategy
                self.active_strategy = get_strategy(new_name, self.kite, self.config)
                self.active_strategy_name = new_name
                self._signals_seen_for_active_strategy = 0
                logging.info(
                    f"[RangeScalp] Rotated strategy → {new_name} "
                    f"(range day: VWAP / RSI extremes only)."
                )

        # Size haircut: reduce on top of the existing multiplier.
        size_factor = float(scalp_cfg.get('size_factor', 0.5))
        current_mult = float(self._trade_size_multiplier)
        tf['_effective_risk_pct_multiplier'] = current_mult * size_factor
        # Log once when scalp mode first activates; suppress on every subsequent tick.
        if not self._last_reported_scalp_state:
            logging.info(
                f"[RangeScalp] Scalp mode active — strategy={self.active_strategy_name}  "
                f"size={current_mult:.2f}×{size_factor:.2f}={current_mult*size_factor:.2f}  "
                f"T1/T2 +{tf['_scalp_t1_gain_pct']:.0f}%/+{tf['_scalp_t2_gain_pct']:.0f}%  "
                f"trail {tf['_scalp_trail_pct']:.0f}%"
            )
            self._last_reported_scalp_state = True
        return True

    def _exit_range_scalp_mode(self):
        """Clears all scalp-mode overrides so the next trending trade uses
        the full config values."""
        tf = self.config['trading_flags']
        for k in ('_scalp_mode', '_scalp_trail_pct', '_scalp_t1_gain_pct', '_scalp_t2_gain_pct'):
            tf.pop(k, None)
        # Also clear the size override set by _enter_range_scalp_mode.
        tf.pop('_effective_risk_pct_multiplier', None)
        self._last_reported_scalp_state = False  # reset so next entry logs once

    def _time_of_day_size_factor(self) -> float:
        """
        Returns a position-size multiplier based on the current time of day.

        Prime windows (institutional flow, second impulse) → 1.0 (full size).
        Degraded windows (consolidation, lunch) → 0.5-0.75 (reduced size).
        After entry_cutoff_time → 0.0 (no new entries; additional safety layer
        on top of the explicit cutoff check in the trading loop).
        """
        flags = self.config.get('trading_flags', {})
        if not flags.get('enable_time_of_day_sizing', True):
            return 1.0

        # Read cutoff from config so it stays in sync with entry_cutoff_time.
        cutoff = self._parse_hhmm(flags.get('entry_cutoff_time', '13:30')) or datetime.time(13, 30)
        now = datetime.datetime.now().time()
        if now >= cutoff:
            return 0.0   # block — hard cutoff
        elif datetime.time(9, 30) <= now < datetime.time(10, 15):
            return 1.0   # prime: institutional open order flow
        elif datetime.time(11, 30) <= now < datetime.time(12, 30):
            return 1.0   # prime: second-impulse window
        elif datetime.time(10, 15) <= now < datetime.time(11, 30):
            return 0.75  # moderate: post-open consolidation
        elif datetime.time(12, 30) <= now < cutoff:
            return 0.50  # degraded: lunch drift
        return 0.75      # pre-open edge case

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

        # POST_EVENT regime: wait out the opening chaos / IV crush before entering.
        rnb = getattr(self, "_regime_entry_not_before", None)
        if rnb and now < rnb:
            return f"post-event wait until {rnb.strftime('%H:%M')} ({self.day_regime})"
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
                            signed_gap_pct = (ltp - prev_close) / prev_close * 100.0
                            # Stash signed gap for the strategy selector (Layer 2).
                            self.open_gap_pct = signed_gap_pct
                            gap_pct = abs(signed_gap_pct)
                            if gap_pct >= threshold:
                                chosen = early_start
                                reason = (f"open-gap {signed_gap_pct:+.2f}% >= threshold {threshold}% "
                                          f"(prev_close={prev_close:.2f}, ltp={ltp:.2f})")
                except Exception as e:
                    logging.debug(f"Open-gap check failed (using default start): {e}")

        self.effective_entry_start_time = chosen
        if chosen == early_start and reason:
            logging.info(f"Early-entry override active: {reason}. Allowing entries from 09:15.")
        else:
            logging.debug(f"Effective entry start: {chosen.strftime('%H:%M')} (default).")
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
        logging.debug("--- Running Bot Setup & Strategy Assessment ---")

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
            # Stash for the loss-analyzer (needs the regime context at exit time).
            self.todays_conditions = todays_conditions

            # ── Regime governor (playbook: selectivity + survival) ───────────
            # Sit out the whole day when (a) the weekly trading-day allotment is
            # used up, or (b) VIX is below the floor (too calm for option buyers
            # to overcome theta). Runs early so we don't burn API calls on a day
            # we won't trade. Returning False makes the bot stand down for today.
            _sit_out, _sit_reason = await self._regime_governor()
            if _sit_out:
                self.no_trade_reason = _sit_reason
                logging.warning(f"Regime governor: standing down today — {_sit_reason}")
                return False

            # 1b. Trigger the YouTube sentiment fetch BEFORE we look at sentiment.
            # Cached for the day after the first call, so reassessment runs are
            # cheap (single cache read, no LLM calls). Best-effort: failures here
            # don't block setup — sentiment will just fall back to news-only.
            # When the price-action trend module drives direction (market_trend.
            # enable: true), news/YouTube sentiment is NOT used for the verdict —
            # so skip the slow, Gemini-dependent YouTube fetch entirely. This
            # avoids burning Gemini free-tier quota and the long 429 retry waits
            # that were stalling startup ~75s with no benefit.
            _trend_drives = bool((self.config.get('market_trend') or {}).get('enable', False))
            if (not _trend_drives) and self.youtube_agent and not self.youtube_agent.is_ready():
                try:
                    await self.youtube_agent.fetch_today()
                except Exception as e:
                    logging.warning(f"YouTube sentiment fetch failed (non-fatal): {e}")

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

                # Directional bias: when the price-action trend module is on, it
                # produces the CE/PE verdict (global cues + gap + GIFT + price
                # action) with its own 20s override. Otherwise fall back to the
                # news/YouTube sentiment flow. Both honour the same hard config/env
                # overrides and set self.day_sentiment on the same scale.
                if (self.config.get('market_trend') or {}).get('enable', False):
                    self.day_sentiment = await self._resolve_market_trend()
                else:
                    self.day_sentiment = await self._resolve_sentiment()
                self._cached_sentiment = self.day_sentiment
                await self._snapshot_sentiment_context()
                # Print the mid-session override hint once per session so
                # the operator knows they can change sentiment at any time.
                if not self._override_hint_shown:
                    self._print_override_hint()
                    self._override_hint_shown = True
            else:
                logging.debug(
                    f"Reusing cached sentiment '{self._cached_sentiment}' — "
                    f"no material market shift since last capture."
                )
                self.day_sentiment = self._cached_sentiment

            if self.day_sentiment == "Neutral":
                self.no_trade_reason = "Market sentiment is Neutral. No new entries today."
                return False

            logging.debug(f"Today's Market Conditions: {todays_conditions} | Final Sentiment: {self.day_sentiment}")

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
                        logging.debug(f"Sufficient historical data found ({trading_days} days). Activating RAG.")
                        rag_context = self.rag_service.retrieve_context_for_strategy_selection(todays_conditions)
                    else:
                        logging.debug(f"RAG disabled: Insufficient data. Found {trading_days}, need {rag_min_days}.")
                else:
                    logging.debug("RAG disabled: No trade log found.")
            else:
                logging.debug("RAG is disabled in config.yaml.")
            
            # 5. Select Strategy — pass full context so the deterministic cascade
            # can apply hard pins / open-gap override / indicator overrides /
            # regime table without ever needing the LLM. Missing data simply
            # skips the relevant cascade layer.
            is_expiry_day_now = False
            try:
                if self.order_agent is not None:
                    is_expiry_day_now = self.order_agent.is_weekly_expiry_today()
            except Exception:
                pass

            bars_for_selector = None
            try:
                # Bars are cached intraday — cheap to ask for again. None pre-open.
                if self.is_market_open() and self.order_agent is not None:
                    bars_for_selector = await self._get_underlying_bars()
            except Exception as e:
                logging.debug(f"Bars-for-selector fetch failed (non-fatal): {e}")

            # ── Regime classification (playbook Part 3 — the mandatory first step) ──
            # Assign one of the 5 regimes + the buy-only strategy family that fits
            # it. A clean regime CONSTRAINS the selector to its family; an UNCLEAR
            # (or pre-event, under buy-only) regime stands the bot down for the day
            # on the FIRST setup. Intraday reassessments only relax/constrain the
            # family — the loop's CHOPPY-skip handles going unclear mid-session.
            _is_first_setup = self._first_setup
            self._first_setup = False
            self._regime_excluded = set()
            self._regime_allowed = ()
            self._regime_entry_not_before = None
            if (self.config.get('regime') or {}).get('enable', True) \
                    and bars_for_selector is not None and not bars_for_selector.empty:
                try:
                    _dq = self._classify_day_quality(bars_for_selector)
                    _vix = await self._current_vix()
                    self._today_vix = _vix
                    # Best-effort FII positioning (informational; never gates).
                    try:
                        self._fii_bias = await asyncio.to_thread(fetch_fii_bias, self.config)
                        if self._fii_bias:
                            logging.info(
                                f"[FII] {self._fii_bias.get('bias')} "
                                f"(net={self._fii_bias.get('net_index_futures')}, "
                                f"src={self._fii_bias.get('source')})"
                            )
                    except Exception:
                        self._fii_bias = None
                    _evt_today = next(
                        (str(c) for c in todays_conditions if str(c).startswith('EVENT_')), None
                    )
                    _rgm = self.regime_classifier.classify(
                        day_quality=_dq, vix=_vix,
                        is_expiry_day=is_expiry_day_now, dte=None,
                        event_today=_evt_today,
                        days_to_next_event=self._days_to_next_event(),
                        direction_hint=self.day_sentiment, gap_pct=self.open_gap_pct,
                    )
                    self.day_regime = _rgm.regime
                    self._regime_reason = _rgm.reason
                    logging.info(f"[Regime] {_rgm.regime} — {_rgm.reason}")
                    if _rgm.sit_out:
                        if _is_first_setup:
                            self.no_trade_reason = f"Regime {_rgm.regime}: {_rgm.reason}"
                            return False
                        # Intraday: don't kill the session; leave the cascade free
                        # and let the loop's day-quality skip manage it.
                    else:
                        self._regime_excluded = _rgm.excluded()
                        self._regime_allowed = _rgm.allowed_strategies
                        if _rgm.entry_not_before:
                            self._regime_entry_not_before = self._parse_hhmm(_rgm.entry_not_before)
                except Exception as _rgm_exc:
                    logging.debug(f"Regime classification skipped (non-fatal): {_rgm_exc}")

            # ── Strategy whitelist (the cull) ─────────────────────────────────
            # Only strategies with demonstrated (or being-evaluated) edge may
            # trade. Everything else — including chronic losers like the old
            # Gemini_Default fallback — is excluded from the selector entirely.
            _wl = list((self.config.get('strategy_selector') or {}).get('whitelist') or [])
            self._whitelist_excluded = (set(ALL_STRATEGIES) - set(_wl)) if _wl else set()
            if self._whitelist_excluded:
                logging.debug(
                    f"[Whitelist] Active universe: {sorted(set(ALL_STRATEGIES) - self._whitelist_excluded)}"
                )

            # ── Expiry 0-DTE gamma scalp (advanced) ──────────────────────────
            # On expiry day, when enabled, trade the expiring contract's gamma:
            # half size, flat by hard_exit_time. _get_trade_details picks the
            # 0-DTE option. (Entry window decided below, after conviction.)
            self._expiry_gamma_active = False
            if getattr(self, 'day_regime', '') == "EXPIRY":
                egs = (self.config.get("expiry_gamma_scalp") or {})
                if egs.get("enable", False):
                    self._expiry_gamma_active = True
                    logging.info(
                        f"[ExpiryGamma] ACTIVE — 0-DTE scalp, half size, "
                        f"flat by {egs.get('hard_exit_time','14:30')}."
                    )

            # ── Conviction score (regime + sentiment + global cues) ──────────
            # Drives position size and the hold-the-winner exit: HIGH conviction
            # trending trades size up and ride to ~15:15; weak ones size down.
            self.day_conviction, self.day_conviction_level = self._compute_conviction()
            logging.info(
                f"[Conviction] {self.day_conviction_level} ({self.day_conviction:.2f}) "
                f"— regime={getattr(self,'day_regime','?')}, sentiment={self.day_sentiment}, "
                f"trend_score={self._trend_composite:+.2f}"
            )

            # ── Expiry-gamma entry window (conviction-gated) ─────────────────
            # A HIGH-conviction gap-and-go may enter EARLY (from start_after,
            # ≥09:45 — never the chaotic first 15 min) to catch the morning trend.
            # A murky open waits for confirmed_after (11:00) to let the fakeout/
            # stop-hunt phase resolve. Early entries still pass every confirmation
            # gate (ORB close + volume, 15m, trap, momentum), so a fake move
            # won't trigger one.
            if self._expiry_gamma_active:
                egs = (self.config.get("expiry_gamma_scalp") or {})
                _early = self._parse_hhmm(str(egs.get("start_after", "09:45")))
                _confirmed = self._parse_hhmm(str(egs.get("confirmed_after", "11:00")))
                if self.day_conviction_level == "HIGH":
                    self._regime_entry_not_before = _early
                    logging.info(
                        f"[ExpiryGamma] HIGH conviction gap-and-go → EARLY entries "
                        f"from {egs.get('start_after','09:45')} to catch the morning trend."
                    )
                else:
                    self._regime_entry_not_before = _confirmed
                    logging.info(
                        f"[ExpiryGamma] Conviction {self.day_conviction_level} → wait until "
                        f"{egs.get('confirmed_after','11:00')} (let the open settle)."
                    )

            # ── Strategy selection ──────────────────────────────────────────
            # Manual mode: operator picks from a numbered menu. The choice is
            # cached in _manual_strategy_name and reused on every reassessment
            # so the bot doesn't keep asking. The operator can re-pick at any
            # time by typing 'm' + Enter while the bot is awaiting signals.
            #
            # Auto mode: 5-layer deterministic cascade (LangGraphAgent).
            if self._manual_mode:
                if self._manual_strategy_name is None:
                    # First run — must pick now.
                    chosen = await self._prompt_strategy_picker()
                    if chosen:
                        self._manual_strategy_name = chosen
                        logging.info(f"[ManualMode] Operator chose strategy: {chosen}")
                    else:
                        # Timed out or bad input on first pick — fall back to auto.
                        logging.warning(
                            "[ManualMode] No strategy chosen at startup; "
                            "falling back to auto-selector for this cycle."
                        )
                best_strategy_name = self._manual_strategy_name  # may still be None → auto below

            else:
                best_strategy_name = None  # will be filled by auto-selector below

            # Auto-selector path (used in auto mode, or as fallback in manual mode).
            if best_strategy_name is None:
                best_strategy_name = await self.langgraph_agent.get_recommended_strategy(
                    market_conditions=todays_conditions,
                    sentiment=self.day_sentiment,
                    is_expiry_day=is_expiry_day_now,
                    open_gap_pct=self.open_gap_pct,
                    underlying_bars=bars_for_selector,
                    exclude_strategies=self._currently_cooled() | self._regime_excluded | self._whitelist_excluded,
                    user_prompt=user_prompt,
                    rag_context=rag_context,
                )
                # Safety valve: every cascade layer excluded by active cooldowns.
                if not best_strategy_name and self._strategy_cooldown_until:
                    cleared = sorted(self._strategy_cooldown_until.keys())
                    logging.warning(
                        f"All strategies currently cooled ({cleared}). Clearing "
                        f"cooldowns and retrying the cascade with a clean slate."
                    )
                    self._strategy_cooldown_until.clear()
                    best_strategy_name = await self.langgraph_agent.get_recommended_strategy(
                        market_conditions=todays_conditions,
                        sentiment=self.day_sentiment,
                        is_expiry_day=is_expiry_day_now,
                        open_gap_pct=self.open_gap_pct,
                        underlying_bars=bars_for_selector,
                        exclude_strategies=set(self._regime_excluded) | self._whitelist_excluded,  # keep regime+whitelist, drop cooldowns
                        user_prompt=user_prompt,
                        rag_context=rag_context,
                    )
                # Default-to-trade fallback: if the selector came back empty,
                # use the first regime-preferred strategy that's also on the
                # whitelist; failing that, the first whitelisted strategy. The
                # bot should produce an attempt on a clean day, not sit idle
                # because every layer excluded everything.
                if not best_strategy_name:
                    _fallback = (
                        [s for s in (self._regime_allowed or ()) if s not in self._whitelist_excluded]
                        or [s for s in ALL_STRATEGIES if s not in self._whitelist_excluded]
                    )
                    if _fallback:
                        best_strategy_name = _fallback[0]
                        logging.info(
                            f"[Selector] Empty after exclusions — defaulting to "
                            f"{best_strategy_name} (regime/whitelist fallback)."
                        )

            if not best_strategy_name:
                self.no_trade_reason = "Selector returned no strategy."
                logging.error(self.no_trade_reason)
                return False
            # Reset the per-strategy signal counter when the strategy changes.
            if best_strategy_name != self.active_strategy_name:
                self._signals_seen_for_active_strategy = 0
                logging.info(
                    f"Strategy changed: '{self.active_strategy_name}' -> "
                    f"'{best_strategy_name}'."
                )
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
                        logging.debug("CPR pivots calculated for the day.")
                    else:
                        logging.debug(
                            "No prior-day data; CPR-dependent strategies will skip."
                        )
                else:
                    logging.warning("Empty daily history; CPR pivots unavailable.")
            except Exception as e:
                logging.error(
                    f"CPR pivot fetch failed (non-fatal): {e}. "
                    f"Continuing without CPR — strategies that need pivots will skip."
                )
            
            # --- Trading mode (MODERATE / AGGRESSIVE) ---
            # Evaluated each setup() so the mode can step down after consecutive
            # losses even mid-session (strategy reassessment triggers setup again).
            _vix_for_mode = 0.0
            try:
                _vix_tok = self.market_condition_identifier.vix_token
                _vix_data = await asyncio.to_thread(self.kite.ltp, str(_vix_tok))
                _vix_for_mode = float((_vix_data or {}).get(str(_vix_tok), {}).get('last_price', 0))
            except Exception:
                pass
            self._trading_mode = self.rag_service.get_trading_mode(vix_value=_vix_for_mode)
            mode_cfg = self.config.get('mode_switching') or {}
            tf = self.config['trading_flags']

            if self._trading_mode == 'AGGRESSIVE':
                eff_risk = float(mode_cfg.get('aggressive_risk_percent', 2.0))
                tf['_agg_max_trades']   = int(mode_cfg.get('aggressive_max_trades', 5))
                tf['_agg_trail_pct']    = float(mode_cfg.get('aggressive_trail_pct', 20.0))
                tf['_agg_t1_gain_pct']  = float(mode_cfg.get('aggressive_t1_gain_pct', 40.0))
                tf['_agg_t2_gain_pct']  = float(mode_cfg.get('aggressive_t2_gain_pct', 80.0))
            else:
                eff_risk = float(tf.get('risk_per_trade_percent', 1.0))
                for _k in ('_agg_max_trades', '_agg_trail_pct', '_agg_t1_gain_pct', '_agg_t2_gain_pct'):
                    tf.pop(_k, None)

            # Written into trading_flags — agents.py reads _effective_risk_pct from flags.
            tf['_effective_risk_pct'] = eff_risk

            self.bot_state = "AWAITING_SIGNAL"
            self.awaiting_signal_since = datetime.datetime.now()

            # ── Setup-complete banner (replaces the wall of individual INFO logs) ──
            _conds_str  = ', '.join(sorted(todays_conditions)) if todays_conditions else '—'
            _cap        = self.starting_capital or 0
            _max_t      = tf.get('_agg_max_trades') or tf.get('max_trades_per_day', 3)
            _trail      = tf.get('_agg_trail_pct') or self.config.get('trailing_stop_loss', {}).get('percentage', 15.0)
            _t1         = tf.get('_agg_t1_gain_pct') or self.config.get('partial_exits', {}).get('t1_gain_pct', 30)
            _t2         = tf.get('_agg_t2_gain_pct') or self.config.get('partial_exits', {}).get('t2_gain_pct', 60)
            _entry_s    = (self.effective_entry_start_time or datetime.time(9, 30)).strftime('%H:%M')
            _is_re      = self.starting_capital is not None  # True on reassessment runs
            _banner_hdr = "RE-ASSESSMENT" if _is_re else "SETUP COMPLETE"
            self._print_event([
                f"{_banner_hdr}  {datetime.datetime.now().strftime('%H:%M:%S')}",
                f"  Strategy   : {self.active_strategy.name}",
                f"  Sentiment  : {self.day_sentiment}  │  Conditions: {_conds_str}",
                f"  Mode       : {self._trading_mode}  │  Risk/trade: {eff_risk:.1f}%  │  Max trades: {_max_t}",
                f"  Targets    : T1 +{_t1:.0f}%  T2 +{_t2:.0f}%  Trail {_trail:.0f}%",
                f"  Capital    : ₹{_cap:,.0f}  │  Entry from {_entry_s}  │  Cutoff 15:20",
            ], level="info")

            # Fix: if setup completed but we're already past the entry cutoff,
            # go straight to STOPPED — no point waiting in AWAITING_SIGNAL all day.
            _now_time = datetime.datetime.now().time()
            _cutoff_dt = self._parse_hhmm(self.config['trading_flags'].get('entry_cutoff_time', '15:20'))
            if _cutoff_dt and _now_time > _cutoff_dt:
                logging.warning(
                    f"Setup completed after entry cutoff "
                    f"({_now_time.strftime('%H:%M')} > {_cutoff_dt.strftime('%H:%M')}). "
                    f"No trades possible today — bot will stop."
                )
                self.bot_state = "STOPPED"
                return False

            return True
        except Exception as e:
            logging.error(f"Setup failed: {e}", exc_info=True)
            self.no_trade_reason = str(e)
            return False

    @staticmethod
    def _is_interactive_tty() -> bool:
        """True only if stdin is connected to a real terminal."""
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  Terminal UI helpers — live status line + clean event banners        #
    # ------------------------------------------------------------------ #

    def _terminal_width(self) -> int:
        try:
            import shutil
            return shutil.get_terminal_size(fallback=(120, 24)).columns - 1
        except Exception:
            return 119

    def _clear_status_line(self) -> None:
        """Erase the current status line so a log message can print cleanly."""
        if not self._is_interactive_tty():
            return
        print(f"\r{' ' * self._terminal_width()}\r", end="", flush=True)

    def _print_status_line(self) -> None:
        """
        Overwrite the current terminal line with a rich live-status ticker.

        AWAITING_SIGNAL:
          ⏳ HH:MM:SS  AWAITING │ Strategy │ MODE │ Sentiment │ DayQuality │ P&L │ ⏱ mm:ss │ hold: …

        IN_POSITION:
          📈 HH:MM:SS  IN TRADE │ Symbol │ entry ₹X │ trail ₹Y │ P&L today ₹Z

        STOPPED / SETUP:
          ⏹  HH:MM:SS  STOPPED / ⚙ SETTING UP

        Only active on interactive TTY; log files and headless runs are unaffected.
        """
        if not self._is_interactive_tty():
            return

        now = datetime.datetime.now()
        ts  = now.strftime('%H:%M:%S')
        w   = self._terminal_width()

        state = self.bot_state

        if state == "AWAITING_SIGNAL":
            elapsed = "00:00"
            if self.awaiting_signal_since:
                secs    = max(0, int((now - self.awaiting_signal_since).total_seconds()))
                elapsed = f"{secs // 60:02d}:{secs % 60:02d}"
            strategy = self.active_strategy_name or "—"
            manual_tag = "[M] " if getattr(self, '_manual_mode', False) else ""
            mode     = getattr(self, '_trading_mode', 'MODERATE')
            sent     = (getattr(self, 'day_sentiment', '') or '—')[:10]
            dq       = getattr(self, '_day_quality', '') or ''
            dq_part  = f" │ {dq}" if dq and dq not in ('UNKNOWN', 'TRENDING') else ""
            pnl      = getattr(self, 'realized_pnl_today', 0.0) or 0.0
            pnl_str  = f"+₹{pnl:,.0f}" if pnl >= 0 else f"-₹{abs(pnl):,.0f}"
            trades   = getattr(self, 'trades_today_count', 0)
            hold_reason = ""
            if self.active_strategy and getattr(self.active_strategy, '_last_hold_reason', ''):
                hold_reason = f" │ {self.active_strategy._last_hold_reason[:60]}"
            wl       = f"W:{self.wins_today} L:{self.losses_today}"
            line = (
                f"⏳ {ts}  AWAITING │ {manual_tag}{strategy} │ {mode} │ {sent}{dq_part}"
                f" │ {pnl_str} ({trades}T {wl}) │ ⏱ {elapsed}{hold_reason}"
            )

        elif state == "IN_POSITION":
            trade   = (getattr(self, 'position_agent', None) and
                       self.position_agent.active_trade) or {}
            symbol  = trade.get('symbol', '—')
            entry   = trade.get('entry_price', 0)
            trail   = trade.get('trailing_stop_loss', 0)
            pnl     = getattr(self, 'realized_pnl_today', 0.0) or 0.0
            pnl_str = f"+₹{pnl:,.0f}" if pnl >= 0 else f"-₹{abs(pnl):,.0f}"
            entry_s = f"entry ₹{entry:.2f}" if entry else ""
            trail_s = f" │ trail ₹{trail:.2f}" if trail else ""
            wl      = f"W:{self.wins_today} L:{self.losses_today}"
            line = (
                f"📈 {ts}  IN TRADE │ {symbol} │ {entry_s}{trail_s}"
                f" │ P&L today {pnl_str} ({wl})"
            )

        elif state == "SETUP":
            line = f"⚙  {ts}  SETTING UP  …"

        elif state == "STOPPED":
            pnl      = getattr(self, 'realized_pnl_today', 0.0) or 0.0
            pnl_str  = f"+₹{pnl:,.0f}" if pnl >= 0 else f"-₹{abs(pnl):,.0f}"
            bal, lbl = self._current_balance()
            placed   = getattr(self, 'trades_today_count', 0)
            line     = (
                f"⏹  {ts}  STOPPED │ Net {pnl_str} │ {placed}T placed "
                f"(W:{self.wins_today} L:{self.losses_today}) │ Bal ₹{bal:,.0f} ({lbl})"
            )

        else:
            line = f"   {ts}  {state}"

        print(f"\r{line[:w].ljust(w)}", end="", flush=True)

    def _print_event(self, lines: list, level: str = "info") -> None:
        """
        Print a clean bordered event banner to the terminal, compatible with
        the live status line. The status line is cleared before and redrawn after.

        `level` controls the left-border char: info="─", warn="!", error="✗", trade="▶"
        """
        if not self._is_interactive_tty():
            return
        self._clear_status_line()
        border_char = {"info": "─", "warn": "!", "error": "✗", "trade": "▶"}.get(level, "─")
        w = min(self._terminal_width(), 76)
        sep = border_char * w
        print(sep)
        for ln in lines:
            print(f"  {ln}")
        print(sep, flush=True)
        # Status line will be redrawn by the ticker on the next 1-second tick.

    async def _run_ticker(self) -> None:
        """Background coroutine: refreshes the status line every second.
        Paused automatically while operator input is in progress so keystrokes
        aren't overwritten by the live status line."""
        try:
            while True:
                if not self._ticker_paused:
                    self._print_status_line()
                # Write the machine-readable status snapshot for the dashboard
                # every tick (works headless too — no TTY required).
                self._write_status_snapshot()
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            self._clear_status_line()
            raise

    def _write_status_snapshot(self) -> None:
        """
        Persist a compact, machine-readable status snapshot to
        state/bot_status.json so the FastAPI dashboard (dashboard.py) can show
        live status, P&L, the active trade and the 'why no trade' rationale —
        without touching the trading process. Best-effort; never raises.
        """
        try:
            active = (getattr(self, 'position_agent', None)
                      and self.position_agent.active_trade) or {}
            bal, bal_lbl = self._current_balance()
            tf = self.config.get('trading_flags', {}) or {}
            snap = {
                "ts":                 datetime.datetime.now().isoformat(),
                "state":              self.bot_state,
                "strategy":           self.active_strategy_name,
                "manual_mode":        bool(getattr(self, '_manual_mode', False)),
                "mode":               getattr(self, '_trading_mode', 'MODERATE'),
                "sentiment":          self.day_sentiment,
                "day_quality":        getattr(self, '_day_quality', 'UNKNOWN'),
                "regime":             getattr(self, 'day_regime', 'PENDING'),
                "regime_reason":      getattr(self, '_regime_reason', ''),
                "conviction":         getattr(self, 'day_conviction_level', 'LOW'),
                "conviction_score":   getattr(self, 'day_conviction', 0.0),
                "setups_scored":      getattr(self, '_scored_today', 0),
                "setups_passed":      getattr(self, '_score_passed_today', 0),
                "best_setup_score":   getattr(self, '_best_score_today', 0.0),
                "vix":                round(float(getattr(self, '_today_vix', 0) or 0), 2),
                "pcr":                (self._pcr_data or {}).get("pcr"),
                "max_pain":           (self._pcr_data or {}).get("max_pain"),
                "call_walls":         (self._pcr_data or {}).get("call_walls"),
                "put_walls":          (self._pcr_data or {}).get("put_walls"),
                "iv_percentile":      getattr(getattr(self, 'order_agent', None), '_last_iv_percentile', None),
                "fii_bias":           (self._fii_bias or {}).get("bias") if self._fii_bias else None,
                "conditions":         sorted(getattr(self, 'todays_conditions', set()) or []),
                "is_expiry_day":      bool(getattr(self, 'is_expiry_day', False)),
                "paper":              bool(tf.get('paper_trading', True)),
                "realized_pnl_today": round(float(self.realized_pnl_today or 0), 2),
                "wins":               self.wins_today,
                "losses":             self.losses_today,
                "scratch":            self.scratch_today,
                "trades_today_count": self.trades_today_count,
                "max_trades":         int(tf.get('max_trades_per_day', 3)),
                "balance":            round(float(bal or 0), 2),
                "balance_label":      bal_lbl,
                "active_trade": ({
                    "symbol": active.get("symbol"),
                    "type":   active.get("type"),
                    "entry":  active.get("entry_price"),
                    "trail":  active.get("trailing_stop_loss"),
                    "sl":     active.get("initial_stop_loss"),
                    "qty":    active.get("quantity"),
                } if active else None),
                "completed_trades":   self.completed_trades,
                "why_no_trade":       (self._no_trade_diagnosis()
                                       if self.trades_today_count == 0 else []),
                "no_trade_reason":    self.no_trade_reason,
            }
            atomic_write_json(state_path("bot_status.json"), snap)
        except Exception as e:
            logging.debug(f"status snapshot failed: {e}")

    @staticmethod
    def _drain_stdin() -> None:
        """
        Discard any buffered/pending stdin BEFORE showing a timed prompt.

        Without this, a stray keystroke the operator typed while the bot was busy
        (e.g. pressing Enter during the long YouTube/Gemini wait at startup) sits
        in the terminal buffer and is instantly consumed by the next prompt —
        making it look like the operator "accepted" with no chance to respond.
        Flushes the OS input queue on Unix (termios), with a non-blocking
        select() drain as a fallback.
        """
        try:
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            return
        except Exception:
            pass
        try:
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if not ready:
                    break
                if not sys.stdin.readline():
                    break
        except Exception:
            pass

    async def _input_with_timeout(self, prompt: str, timeout: float = 20.0):
        """
        Reads a line from stdin with a timeout. Returns the stripped input
        line, "" if the user just pressed Enter, or None if no input arrived
        within `timeout` seconds (the bot will then take its own decision).

        Pauses the status-line ticker while waiting so keystrokes aren't
        overwritten by the 1-second refresh.
        """
        if not self._is_interactive_tty():
            return None  # headless: caller falls back to its default

        self._ticker_paused = True
        self._clear_status_line()
        # Drop any stale buffered input so a keystroke typed during a prior wait
        # doesn't auto-dismiss this prompt before the operator can respond.
        self._drain_stdin()
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
        self._ticker_paused = False
        if line is None:
            print(f"\n  [no response in {int(timeout)}s — bot will take its own decision]")
            logging.info(f"Operator input timeout ({int(timeout)}s); bot using its own default.")
            return None
        return line.rstrip("\n").strip()

    # ------------------------------------------------------------------ #
    #  Mid-session sentiment override                                      #
    # ------------------------------------------------------------------ #
    _VALID_SENTIMENTS: set = {"Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish"}
    _SENTIMENT_BY_NORM: dict = {
        " ".join(s.split()).lower(): s
        for s in {"Very Bullish", "Bullish", "Neutral", "Bearish", "Very Bearish"}
    }

    async def _check_mid_session_override(self) -> "str | None":
        """
        Non-blocking check for a mid-session sentiment override. Called at the
        start of every AWAITING_SIGNAL iteration.

        Two sources (checked in order):
          1. File  — create  output/sentiment_override.txt  and write the
                     desired sentiment (e.g. 'Bullish'). The bot reads it,
                     applies it, and deletes the file.
          2. Stdin — while the bot is running, just type a sentiment and press
                     Enter in the same terminal. The loop's non-blocking select()
                     picks it up on the next iteration without interrupting the
                     status line. Type '?' + Enter to print valid options.

        Returns the canonical sentiment string when an override is pending,
        otherwise None. Caller is responsible for applying the new value.
        """
        # ── 1. File sentinel (works headless / remote) ──────────────────
        override_file = os.path.join("output", "sentiment_override.txt")
        if os.path.exists(override_file):
            try:
                raw = open(override_file).read().strip()
                try:
                    os.remove(override_file)
                except OSError:
                    pass
            except Exception as _fe:
                logging.debug(f"[SentimentOverride] Could not read override file: {_fe}")
                raw = ""
            if raw:
                canonical = self._SENTIMENT_BY_NORM.get(" ".join(raw.split()).lower())
                if canonical:
                    return canonical
                logging.warning(
                    f"[SentimentOverride] File contained invalid value '{raw}'. "
                    f"Valid: {sorted(self._VALID_SENTIMENTS)}"
                )

        # ── 2. Non-blocking stdin check (TTY only) ──────────────────────
        if not self._is_interactive_tty() or self._ticker_paused:
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
        except Exception:
            return None
        if not ready:
            return None

        # Input is waiting — pause the ticker so it doesn't overwrite what
        # the operator typed, then read the buffered line.
        self._ticker_paused = True
        self._clear_status_line()
        try:
            raw = sys.stdin.readline().strip()
        except Exception:
            raw = ""
        finally:
            self._ticker_paused = False

        if not raw:
            return None

        # '?' / 'help' → print current state and valid options.
        if raw.lower() in ("?", "help", "h"):
            valid_sorted = sorted(self._VALID_SENTIMENTS)
            strat = self.active_strategy_name or "—"
            mode_tag = " (manual)" if self._manual_mode else " (auto)"
            print(
                f"\n  Current sentiment : {self.day_sentiment}\n"
                f"  Current strategy  : {strat}{mode_tag}\n"
                f"\n  Sentiment values  : {valid_sorted}\n"
                f"  Stdin override    : type a sentiment value + Enter\n"
                f"  File override     : echo 'Bullish' > output/sentiment_override.txt\n"
                f"\n  Strategy re-pick  : type  m  + Enter\n"
            )
            return None

        # 'm' / 'manual' → strategy picker (works in both manual and auto mode).
        if raw.lower() in ("m", "manual", "strategy"):
            chosen = await self._prompt_strategy_picker()
            if chosen:
                old_name = self.active_strategy_name
                self._manual_strategy_name = chosen
                self.active_strategy_name = chosen
                self.active_strategy = get_strategy(chosen, self.kite, self.config)
                self._signals_seen_for_active_strategy = 0
                logging.info(f"[ManualMode] Strategy switched: {old_name} → {chosen}")
                self._print_event(
                    [
                        f"Strategy switched by operator:  {old_name}  →  {chosen}",
                        "Bot will use this strategy from the next signal evaluation.",
                    ],
                    level="warn",
                )
            return None  # not a sentiment value

        canonical = self._SENTIMENT_BY_NORM.get(" ".join(raw.split()).lower())
        if canonical:
            return canonical

        # Unknown input — rate-limit the warning to once per 30 s.
        import time as _time
        _now = _time.monotonic()
        if _now - self._last_stdin_override_warn > 30:
            self._last_stdin_override_warn = _now
            logging.warning(
                f"[Override] '{raw}' not recognised as sentiment or command. "
                f"Type '?' for help. Sentiment options: {sorted(self._VALID_SENTIMENTS)}"
            )
        return None

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
        # Weekly report on the configured day (default Friday).
        try:
            rep_cfg = (self.config.get('reporting') or {})
            if rep_cfg.get('weekly_report', True):
                want_day = str(rep_cfg.get('weekly_report_day', 'Friday')).strip().lower()
                today_day = datetime.date.today().strftime('%A').lower()
                if today_day == want_day:
                    send_weekly_report(self.config, datetime.date.today().strftime("%G-W%V"))
                    logging.info("Weekly report sent.")
        except Exception as e:
            logging.error(f"Failed to send weekly report: {e}", exc_info=True)

    def _handle_losing_trade(self, completed_trade: dict, underlying_df):
        """
        Builds a detailed deterministic post-mortem for a losing trade, prints
        it to the terminal, and emails it. Never raises into the trading loop —
        a reporting failure must not interrupt the bot.
        """
        try:
            report = build_loss_report(
                trade=completed_trade,
                underlying_df=underlying_df,
                market_conditions=self.todays_conditions,
                sentiment=self.day_sentiment,
            )
        except Exception as e:
            logging.error(f"Loss-analysis report build failed: {e}", exc_info=True)
            return

        # Terminal: print the full report so it's visible in the live log.
        print("\n" + report + "\n")
        logging.info(
            f"Loss post-mortem generated for {completed_trade.get('Symbol', '?')} "
            f"(P&L {completed_trade.get('ProfitLoss', 0):,.2f})."
        )

        # Email: best-effort, gated by email_settings.send_daily_report.
        try:
            send_loss_analysis_email(self.config, report, completed_trade)
        except Exception as e:
            logging.error(f"Loss-analysis email failed: {e}", exc_info=True)

    async def _resolve_market_trend(self):
        """
        Pre-market directional resolver (CE / PE / Neutral) driven by the
        MarketTrendAnalyzer: global market fluctuations, GIFT-Nifty implied gap,
        the NIFTY open gap, and recent price action. Mirrors _resolve_sentiment:

          1. Hard config/env override (daily_overrides.sentiment / DAILY_SENTIMENT)
             wins outright — used for unattended runs.
          2. Compute the automated price-action verdict and show its drivers.
          3. On a TTY, give the operator `operator_input_timeout_seconds` (20s
             default) to override; Enter or timeout accepts the automated verdict.
          4. Returns a sentiment string on the same scale day_sentiment uses
             (Very Bullish / Bullish / Neutral / Bearish / Very Bearish), so the
             rest of the bot (signal-direction gate → CE/PE) is unchanged.
        """
        valid = {"Very Bullish", "Bullish", "Bearish", "Very Bearish", "Neutral"}

        # 1. Hard override wins (unattended runs).
        hard_override = (
            (self.config.get("daily_overrides", {}) or {}).get("sentiment")
            or os.environ.get("DAILY_SENTIMENT")
        )
        if hard_override and hard_override in valid:
            logging.info(f"[MarketTrend] Hard-override from config/env: {hard_override}")
            return hard_override

        # 2. Automated price-action verdict.
        try:
            trend = await self.market_trend_analyzer.get_trend()
        except Exception as e:
            logging.warning(f"[MarketTrend] analysis failed ({e}); defaulting to Neutral.")
            trend = {"verdict": "NEUTRAL", "sentiment": "Neutral",
                     "composite": 0.0, "components": {},
                     "summary": "analysis failed — Neutral"}
        automated = trend.get("sentiment", "Neutral")
        self._trend_composite = float(trend.get("composite", 0.0) or 0.0)
        logging.info(
            f"[MarketTrend] Verdict: {trend.get('verdict')} "
            f"(score {trend.get('composite'):+.2f}) → {automated}"
        )

        # Show the driver breakdown on the terminal.
        if self._is_interactive_tty():
            try:
                self._print_event(
                    self.market_trend_analyzer.format_breakdown(trend), level="info"
                )
            except Exception:
                pass

        # 3. Headless / non-interactive → accept automated silently.
        if not self._is_interactive_tty():
            logging.info(f"[MarketTrend] No TTY → using automated verdict '{automated}'.")
            return automated

        # 4. 20-second operator override.
        canonical_by_norm = {" ".join(s.split()).lower(): s for s in valid}
        # Convenience aliases so the operator can just type CE / PE.
        canonical_by_norm.update({"ce": "Bullish", "pe": "Bearish",
                                  "buy ce": "Bullish", "buy pe": "Bearish"})
        timeout = float(self.config['trading_flags'].get('operator_input_timeout_seconds', 20))

        while True:
            user_input = await self._input_with_timeout(
                f"\nPre-market trend verdict: {trend.get('verdict')} ({automated})\n"
                f"Press Enter to accept, type CE / PE, or a sentiment "
                f"{sorted(valid)} to override (auto-accepts in {int(timeout)}s): ",
                timeout=timeout,
            )
            if user_input is None:
                logging.info(f"[MarketTrend] Auto-resolved on timeout: {automated}")
                return automated
            if user_input == "":
                logging.info(f"[MarketTrend] Operator accepted automated verdict: {automated}")
                return automated
            canonical = canonical_by_norm.get(" ".join(user_input.split()).lower())
            if canonical:
                logging.info(f"[MarketTrend] Operator overrode {automated} → {canonical}")
                return canonical
            logging.warning(
                f"Invalid input '{user_input}'. Type CE, PE, one of {sorted(valid)} "
                f"(case-insensitive), or press Enter to accept '{automated}'."
            )

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

        # Surface YouTube analyst verdicts (if any) BEFORE the news headlines so
        # the operator can see exactly what every input is saying.
        try:
            yt_verdicts = (
                self.youtube_agent.get_verdicts()
                if self.youtube_agent and self.youtube_agent.is_ready()
                else []
            )
        except Exception:
            yt_verdicts = []
        if yt_verdicts:
            print("\n" + "=" * 78)
            print("YouTube analyst views (last 18 hours, sponsor segments stripped):")
            print("=" * 78)
            for v in yt_verdicts:
                name = v.get("channel_display_name", "")
                direction = v.get("direction", "Neutral")
                conf = float(v.get("confidence", 0) or 0)
                exc = float(v.get("excitement_level", 0) or 0)
                weight = float(v.get("channel_weight", 10) or 10)
                strip_method = v.get("sponsor_strip_method", "?")
                lv = v.get("specific_levels") or {}
                lv_bits = []
                if lv.get("nifty_target"): lv_bits.append(f"target {lv['nifty_target']}")
                if lv.get("support"):      lv_bits.append(f"support {lv['support']}")
                if lv.get("resistance"):   lv_bits.append(f"resistance {lv['resistance']}")
                lv_str = f"  [{', '.join(lv_bits)}]" if lv_bits else ""
                print(
                    f"  {name:30} | {direction:14}  conf={conf:.2f}  "
                    f"excite={exc:.2f}  wgt={weight:.0f}  [{strip_method}]{lv_str}"
                )
                thesis = v.get("key_thesis") or ""
                if thesis:
                    if len(thesis) > 130:
                        thesis = thesis[:127] + "..."
                    print(f"    \"{thesis}\"")
            print("=" * 78)

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
                src = h.get("source") or ""
                pub = h.get("published_at") or ""
                try:
                    pub_fmt = datetime.datetime.fromisoformat(
                        pub.replace("Z", "+00:00")
                    ).strftime("%d %b %H:%M")
                except Exception:
                    pub_fmt = ""
                meta = f"  ({src}{', ' + pub_fmt if pub_fmt else ''})" if src or pub_fmt else ""
                print(f"  {marker} {p:+.3f}  {title}{meta}")
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

    def _print_override_hint(self) -> None:
        """
        Prints a one-time instruction block explaining how to override sentiment
        mid-session. Called once after the startup sentiment is locked in.
        Only prints on a real TTY; silent in headless/CI runs.
        """
        if not self._is_interactive_tty():
            return
        manual_line = (
            "     • Strategy: type  m  + Enter  to re-pick the active strategy.\n"
            if self._manual_mode else
            "     • Strategy: type  m  + Enter  to manually pick a strategy.\n"
        )
        print(
            "\n" + "─" * 78 + "\n"
            "  💡 Mid-session overrides (while bot is running):\n"
            "     • Sentiment: type  Bullish / Bearish / Very Bullish / Very Bearish / Neutral\n"
            "                  then press Enter — or write to output/sentiment_override.txt\n"
            + manual_line +
            "     • Help     : type  ?  + Enter  to see current values and valid options.\n"
            + "─" * 78
        )

    # ------------------------------------------------------------------ #
    #  Manual Strategy Selection                                          #
    # ------------------------------------------------------------------ #

    # One-liner description shown in the picker menu for each strategy.
    _STRATEGY_DESCRIPTIONS: dict = {
        "Gemini_Default":               "AI-driven adaptive strategy (default fallback)",
        "Supertrend_MACD":              "Trend-following — Supertrend + MACD confirmation",
        "Volatility_Cluster_Reversal":  "Mean reversion after volatility-cluster bursts",
        "Volume_Spread_Analysis":       "VSA — price+volume relationship, smart-money signals",
        "Momentum_VWAP_RSI":            "Momentum burst above VWAP with RSI confirmation",
        "Breakout_Prev_Day_HL":         "Breakout above/below previous day high-low",
        "Opening_Range_Breakout":       "ORB — first 15-min range breakout (news/event days)",
        "BB_Squeeze_Breakout":          "Bollinger Band squeeze → volatility expansion entry",
        "MA_Crossover":                 "Moving-average crossover signal",
        "RSI_Divergence":               "RSI divergence from price — reversal signal",
        "EMA_Cross_RSI":                "EMA 20/50 crossover filtered by RSI — trending days",
        "Reversal_Detector":            "Multi-indicator reversal (hammer, engulf, PSAR flip)",
        "VWAP_Reversion":               "Mean reversion to VWAP — overextended range days",
        "NR7_Compression":              "NR7 compression breakout — tightest range in 7 bars",
        "Expiry_Momentum_Scalp":        "Expiry-day scalp — momentum + gamma pin risk",
    }

    async def _prompt_strategy_picker(self) -> "str | None":
        """
        Shows a numbered strategy menu and waits for the operator to pick one.
        Returns the canonical strategy name, or None on timeout / invalid input.

        Pauses the ticker while waiting so keystrokes aren't overwritten.
        Can be called at startup (first setup) or mid-session ('m' command).
        """
        if not self._is_interactive_tty():
            return None

        names = list(self._STRATEGY_DESCRIPTIONS.keys())
        w = 78

        self._ticker_paused = True
        self._clear_status_line()
        try:
            print("\n" + "═" * w)
            print("  📊  Manual Strategy Selection")
            sent = getattr(self, 'day_sentiment', '') or 'Unknown'
            conds = ", ".join(sorted(getattr(self, 'todays_conditions', set()) or set())) or "—"
            print(f"  Sentiment: {sent}  │  Conditions: {conds}")
            print("═" * w)
            for i, name in enumerate(names, 1):
                desc = self._STRATEGY_DESCRIPTIONS.get(name, "")
                print(f"  {i:>2}.  {name:<35} {desc}")
            print("═" * w)
            timeout = float(self.config['trading_flags'].get('operator_input_timeout_seconds', 60))
            print(
                f"  Enter number (1–{len(names)}) or strategy name "
                f"({int(timeout)}s timeout → auto-selector takes over): ",
                end="", flush=True,
            )

            def _wait_for_line():
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], timeout)
                except Exception:
                    return None
                if not ready:
                    return None
                try:
                    line = sys.stdin.readline()
                    return line if line else None
                except Exception:
                    return None

            raw_line = await asyncio.to_thread(_wait_for_line)
        finally:
            self._ticker_paused = False

        if raw_line is None:
            print(f"\n  [no response in {int(timeout)}s — falling back to auto-selector]")
            logging.info("Strategy picker timed out; auto-selector will be used.")
            return None

        raw = raw_line.strip()
        if not raw:
            logging.info("Strategy picker: empty input; auto-selector will be used.")
            return None

        # Accept a number.
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(names):
                chosen = names[idx]
                print(f"  ✔  Selected: {chosen}")
                return chosen
            print(f"  ✘  '{raw}' out of range (1–{len(names)}). Using auto-selector.")
            return None

        # Accept a strategy name (case-insensitive, underscore/space tolerant).
        norm = raw.replace(" ", "_").strip().lower()
        for name in names:
            if name.lower() == norm or name.replace("_", "").lower() == norm.replace("_", ""):
                print(f"  ✔  Selected: {name}")
                return name

        print(f"  ✘  '{raw}' not recognised. Using auto-selector.")
        logging.warning(f"[ManualMode] Unrecognised strategy input: '{raw}'")
        return None

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

        # Use the futures token (resolved at agent init) — indices return
        # volume=0 on historical bars, which silently breaks every
        # volume-using strategy. signal_data_token falls back to the
        # underlying/index token if no futures contract was found.
        signal_token = getattr(
            self.order_agent, "signal_data_token", self.order_agent.underlying_token
        )
        hist = await asyncio.to_thread(
            self.kite.historical_data,
            signal_token,
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
        Refreshes the terminal status line on each tick while awaiting a signal.
        """
        self._print_status_line()
        sleep_for = max(1.0, min(max_seconds, 5.0))
        await asyncio.sleep(sleep_for)

    # ------------------------------------------------------------------
    # 15-minute bar confirmation helpers
    # ------------------------------------------------------------------

    async def _get_15min_bars(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Returns 15-minute intraday bars with all indicators pre-computed.
        Uses a separate cache from the 5-min bars — refreshed only when a new
        15-min bar closes (every 15 minutes) to avoid extra API calls.
        """
        timeframe = "15minute"
        bar_idx = self._current_bar_index(timeframe)
        if (
            not force_refresh
            and self._bars_15m_cache is not None
            and self._bars_15m_cached_at_bar == bar_idx
        ):
            return self._bars_15m_cache

        signal_token = getattr(
            self.order_agent, "signal_data_token", self.order_agent.underlying_token
        )
        try:
            hist = await asyncio.to_thread(
                self.kite.historical_data,
                signal_token,
                datetime.datetime.now() - datetime.timedelta(days=5),
                datetime.datetime.now(),
                timeframe,
            )
            df = pd.DataFrame(hist)
            if not df.empty:
                df = calculate_all_indicators(df, self.config)
        except Exception as exc:
            logging.warning(f"[15m bars] Fetch failed (non-fatal): {exc}")
            df = pd.DataFrame()
        self._bars_15m_cache      = df
        self._bars_15m_cached_at_bar = bar_idx
        return df

    @staticmethod
    def _check_15min_confirmation(signal: str, df_15m: pd.DataFrame) -> tuple:
        """
        Returns (confirmed: bool, reason: str).

        BUY  confirmed when 15-min RSI > 50  AND  close > 15-min EMA20.
        SELL confirmed when 15-min RSI < 50  AND  close < 15-min EMA20.

        The gate is bypassed (confirmed=True) when:
          - df_15m is empty or missing required columns
          - any indicator is NaN on the latest bar
        so a data outage never silently blocks all trades.
        """
        import math
        if df_15m is None or df_15m.empty:
            return True, "15m: no data — gate bypassed"
        if "rsi" not in df_15m.columns or "ema_20" not in df_15m.columns:
            return True, "15m: indicators missing — gate bypassed"
        last  = df_15m.iloc[-1]
        rsi   = float(last.get("rsi",   float("nan")))
        ema20 = float(last.get("ema_20", float("nan")))
        close = float(last.get("close",  float("nan")))
        if any(math.isnan(v) for v in (rsi, ema20, close)):
            return True, "15m: NaN indicators — gate bypassed"

        if signal == "BUY":
            rsi_ok = rsi   > 50
            ema_ok = close > ema20
            if rsi_ok and ema_ok:
                return True, (
                    f"15m BUY confirmed: RSI={rsi:.1f}>50, "
                    f"close={close:.2f}>EMA20={ema20:.2f}"
                )
            reasons = []
            if not rsi_ok:
                reasons.append(f"RSI={rsi:.1f}≤50")
            if not ema_ok:
                reasons.append(f"close={close:.2f}≤EMA20={ema20:.2f}")
            return False, f"15m BUY not confirmed: {', '.join(reasons)}"

        if signal == "SELL":
            rsi_ok = rsi   < 50
            ema_ok = close < ema20
            if rsi_ok and ema_ok:
                return True, (
                    f"15m SELL confirmed: RSI={rsi:.1f}<50, "
                    f"close={close:.2f}<EMA20={ema20:.2f}"
                )
            reasons = []
            if not rsi_ok:
                reasons.append(f"RSI={rsi:.1f}≥50")
            if not ema_ok:
                reasons.append(f"close={close:.2f}≥EMA20={ema20:.2f}")
            return False, f"15m SELL not confirmed: {', '.join(reasons)}"

        return True, "HOLD — no confirmation needed"

    # ------------------------------------------------------------------
    # PCR gate helper
    # ------------------------------------------------------------------

    def _is_pcr_aligned(self, signal: str) -> bool:
        """
        Returns True if the most-recent PCR reading is compatible with the
        trade direction, or if PCR data is unavailable / neutral (gate bypassed).

        Alignment rules (contrarian interpretation):
          PCR_BULLISH + BUY  → aligned   (put build-up confirms long bias)
          PCR_BULLISH + SELL → misaligned
          PCR_BEARISH + SELL → aligned   (call build-up confirms short bias)
          PCR_BEARISH + BUY  → misaligned
          PCR_NEUTRAL / no data → always aligned (bypass)
        """
        tag = (self._pcr_data or {}).get("tag", "")
        if not tag or tag in ("PCR_NEUTRAL", "PCR_DISABLED", "PCR_ERROR", ""):
            return True
        if signal == "BUY"  and tag == "PCR_BEARISH":
            return False
        if signal == "SELL" and tag == "PCR_BULLISH":
            return False
        return True

    def _is_reentry_blocked(self, signal: str) -> bool:
        """
        Re-entry guard: after a trade exits, refuse to re-enter the SAME direction
        for `reentry_cooldown_minutes`. Stops the chop loop where the indicator
        exit cuts a position at a small loss and the still-valid signal instantly
        re-buys the same option — paying costs twice and bleeding on every wobble.
        Set reentry_cooldown_minutes: 0 to disable.
        """
        mins = float((self.config.get('trading_flags') or {}).get('reentry_cooldown_minutes', 0) or 0)
        if mins <= 0 or not self._last_exit_time or not self._last_exit_direction:
            return False
        if signal != self._last_exit_direction:
            return False
        elapsed_min = (datetime.datetime.now() - self._last_exit_time).total_seconds() / 60.0
        if elapsed_min < mins:
            logging.warning(
                f"Re-entry guard: blocking '{signal}' — same direction was exited "
                f"{elapsed_min:.1f} min ago (cooldown {mins:.0f} min). "
                f"Avoiding a chop in/out of the same setup."
            )
            return True
        return False

    # Default confluence weights for the setup score (overridable in config).
    _SCORE_WEIGHTS_DEFAULT = {
        "direction": 20,   # signal agrees with the day's directional bias
        "htf_15m":   25,   # 15-min higher-timeframe confirmation
        "pcr":       15,   # option-flow (PCR) alignment
        "oi_wall":   15,   # not buying into the dominant call-OI wall
        "momentum":  15,   # ATR momentum sufficient for options buying
        "day_quality": 10, # TRENDING > RANGE > UNKNOWN
    }

    async def _setup_score(self, signal: str, df, is_counter_sentiment: bool) -> tuple:
        """
        Weighted confluence score (0-100) replacing the old AND-gate stack.
        Each factor contributes 0..1 × weight; factors with NO data are EXCLUDED
        (score normalized over available weights) instead of counting against
        the setup. Returns (score, size_factor, breakdown).
        """
        cfg = (self.config.get('setup_score') or {})
        weights = dict(self._SCORE_WEIGHTS_DEFAULT)
        weights.update(cfg.get('weights') or {})
        factors: dict = {}   # name -> (value 0..1, weight)

        # 1. Direction alignment (counter-trades get partial credit, not a veto).
        w = float(weights.get('direction', 20))
        if w > 0:
            factors['direction'] = (
                float(cfg.get('counter_direction_value', 0.4)) if is_counter_sentiment else 1.0, w)

        # 2. 15-min higher-timeframe confirm (excluded for mean-reversion + no-data).
        w = float(weights.get('htf_15m', 25))
        _is_mr = (getattr(self.active_strategy, 'is_reversal_trade', False)
                  or self.active_strategy_name in self._MEAN_REVERSION_STRATEGIES)
        if (w > 0 and not _is_mr
                and (self.config.get('trading_flags') or {}).get('enable_15m_confirmation', True)):
            try:
                df_15m = await self._get_15min_bars()
                _ok, _reason = self._check_15min_confirmation(signal, df_15m)
                if 'bypassed' not in _reason:
                    factors['htf_15m'] = (1.0 if _ok else 0.0, w)
            except Exception:
                pass

        # 3. PCR option-flow alignment (excluded when neutral / unavailable).
        w = float(weights.get('pcr', 15))
        if w > 0 and (self._pcr_data or {}).get('tag') in ('PCR_BULLISH', 'PCR_BEARISH'):
            factors['pcr'] = (1.0 if self._is_pcr_aligned(signal) else 0.0, w)

        # 4. OI wall (excluded when wall data unavailable).
        w = float(weights.get('oi_wall', 15))
        if w > 0 and (self._pcr_data or {}).get('call_walls'):
            factors['oi_wall'] = (0.0 if self._is_oi_wall_blocked(signal) else 1.0, w)

        # 5. ATR momentum (excluded when the gate is disabled in config).
        w = float(weights.get('momentum', 15))
        if w > 0 and float((self.config.get('trading_flags') or {}).get('min_atr_per_bar', 0) or 0) > 0:
            factors['momentum'] = (0.0 if self._is_momentum_too_low(df) else 1.0, w)

        # 6. Day quality.
        w = float(weights.get('day_quality', 10))
        if w > 0:
            dq = getattr(self, '_day_quality', 'UNKNOWN')
            factors['day_quality'] = ({'TRENDING': 1.0, 'RANGE': 0.6}.get(dq, 0.5), w)

        total_w = sum(wt for _, wt in factors.values())
        score = (100.0 * sum(v * wt for v, wt in factors.values()) / total_w) if total_w else 100.0
        full_above = float(cfg.get('size_full_above', 85))
        min_factor = float(cfg.get('size_min_factor', 0.75))
        size_factor = 1.0 if score >= full_above else max(min_factor, round(score / full_above, 2))
        breakdown = {k: round(v, 2) for k, (v, _) in factors.items()}

        # Funnel bookkeeping (the gate-attribution metric).
        self._scored_today += 1
        if score > self._best_score_today:
            self._best_score_today = round(score, 1)
            self._best_score_info = f"{signal} via {self.active_strategy_name} {breakdown}"
        return round(score, 1), size_factor, breakdown

    async def _score_gate(self, signal: str, df, is_counter_sentiment: bool,
                          force_mode: bool) -> bool:
        """Entry decision from the setup score. Stashes the score-based size
        factor for the sizing path. Returns True = proceed to entry."""
        self._last_score_size = 1.0
        self._last_score = None
        cfg = (self.config.get('setup_score') or {})
        if force_mode or not cfg.get('enable', True):
            return True
        score, size_factor, breakdown = await self._setup_score(signal, df, is_counter_sentiment)
        threshold = float(cfg.get('entry_threshold', 65))
        self._last_score = score
        if score >= threshold:
            self._last_score_size = size_factor
            logging.info(
                f"[SetupScore] {signal} PASS — {score:.0f} ≥ {threshold:.0f} "
                f"(size ×{size_factor:.2f})  factors={breakdown}"
            )
            return True
        self._log_score_block(signal, score, threshold, breakdown)
        return False

    def _log_score_block(self, signal, score, threshold, breakdown) -> None:
        """Deduped log for score-blocked entries (avoids per-tick spam)."""
        key = (signal, round(score))
        if key != getattr(self, '_last_score_block_key', None):
            logging.warning(
                f"[SetupScore] {signal} blocked — {score:.0f} < {threshold:.0f}. "
                f"Factors: {breakdown}"
            )
            self._last_score_block_key = key

    def _validate_risk_coherence(self) -> None:
        """
        Startup sanity check on the risk triangle: per-trade risk must be ≤ half
        the daily loss cap, or the daily breaker halts the day after ONE stop-out
        and every multi-trade frequency setting is unreachable. Warns loudly —
        does not silently change the operator's numbers.
        """
        try:
            rm = self.config.get('risk_management', {}) or {}
            daily = float(rm.get('max_daily_loss_percent', 2.5))
            tiers = self.config.get('risk_tiers') or []
            worst = max(
                (float(t.get('risk_pct', 0) or 0) for t in tiers),
                default=float((self.config.get('trading_flags') or {}).get('risk_per_trade_percent', 1.0)),
            )
            if worst > daily / 2.0:
                logging.warning(
                    f"[RiskCoherence] INCOHERENT: max per-trade risk {worst:.1f}% > "
                    f"half the daily-loss cap ({daily:.1f}%). One stop-out can end the "
                    f"day; max_trades_per_day is effectively unreachable on red days. "
                    f"Coherent rule: per-trade risk ≤ daily_cap/2."
                )
        except Exception:
            pass

    def _is_oi_wall_blocked(self, signal: str) -> bool:
        """
        Playbook OI-wall gate: don't buy CALLS into a heavy call-OI wall (where
        institutions are writing calls = resistance). For a BUY (long CE), if spot
        (ATM) has already reached/passed the nearest heavy call-OI strike, the
        upside is being defended — skip. PE buys are the intended break-down
        direction, so they're not gated here. Bypassed when data/feature absent.
        """
        cfg = (self.config.get('oi_analytics') or {})
        if not cfg.get('enable', True):
            return False
        pcr = self._pcr_data or {}
        atm = pcr.get('atm')
        walls = pcr.get('call_walls') or []   # sorted by OI desc → [0] is dominant
        if signal == 'BUY' and atm and walls:
            dominant = walls[0]
            # Block only when spot is sitting ON the single heaviest call-OI strike
            # (a clear resistance test). Below it = room to run (OK); above it =
            # already broke out (OK). This avoids over-blocking on minor walls.
            if abs(float(atm) - float(dominant)) < 1e-6:
                logging.warning(
                    f"OI-wall gate: ATM {atm:.0f} sits ON the heaviest call-OI wall "
                    f"({dominant:.0f}) — buying calls into institutional resistance. Skipping."
                )
                return True
        return False

    # Human-readable rationale for each entry-blocking reason.
    _BLOCK_LABELS: dict = {
        "no_signal":       "strategy returned HOLD — waiting for its entry pattern",
        "choppy":          "day classified CHOPPY — too many direction changes, no trend to ride",
        "range":           "day classified RANGE / no scalp setup — price boxed in",
        "momentum_low":    "ATR momentum too low — market too slow for options buying",
        "vix_high":        "VIX above the max — too volatile/risky to enter",
        "pcr":             "PCR option-flow contradicted the signal direction",
        "oi_wall":         "spot at/above a heavy call-OI wall — buying calls into resistance",
        "reentry_cooldown":"re-entry guard — same direction was just exited (cooldown active)",
        "score_below_threshold": "setup score below entry threshold — confluence too weak",
        "trap":            "false-breakout trap detected — stop-hunt pattern",
        "confirm_15m":     "15-min higher-timeframe didn't confirm the signal",
        "no_trade_window": "outside the entry window (pre-open / lunch / past cutoff)",
        "max_trades":      "daily max-trades already reached",
        "cooldown":        "active strategy cooled (0 signals last window) — rotating",
    }

    def _note_block(self, reason: str) -> None:
        """Tally an entry-blocking reason for the 'why no trade' diagnosis."""
        self._block_reasons[reason] = self._block_reasons.get(reason, 0) + 1

    def _plain_verdict(self, top_key: str) -> str:
        dq = getattr(self, '_day_quality', 'UNKNOWN')
        if top_key == "choppy" or dq == "CHOPPY":
            return ("market is CHOPPY (whipsawing, no clean trend) — sitting out is the "
                    "correct call; forcing trades here just bleeds theta + costs.")
        if top_key == "range" or dq == "RANGE":
            return ("market is RANGE-BOUND (no breakout) — no edge for options buying; "
                    "staying flat.")
        if top_key == "vix_high":
            return "market too volatile (VIX above the limit) — risk too high to enter."
        if top_key == "momentum_low":
            return "market too slow (low ATR) — moves can't cover premium decay + costs."
        if top_key in ("confirm_15m", "pcr", "trap"):
            return ("signals kept failing confirmation/quality gates — the setups that "
                    "appeared were low-quality, so the bot passed.")
        if top_key == "no_signal":
            return ("the selected strategy never found its entry pattern in today's price "
                    "action — no valid signal to act on.")
        if top_key == "no_trade_window":
            return "still outside the allowed entry window."
        if top_key == "max_trades":
            return "daily trade cap reached — no more entries by design."
        return "entry conditions never aligned across the gate checks."

    def _no_trade_diagnosis(self) -> list:
        """
        Human-readable explanation of WHY no trade has fired, ranked by how often
        each gate blocked an entry. Surfaced live at each reassessment and in the
        end-of-day session summary.
        """
        lines = []
        if self.no_trade_reason:
            lines.append(f"Setup verdict: {self.no_trade_reason}")
        if not self._block_reasons:
            if not lines:
                lines.append("No entry evaluated yet (warming up / outside entry window).")
            return lines
        ranked = sorted(self._block_reasons.items(), key=lambda kv: kv[1], reverse=True)
        lines.append("Why no trade so far (most frequent blockers):")
        for key, n in ranked[:5]:
            lines.append(f"   • {self._BLOCK_LABELS.get(key, key)}  ×{n}")
        # Setup-score funnel: makes idleness falsifiable instead of mysterious.
        if self._scored_today:
            _thr = float((self.config.get('setup_score') or {}).get('entry_threshold', 65))
            lines.append(
                f"Funnel: {self._scored_today} setup(s) scored │ best "
                f"{self._best_score_today:.0f} vs threshold {_thr:.0f} │ "
                f"passed {self._score_passed_today}"
            )
            if self._best_score_info:
                lines.append(f"   best setup: {self._best_score_info}")
        # KPI floor: a TRENDING day with zero attempts is a system failure to
        # investigate, not prudence to be proud of.
        if (getattr(self, 'day_regime', '') == 'TRENDING'
                and self.trades_today_count == 0 and self._scored_today):
            lines.append(
                "⚠ SYSTEM CHECK: TRENDING day with 0 trades — review the funnel "
                "line above; the top blocker is the calibration to fix."
            )
        lines.append(f"Verdict: {self._plain_verdict(ranked[0][0])}")
        return lines

    def _ledger_path(self) -> str:
        return state_path(f"trade_ledger_{self._today_str}.json")

    def _persist_ledger(self) -> None:
        """
        Persist today's trade ledger + counters to disk so a same-day RESTART
        restores the full day (for the end-of-day summary AND the daily-max
        trade cap). Written on every entry and every completed trade. Realized
        P&L itself is already persisted separately (daily_pnl.json).
        """
        try:
            atomic_write_json(self._ledger_path(), {
                "date": self._today_str,
                "trades_today_count": int(self.trades_today_count),
                "wins": int(self.wins_today),
                "losses": int(self.losses_today),
                "scratch": int(self.scratch_today),
                "completed_trades": self.completed_trades,
                "starting_capital": self.starting_capital,
                "ending_capital": self.ending_capital,
            })
        except Exception as e:
            logging.debug(f"Could not persist trade ledger: {e}")

    def _load_ledger(self) -> None:
        """
        Restore today's ledger from disk on startup (same-day restart). No-op if
        the file is missing or belongs to a previous day — so a fresh day starts
        clean. Keeps the EOD report/summary and the daily-max cap correct across
        restarts.
        """
        try:
            data = read_json(self._ledger_path(), default=None)
            if not isinstance(data, dict) or data.get("date") != self._today_str:
                return
            self.completed_trades = data.get("completed_trades", []) or []
            self.wins_today    = int(data.get("wins", 0) or 0)
            self.losses_today  = int(data.get("losses", 0) or 0)
            self.scratch_today = int(data.get("scratch", 0) or 0)
            self.trades_today_count = int(data.get("trades_today_count", 0) or 0)
            if self.completed_trades or self.trades_today_count:
                logging.info(
                    f"Restored today's ledger from disk: {self.trades_today_count} "
                    f"trade(s) placed, {len(self.completed_trades)} closed "
                    f"(W:{self.wins_today} L:{self.losses_today}). "
                    f"Daily-max cap and EOD summary now cover the full day."
                )
        except Exception as e:
            logging.debug(f"Could not load trade ledger: {e}")

    def _current_balance(self) -> tuple:
        """
        Returns (balance, label) for the running account balance.

        LIVE  → broker margin (self.starting_capital is re-fetched each loop via
                _refresh_starting_capital, so it already reflects realized P&L —
                this is a real-time value, polled every cycle).
        PAPER → starting capital + simulated realized P&L (paper fills never
                touch the real broker balance, so we track it ourselves).
        """
        start = float(self.starting_capital or 0.0)
        is_paper = bool((self.config.get('trading_flags') or {}).get('paper_trading', True))
        if is_paper:
            return start + float(self.realized_pnl_today or 0.0), "sim"
        return start, "live"

    def _print_session_summary(self) -> None:
        """
        Print a one-time end-of-session summary: every trade taken, gross vs net
        (after costs), win/loss tally, and the start→end balance with its source
        (live broker vs simulated). Called from run()'s finally block.
        """
        if self._session_summary_printed:
            return
        self._session_summary_printed = True
        if not self._is_interactive_tty():
            return

        bal, lbl = self._current_balance()
        start = float(self.starting_capital or 0.0)
        net   = float(self.realized_pnl_today or 0.0)
        gross = sum(t['gross'] for t in self.completed_trades)
        costs = sum(t['costs'] for t in self.completed_trades)
        net_str = f"+₹{net:,.2f}" if net >= 0 else f"-₹{abs(net):,.2f}"
        bal_src = ("paper — real broker balance unchanged; this is start + simulated P&L"
                   if lbl == "sim" else
                   "live broker balance, polled every loop cycle (real-time)")

        lines = [
            f"SESSION SUMMARY  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Trades placed : {self.trades_today_count}   "
            f"│  Closed: {len(self.completed_trades)}  "
            f"(W:{self.wins_today}  L:{self.losses_today}  scratch:{self.scratch_today})",
            f"  Gross P&L     : ₹{gross:,.2f}   │  Costs: ₹{costs:,.2f}   │  NET: {net_str}",
            f"  Balance       : ₹{start:,.0f}  →  ₹{bal:,.0f}   ({lbl})",
            f"                  {bal_src}",
        ]
        if self.completed_trades:
            lines.append("  Trades:")
            for i, t in enumerate(self.completed_trades, 1):
                tnet = t['net']
                tnet_str = f"+₹{tnet:,.2f}" if tnet >= 0 else f"-₹{abs(tnet):,.2f}"
                lines.append(
                    f"   {i:>2}. {t['type']:<4} {t['symbol']:<22} "
                    f"net {tnet_str:>12}  (gross ₹{t['gross']:,.0f} − costs ₹{t['costs']:,.0f})  "
                    f"[{t['strategy']}]"
                )
        else:
            lines.append("  No trades were taken this session.")
            # Explain WHY — the dominant blockers + a plain-English verdict.
            for dl in self._no_trade_diagnosis():
                lines.append(f"  {dl}")
        self._print_event(lines, level="info")

    def _record_realized_pnl(self, delta_pnl: float) -> None:
        amount = float(delta_pnl or 0)
        self.realized_pnl_today += amount
        self.realized_pnl_week  += amount

        # Win/loss tally (on NET P&L — `amount` is already net of costs).
        if amount > 0:
            self.wins_today += 1
        elif amount < 0:
            self.losses_today += 1
        else:
            self.scratch_today += 1

        # Update the consecutive-loss streak and progressive size multiplier.
        if amount < 0:
            self.consecutive_losses += 1
            logging.warning(
                f"[Risk] Consecutive losses: {self.consecutive_losses} "
                f"(trade P&L={amount:,.2f})"
            )
            # Progressive loss sizing — reduce size to protect capital.
            if self.consecutive_losses == 1:
                self._trade_size_multiplier = 0.5
                logging.warning(
                    "[ProSize] 1 consecutive loss: next trade size reduced to 50%."
                )
            elif self.consecutive_losses >= 2:
                self._trade_size_multiplier = 0.25
                logging.warning(
                    f"[ProSize] {self.consecutive_losses} consecutive losses: "
                    f"next trade size reduced to 25%. Requires stricter confirmation."
                )
        elif amount > 0:
            if self.consecutive_losses > 0:
                logging.info(
                    f"[Risk] Consecutive-loss streak broken after "
                    f"{self.consecutive_losses} loss(es) — resetting to 0."
                )
            if getattr(self, '_trade_size_multiplier', 1.0) < 1.0:
                logging.info(
                    "[ProSize] Winning trade: restoring full position size."
                )
            self.consecutive_losses = 0
            self._trade_size_multiplier = 1.0

        try:
            save_daily_pnl(self._today_str, self.realized_pnl_today)
            save_weekly_pnl(self._today_week_str, self.realized_pnl_week)
        except Exception as e:
            logging.warning(f"Could not persist P&L: {e}")

    async def _run_monitor(self):
        """
        Status-only mode (--monitor): publish a live status snapshot to the
        dashboard every 2 seconds without trading or needing a Zerodha session.
        Used to verify the bot→dashboard pipeline off-hours. Ctrl+C to stop.
        """
        self.bot_state = "MONITOR"
        self.no_trade_reason = (
            "Monitor mode — publishing live status to the dashboard; not trading."
        )
        self._report_sent = True  # never email a report from monitor mode
        print("\n" + "=" * 70)
        print("  MONITOR MODE — publishing live status to the dashboard.")
        print("  The bot will NOT trade and needs no Zerodha login. Ctrl+C to stop.")
        print("=" * 70 + "\n")
        logging.info("MONITOR MODE active — dashboard will show live bot status.")
        try:
            while True:
                self._write_status_snapshot()
                await asyncio.sleep(2)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.bot_state = "STOPPED"
            self._write_status_snapshot()
            print("\nMonitor stopped.")

    async def run(self):
        """The main event loop for the trading bot."""
        # Publish an initial status snapshot immediately so the dashboard shows
        # the REAL bot state from the first moment (not 'offline'), even before
        # the market-hours check and on any early exit below.
        self._write_status_snapshot()
        # --monitor: a pure status publisher. Keeps the REAL bot process alive,
        # writing live status to the dashboard every 2s, but NEVER trades and
        # needs no Zerodha session. Lets you verify the bot→dashboard pipeline
        # any time (off-hours, weekends) before a real session.
        if getattr(self, "_monitor_mode", False):
            await self._run_monitor()
            return
        # Accept startup during market hours OR the pre-market warm-up window
        # (default 08:50 -> 09:15 IST). Anything else -> closed-info banner.
        if not (self.is_market_open() or self.is_pre_market_window()):
            # Safety net for edge case: bot started just after 15:30 close.
            # Holiday / weekend exits are caught in __main__ before auth runs.
            # Record the real reason so the dashboard explains the idle state.
            self.bot_state = "STOPPED"
            self.no_trade_reason = (
                "Market is closed — outside 09:15–15:30 IST (or weekend/holiday). "
                "Start the bot during market/pre-market hours for live updates, "
                "or run with --monitor to publish live status without trading."
            )
            self._write_status_snapshot()
            await self.display_market_closed_info()
            return  # No report — bot never attempted trading.

        if self.is_pre_market_window() and not self.is_market_open():
            now_str = datetime.datetime.now().strftime('%H:%M:%S')
            print("\n" + "=" * 78)
            print(f"  Pre-market start at {now_str} — running setup ahead of 09:15 open.")
            print("=" * 78)

        # Install a log handler that clears the status line before every log
        # message so multi-line logs don't overlap the live status ticker.
        _root_logger = logging.getLogger()
        _status_handler = _StatusLineAwareHandler(self)
        _status_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        _root_logger.addHandler(_status_handler)

        # Remove the default StreamHandler so messages aren't printed twice.
        _orig_handlers = [h for h in _root_logger.handlers if h is not _status_handler]
        for h in _orig_handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                _root_logger.removeHandler(h)

        # Start the 1-second status ticker.
        ticker = asyncio.ensure_future(self._run_ticker())

        try:
            await self._run_inner()
        except (KeyboardInterrupt, asyncio.CancelledError, SystemExit) as e:
            self._clear_status_line()
            print(f"\nBot stopped: {type(e).__name__}.")
        except Exception as e:
            logging.error(f"Unhandled exception in run(): {e}", exc_info=True)
        finally:
            ticker.cancel()
            try:
                await ticker
            except asyncio.CancelledError:
                pass
            # Restore original handlers.
            _root_logger.removeHandler(_status_handler)
            for h in _orig_handlers:
                _root_logger.addHandler(h)
            # Capture end-of-day capital (best-effort), print the session
            # summary, then send the daily (and, on the configured day, weekly) report.
            try:
                await self._capture_ending_capital()
            except Exception:
                pass
            self._print_session_summary()
            self._send_shutdown_report_once()

    async def _run_inner(self):
        """The actual trading orchestration — wrapped by run() in a try/finally."""
        # ── Token health-check before any setup work ──────────────────────────
        # A stale / expired Zerodha token surfaces here (with an email alert)
        # rather than failing mid-session after the user has already waited for
        # sentiment capture and strategy selection.
        if not await self._validate_token():
            return  # _validate_token already sent the email and set _abort=True.

        # Reconcile any persisted open position from a previous session before setup.
        try:
            resumed = await self.position_agent.reconcile_open_position()
        except Exception as e:
            logging.warning(f"Reconciliation skipped: {e}")
            resumed = False

        # Fetch capital BEFORE setup so the setup banner shows the real balance.
        # _capture_starting_capital is safe pre-market — Zerodha margins API
        # works at any time once the access token is valid.
        await self._capture_starting_capital()

        if not await self.setup():
            logging.warning(f"Setup failed. Reason: {self.no_trade_reason or 'Unknown'}. Bot will exit.")
            return  # Finally block in run() will send the report.

        # Warm-up backtest (optional): populate the RAG trade log with recent
        # historical signals before today's session starts. Uses an aligned date
        # window so the generated data falls within RAG's recency_window_days.
        await self._run_startup_backtest()

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
        logging.debug(f"Bot running in {'PAPER TRADING' if is_paper else 'LIVE TRADING'} mode.")
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
                    if self._is_consecutive_loss_breached():
                        self.bot_state = "STOPPED"; continue
                    if await self._is_weekly_loss_breached():
                        self.bot_state = "STOPPED"; continue
                    if await self._is_daily_profit_target_hit():
                        self.bot_state = "STOPPED"; continue

                    # ── Mid-session sentiment override ────────────────────────
                    # Check stdin (non-blocking) and the sentinel file on every
                    # loop tick. When the operator overrides, apply immediately
                    # and reset the auto-refresh baselines so the refresh logic
                    # doesn't flip back over the manual choice.
                    _override = await self._check_mid_session_override()
                    if _override and _override != self.day_sentiment:
                        _old_sent = self.day_sentiment
                        self.day_sentiment = _override
                        self._cached_sentiment = _override
                        # Reset baselines so sentiment_refresh doesn't undo the
                        # operator's choice on the very next reassessment cycle.
                        self._sentiment_baseline_auto = _override
                        logging.info(
                            f"[SentimentOverride] {_old_sent} → {_override} "
                            f"(operator mid-session override)"
                        )
                        self._print_event(
                            [
                                f"Sentiment overridden by operator:  {_old_sent}  →  {_override}",
                                "Strategy will apply the new sentiment from the next signal evaluation.",
                            ],
                            level="warn",
                        )

                    # Strategy reassessment timer — runs BEFORE the time gate so
                    # that the strategy keeps rotating even after 13:30 (e.g. when
                    # the bot started late). The `continue` inside the time gate
                    # previously bypassed this block entirely, leaving the bot
                    # frozen on the same strategy until kill.
                    reassessment_period = self.config['trading_flags'].get('strategy_reassessment_period_minutes', 60)
                    if self.awaiting_signal_since and (datetime.datetime.now() - self.awaiting_signal_since).total_seconds() > reassessment_period * 60:
                        # Cool down the current strategy if it produced zero
                        # non-HOLD signals during this window — re-picking the
                        # same strategy is wasteful. Time-based cooldown so it
                        # can come back later as market conditions evolve.
                        if (self.active_strategy_name
                                and self._signals_seen_for_active_strategy == 0
                                and self.active_strategy_name not in self._strategy_cooldown_until):
                            logging.info(
                                f"'{self.active_strategy_name}' produced 0 non-HOLD "
                                f"signals in {reassessment_period} min."
                            )
                            self._cool_strategy(self.active_strategy_name)
                            self._note_block("cooldown")
                        logging.warning(f"No trade signal for over {reassessment_period} minutes. Re-assessing strategy...")
                        # Surface the live "why no trade yet" rationale so the
                        # operator can see WHAT is blocking entries, not just that
                        # nothing fired.
                        if self.trades_today_count == 0:
                            self._print_event(self._no_trade_diagnosis(), level="info")
                        if not await self.setup():
                            self.bot_state = "STOPPED"; continue

                    # Time-of-day hard cutoff — earlier than entry_cutoff_time
                    # for new entries; protects against theta eating afternoon gains.
                    if self._time_of_day_size_factor() == 0.0:
                        _cutoff_str = self.config['trading_flags'].get('entry_cutoff_time', '13:30')
                        logging.debug(f"Past {_cutoff_str} — no new entries allowed.")
                        await asyncio.sleep(30)
                        continue
                    # AGGRESSIVE mode can raise the daily trade cap; expiry day
                    # overrides always take the minimum (tightest) of all limits.
                    tf = self.config['trading_flags']
                    max_trades = int(tf.get('_agg_max_trades') or tf['max_trades_per_day'])
                    if getattr(self, "is_expiry_day", False):
                        exp_cfg = self.config.get('expiry_day_overrides', {}) or {}
                        if exp_cfg.get('enable', True):
                            max_trades = min(max_trades, int(exp_cfg.get('max_trades', max_trades)))
                    if self.trades_today_count >= max_trades:
                        self.bot_state = "STOPPED"; continue

                    # Soft gate: in a no-trade window we just sleep and try again later.
                    no_trade = self._no_trade_window_reason()
                    if no_trade:
                        self._note_block("no_trade_window")
                        logging.debug(f"In no-trade window ({no_trade}); waiting.")
                        await asyncio.sleep(30)
                        continue

                    day_df_for_signal = await self._get_underlying_bars()
                    if day_df_for_signal is None or day_df_for_signal.empty:
                        logging.debug("Underlying bars unavailable; skipping iteration.")
                        await self._aligned_sleep()
                        continue

                    # Refresh PCR (cached for one bar; non-fatal on failure).
                    if self.pcr_feed is not None:
                        try:
                            spot = float(day_df_for_signal["close"].iloc[-1])
                            self._pcr_data = await self.pcr_feed.get_pcr(spot_price=spot)
                        except Exception as _pcr_exc:
                            logging.debug(f"PCR refresh skipped: {_pcr_exc}")

                    signal = self.active_strategy.generate_signals(
                        day_df_for_signal, self.day_sentiment,
                        cpr_pivots=self.position_agent.cpr_pivots,
                        vix_conditions=self.todays_conditions,
                        is_expiry_day=getattr(self, "is_expiry_day", False),
                    )

                    # Day quality filter.
                    #   TRENDING → normal flow, clear any leftover scalp flags.
                    #   RANGE    → scalp mode (VWAP/RSI-extreme, half-size, tight targets).
                    #   CHOPPY   → fully blocked; too many direction changes for any edge.
                    self._day_quality = self._classify_day_quality(day_df_for_signal)
                    # Log day quality only when it changes — avoid repeating the
                    # same line every 60 s while conditions are stable.
                    if self._day_quality != self._last_reported_day_quality:
                        logging.info(f"[DayQuality] → {self._day_quality}")
                        self._last_reported_day_quality = self._day_quality

                    if self._day_quality == 'CHOPPY':
                        self._note_block("choppy")
                        self._exit_range_scalp_mode()
                        logging.debug("[DayQuality] CHOPPY — skipping entry, sleeping 60 s.")
                        await asyncio.sleep(60)
                        continue
                    elif self._day_quality == 'RANGE':
                        scalp_ok = self._enter_range_scalp_mode(day_df_for_signal)
                        if not scalp_ok:
                            self._note_block("range")
                            await asyncio.sleep(60)
                            continue
                        # Fall through — scalp strategy + tight params are now set.
                    else:
                        # TRENDING or UNKNOWN — ensure scalp overrides are cleared.
                        self._exit_range_scalp_mode()

                    if signal == 'HOLD':
                        self._note_block("no_signal")
                    if signal != 'HOLD':
                        # The strategy fired *something* — bumps the per-strategy
                        # signal counter so reassessment doesn't cool it down.
                        # We still gate the trade below; cooldown only triggers
                        # on strategies that produce zero non-HOLD signals.
                        self._signals_seen_for_active_strategy += 1
                        is_primary_signal = (signal == 'BUY' and self.day_sentiment in ['Bullish', 'Very Bullish']) or (signal == 'SELL' and self.day_sentiment in ['Bearish', 'Very Bearish'])

                        # FORCE-MODE: skip sentiment-direction match and VIX gate.
                        # Liquidity / daily-loss / max-trades / IP whitelist
                        # remain enforced inside the order-placement path.
                        force_mode_now = self._force_mode_armed
                        if force_mode_now:
                            logging.warning(
                                f"FORCE-TRADE MODE armed: taking '{signal}' regardless "
                                f"of sentiment ({self.day_sentiment}) and VIX gate. "
                                f"Will auto-disarm after this trade fires."
                            )

                        # ── SOFT sentiment gate (Balanced fix) ────────────────
                        # Counter-sentiment signals are no longer HARD-BLOCKED.
                        # A signal that agrees with the locked sentiment (or a
                        # reversal strategy / force mode) is "full conviction" and
                        # trades at full size. A counter-sentiment signal still
                        # trades, but at a reduced size (counter_sentiment_size_factor,
                        # default 0.5). Every genuine risk control below
                        # (momentum / VIX / PCR / trap / 15m / SL / loss limits)
                        # is unchanged and still applies.
                        is_full_conviction = (
                            force_mode_now
                            or getattr(self.active_strategy, 'is_reversal_trade', False)
                            or is_primary_signal
                        )
                        is_counter_sentiment = not is_full_conviction
                        if is_counter_sentiment:
                            # Log only when (signal, sentiment) changes — avoids
                            # flooding the terminal every tick.
                            _counter_key = (signal, self.day_sentiment)
                            if _counter_key != self._last_counter_signal:
                                _cs_factor = float(
                                    (self.config.get('trading_flags', {}) or {})
                                    .get('counter_sentiment_size_factor', 0.5)
                                )
                                logging.warning(
                                    f"COUNTER-SENTIMENT: '{signal}' disagrees with "
                                    f"sentiment '{self.day_sentiment}'. Taking it at "
                                    f"reduced size (×{_cs_factor:.2f}). "
                                    f"(Suppressing repeats until this changes.)"
                                )
                                self._last_counter_signal = _counter_key

                        # ── HARD VETOES — capital protection only ─────────────
                        # PCR / OI-wall / 15m / momentum / day-quality / direction
                        # are no longer individual vetoes: they are CONFLUENCE
                        # factors in a weighted setup score (gate below). One
                        # mediocre factor can't kill a strong setup anymore —
                        # that AND-stack was why the bot sat idle.
                        if not force_mode_now and self._is_reentry_blocked(signal):
                            self._note_block("reentry_cooldown")  # already logged inside helper
                        # VIX ceiling — extreme volatility stays a hard no.
                        elif not force_mode_now and await self._is_vix_too_high():
                            self._note_block("vix_high")
                            logging.warning("Skipping entry due to VIX gate.")
                        # Trap detection — manipulative fake-breakout: hard no.
                        elif not force_mode_now and self._is_false_breakout(signal, day_df_for_signal):
                            self._note_block("trap")  # already logged inside helper
                        # ── WEIGHTED SETUP SCORE gate (replaces the AND-stack) ──
                        elif not (await self._score_gate(signal, day_df_for_signal,
                                                          is_counter_sentiment, force_mode_now)):
                            self._note_block("score_below_threshold")  # logged inside helper
                        else:
                            # Inject professional size multiplier so the order
                            # agent applies progressive loss sizing + time-of-day
                            # weighting when computing quantity.
                            tod_factor = self._time_of_day_size_factor()
                            effective_multiplier = self._trade_size_multiplier * tod_factor
                            # Counter-sentiment haircut (soft sentiment gate): trade
                            # the signal but at reduced size.
                            if is_counter_sentiment:
                                effective_multiplier *= float(
                                    (self.config.get('trading_flags', {}) or {})
                                    .get('counter_sentiment_size_factor', 0.5)
                                )
                            # Conviction sizing: HIGH = full, MEDIUM/LOW = reduced.
                            _conv_cfg = (self.config.get('conviction') or {})
                            if _conv_cfg.get('enable', True):
                                _conv_factor = float(
                                    (_conv_cfg.get('size_factors') or {})
                                    .get(self.day_conviction_level, 1.0)
                                )
                                if _conv_factor != 1.0:
                                    logging.info(
                                        f"[Conviction] {self.day_conviction_level} → size "
                                        f"×{_conv_factor:.2f}"
                                    )
                                effective_multiplier *= _conv_factor
                            # Expiry gamma scalp: extra half-size haircut (high leverage).
                            if getattr(self, '_expiry_gamma_active', False):
                                _egf = float((self.config.get('expiry_gamma_scalp') or {}).get('size_factor', 0.5))
                                effective_multiplier *= _egf
                                logging.info(f"[ExpiryGamma] 0-DTE scalp size ×{_egf:.2f}")
                            # Setup-score sizing: marginal setups trade smaller,
                            # strong confluence trades full (set by _score_gate).
                            effective_multiplier *= float(getattr(self, '_last_score_size', 1.0))
                            self.config['_effective_risk_pct_multiplier'] = effective_multiplier
                            if effective_multiplier < 1.0:
                                logging.info(
                                    f"[ProSize] Effective size multiplier: "
                                    f"{effective_multiplier:.2f} "
                                    f"(loss_factor={self._trade_size_multiplier:.2f} × "
                                    f"time_factor={tod_factor:.2f} × "
                                    f"score_factor={getattr(self, '_last_score_size', 1.0):.2f}"
                                    + (" × counter_sentiment)" if is_counter_sentiment else ")")
                                )

                            # (15m confirmation is now a weighted factor inside the
                            # setup score — it no longer hard-blocks entries.)
                            trade_details = (
                                await self.order_agent.place_trade(signal, force_mode=force_mode_now)
                                if not is_paper
                                else await self.order_agent.get_paper_trade_details(signal, force_mode=force_mode_now)
                            )
                            if trade_details:
                                trade_details['Strategy'] = self.active_strategy_name
                                # Hold-the-winner: tag high-conviction trending
                                # aligned trades so manage() rides them to ~15:15.
                                trade_details['hold_to_close'] = self._is_hold_to_close(signal)
                                trade_details['conviction'] = self.day_conviction_level
                                trade_details['setup_score'] = getattr(self, '_last_score', None)
                                trade_details['expiry_gamma'] = getattr(self, '_expiry_gamma_active', False)
                                if trade_details['hold_to_close']:
                                    logging.info(
                                        f"[HoldToClose] {signal} tagged to ride the trend "
                                        f"to near-close (HIGH conviction, TRENDING, aligned)."
                                    )
                                self.position_agent.start_trade(trade_details)
                                if not is_paper:
                                    await self.position_agent.attach_broker_stop_loss(self.order_agent)
                                self.trades_today_count += 1
                                self._score_passed_today += 1
                                # Record today as a TRADED day for the weekly
                                # frequency governor (on the first fill only).
                                if self.trades_today_count == 1:
                                    try:
                                        add_week_trade_day(
                                            datetime.date.today().strftime("%G-W%V"),
                                            datetime.date.today().isoformat(),
                                        )
                                    except Exception as _wd_exc:
                                        logging.debug(f"weekly trade-day record failed: {_wd_exc}")
                                self._persist_ledger()   # daily-max cap survives restarts
                                # Clear the \r status line before trade logs print.
                                if self._is_interactive_tty():
                                    print(flush=True)
                                self.bot_state = "IN_POSITION"
                                self.awaiting_signal_since = None
                                # Auto-disarm force mode after the first trade.
                                if self._force_mode_armed:
                                    self._force_mode_armed = False
                                    logging.warning(
                                        "FORCE-TRADE MODE disarmed: first diagnostic "
                                        "trade fired. Normal gating resumes for any "
                                        "subsequent entries this session."
                                    )

                elif self.bot_state == "IN_POSITION":
                    underlying_df_hist = await self._get_underlying_bars()

                    # 6B — Re-attach SL-M if broker rejected it (at attach time
                    # or mid-session). Re-attempt once per loop tick; the retry
                    # loop inside attach_broker_stop_loss caps total attempts.
                    active = self.position_agent.active_trade or {}
                    _slm_absent = active.get("_slm_absent", False)
                    if not is_paper and _slm_absent:
                        logging.info(
                            "[SLM-Reattach] _slm_absent=True — attempting SL-M re-attachment."
                        )
                        new_sl_id = await self.position_agent.attach_broker_stop_loss(
                            self.order_agent
                        )
                        if new_sl_id:
                            logging.info(
                                f"[SLM-Reattach] SL-M successfully re-attached: "
                                f"order_id={new_sl_id}"
                            )
                            # Clear the flag so we stop polling aggressively.
                            if self.position_agent.active_trade:
                                self.position_agent.active_trade["_slm_absent"] = False
                                self.position_agent._save_state()

                    status = await self.position_agent.manage(
                        is_paper,
                        underlying_hist_df=underlying_df_hist,
                        sentiment_agent=self.sentiment_agent,
                        gemini_api_key=self.config.get('google_api', {}).get('api_key'),
                    )
                    if isinstance(status, dict):
                        log_trade(status)
                        self._record_realized_pnl(status.get('ProfitLoss', 0))
                        # Ledger entry for the live status line + EOD summary.
                        self.completed_trades.append({
                            'symbol':   status.get('Symbol', '?'),
                            'type':     status.get('TradeType', '?'),
                            'gross':    float(status.get('GrossProfitLoss',
                                              status.get('ProfitLoss', 0)) or 0),
                            'costs':    float(status.get('Costs', 0) or 0),
                            'net':      float(status.get('ProfitLoss', 0) or 0),
                            'strategy': status.get('Strategy', '?'),
                            'regime':   getattr(self, 'day_regime', 'PENDING'),
                            'vix':      round(float(getattr(self, '_today_vix', 0) or 0), 2),
                            'iv_percentile': getattr(getattr(self, 'order_agent', None), '_last_iv_percentile', None),
                            'fii_bias': (self._fii_bias or {}).get('bias') if self._fii_bias else None,
                        })
                        self._persist_ledger()   # survive same-day restarts
                        # Record the exit so the re-entry guard can stop the bot
                        # from instantly re-buying the same direction it just got
                        # chopped out of (death-by-cuts loop).
                        self._last_exit_direction = status.get('TradeType')
                        self._last_exit_symbol = status.get('Symbol')
                        self._last_exit_time = datetime.datetime.now()
                        # On a losing trade, build a detailed post-mortem,
                        # print it to the terminal, and email it.
                        if float(status.get('ProfitLoss', 0) or 0) < 0:
                            self._handle_losing_trade(status, underlying_df_hist)
                        self.bot_state = "AWAITING_SIGNAL"
                        self.awaiting_signal_since = datetime.datetime.now()

                # 6B — When broker SL-M is absent, poll every ~2s so the
                # software stop reacts quickly to a fast adverse move.
                _active_now = self.position_agent.active_trade or {}
                _poll_fast = (
                    self.bot_state == "IN_POSITION"
                    and not is_paper
                    and _active_now.get("_slm_absent", False)
                )
                await self._aligned_sleep(max_seconds=2.0 if _poll_fast else 30.0)
            except exceptions.TokenException as e:
                logging.error(f"Zerodha session expired or invalidated: {e}. Halting bot.")
                try:
                    login_url = self.kite.login_url() if self.kite else "https://kite.zerodha.com"
                    send_token_expiry_alert(self.config, str(e), login_url)
                except Exception as _alert_exc:
                    logging.debug(f"Token-expiry alert email failed: {_alert_exc}")
                self._abort = True
                break
            except exceptions.PermissionException as e:
                # App-level config issue (IP not whitelisted, API perms missing).
                # Retrying every signal cycle won't fix it — bail out, send the
                # daily report (no trades), and surface the cause to the operator.
                logging.error(
                    f"Zerodha permission error: {e}. "
                    f"Most likely your Kite Connect app's IP whitelist is empty "
                    f"or doesn't include your current public IP. Visit "
                    f"https://developers.kite.trade/apps to fix. Halting bot."
                )
                if self._is_interactive_tty():
                    print("\n" + "!" * 78)
                    print("  Kite Connect PERMISSION ERROR — bot is halting.")
                    print(f"  {e}")
                    print("  Fix: add your public IP to the app's whitelist at")
                    print("       https://developers.kite.trade/apps")
                    print(f"       Find your IP via: curl -s ifconfig.me")
                    print("!" * 78 + "\n")
                self._abort = True
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(15)
        
        logging.info("Market is now closed. Shutting down trading loop.")
        # Report is sent by the finally block in run() — no need to call here.


if __name__ == "__main__":
    import argparse as _argparse
    import random as _rand

    _ap = _argparse.ArgumentParser(description="zAck Trading Bot")
    _ap.add_argument(
        "--manual", action="store_true",
        help=(
            "Manual strategy selection mode: you choose the strategy from a "
            "numbered menu at startup instead of the automated selector. "
            "Type 'm' + Enter while the bot is running to switch strategies "
            "mid-session. All other logic (SL, targets, risk) is unchanged."
        ),
    )
    _ap.add_argument(
        "--monitor", action="store_true",
        help=(
            "Status-only mode: publish live status to the dashboard without "
            "trading and without a Zerodha login. Runs any time (off-hours/"
            "weekends) so you can verify the bot→dashboard pipeline. Ctrl+C to stop."
        ),
    )
    _args = _ap.parse_args()

    # ── Zero-cost holiday / weekend guard ────────────────────────────────
    # Runs BEFORE config load, Zerodha auth, NewsAPI, YouTube — nothing.
    # No wasted API credits, no 60-second prompts, no email on a day off.
    _today = datetime.date.today()
    _today_str = _today.strftime("%Y-%m-%d")
    _HOLIDAY_MSGS = [
        "Aaj chutti hai bhai! Shaktimaan bhi rest karta hai kabhi kabhi 🦸",
        "NSE ne bhi OOO laga diya aaj 🏖️  Bot agrees — go touch some grass.",
        "Market holiday! Even the algo needs chai & nap time ☕ 😴",
        "NSE: Gone fishing 🎣  Come back tomorrow with fresh setups.",
        "Aaj koi trade nahi! Bot is out of office 📴  Auto-reply: Try tomorrow.",
        "Holiday mode ON 🏖️  No charts, no stress — SEBI approved.",
    ]

    # Monitor mode bypasses the weekend/holiday exit — it never trades, it only
    # publishes status, so it's useful to run any day for dashboard testing.
    if _today.weekday() >= 5 and not _args.monitor:
        _day = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][_today.weekday()]
        print("\n" + "📅 " * 19)
        print(f"  Today is {_day} — markets are closed on weekends.")
        print(f"  {_rand.choice(_HOLIDAY_MSGS)}")
        print("📅 " * 19 + "\n")
        sys.exit(0)

    if is_nse_holiday(_today) and not _args.monitor:
        _holiday_name = NSE_HOLIDAY_NAMES.get(_today_str, "Market Holiday")
        print("\n" + "🎊 " * 19)
        print(f"  🏦  {_holiday_name.upper()}  —  NSE is closed today.")
        print(f"  {_rand.choice(_HOLIDAY_MSGS)}")
        print("🎊 " * 19 + "\n")
        sys.exit(0)

    # ── Normal trading day — full startup ────────────────────────────────
    multiprocessing.freeze_support()
    bot = TradingBotOrchestrator(load_config(), manual_mode=_args.manual)
    bot._monitor_mode = _args.monitor
    if _args.monitor:
        # Status-only: no Zerodha login required.
        asyncio.run(bot.run())
    elif bot.authenticate():
        asyncio.run(bot.run())
