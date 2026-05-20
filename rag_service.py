import datetime
import pandas as pd
import logging
import os
import math

class RAGService:
    """
    Handles Retrieval-Augmented Generation by creating context from
    historical trade logs and backtesting results.

    Strategy-selection RAG honours two filters from config['rag']:
      - recency_window_days: ignore trades older than N calendar days
      - min_trades_per_strategy: drop strategies with fewer than M recent trades
    Both default to sensible values when the config block is absent.
    """
    def __init__(self, config):
        self.config = config
        self.trade_log_path = 'output/trade_log.xlsx'
        self.backtest_log_path = 'output/backtest_results.csv'
        rag_cfg = (config or {}).get('rag', {}) or {}
        self.recency_window_days = int(rag_cfg.get('recency_window_days', 30))
        self.min_trades_per_strategy = int(rag_cfg.get('min_trades_per_strategy', 3))

    def _load_data(self, file_path):
        """Helper to load data from Excel or CSV."""
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return None
        try:
            if file_path.endswith('.xlsx'):
                return pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                return pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"[RAGService] Failed to read {file_path}: {e}")
            return None
        return None

    def _recent_trades_df(self) -> pd.DataFrame | None:
        """
        Returns the trade log filtered to trades within the recency window.
        Trades older than `recency_window_days` are dropped because the bot's
        behaviour materially changes between code versions — old trades are
        not a faithful sample of how the current bot trades.
        """
        df = self._load_data(self.trade_log_path)
        if df is None or df.empty:
            return None
        if 'Timestamp' not in df.columns:
            return df
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        cutoff = pd.Timestamp(datetime.datetime.now() - datetime.timedelta(days=self.recency_window_days))
        df = df[df['Timestamp'] >= cutoff]
        return df

    def compute_strategy_stats(self) -> dict:
        """
        Returns a dict of {strategy_name: stats_dict} for every strategy with at
        least `min_trades_per_strategy` trades inside the recency window.
        stats_dict keys: trades, wins, win_rate, avg_pnl, total_pnl, score.

        `score` is a small risk-adjusted ranking number used by the deterministic
        fallback when the LLM is unavailable. Higher = better.
        """
        df = self._recent_trades_df()
        if df is None or df.empty or 'Strategy' not in df.columns or 'ProfitLoss' not in df.columns:
            return {}

        out: dict = {}
        for name, group in df.groupby('Strategy'):
            n = len(group)
            if n < self.min_trades_per_strategy:
                continue
            wins = int((group['ProfitLoss'] > 0).sum())
            win_rate = wins / n * 100.0
            avg_pnl = float(group['ProfitLoss'].mean())
            total_pnl = float(group['ProfitLoss'].sum())
            std = float(group['ProfitLoss'].std() or 0.0)
            # Sharpe-like: avg/std × sqrt(n). Falls back to avg_pnl if std is zero.
            score = (avg_pnl / std) * math.sqrt(n) if std > 0 else avg_pnl
            out[name] = {
                "trades": n,
                "wins": wins,
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "total_pnl": total_pnl,
                "score": score,
            }
        return out

    def retrieve_context_for_strategy_selection(self, market_conditions: set):
        """
        Recency-windowed historical performance, suitable for the LLM prompt.
        Strategies with fewer than `min_trades_per_strategy` trades are excluded —
        small samples are noise, not signal.
        """
        logging.debug(
            f"[RAGService] Retrieving context for strategy selection "
            f"(window={self.recency_window_days}d, min_trades={self.min_trades_per_strategy})..."
        )
        stats = self.compute_strategy_stats()
        if not stats:
            return (
                f"No strategy has at least {self.min_trades_per_strategy} trades in the last "
                f"{self.recency_window_days} days; performance context is not yet meaningful."
            )

        lines = [
            f"Recent Strategy Performance (last {self.recency_window_days} days, "
            f"min {self.min_trades_per_strategy} trades):"
        ]
        # Most-confident strategies first by score.
        for name, s in sorted(stats.items(), key=lambda kv: kv[1]['score'], reverse=True):
            lines.append(
                f"- '{name}': WinRate={s['win_rate']:.1f}% over {s['trades']} trades, "
                f"AvgP/L={s['avg_pnl']:,.2f}, TotalP/L={s['total_pnl']:,.2f}, "
                f"Score={s['score']:.2f}"
            )
        context = "\n".join(lines)
        logging.debug(f"[RAGService] Generated context:\n{context}")
        return context

    def get_trading_mode(self, vix_value: float = 0.0) -> str:
        """
        Returns 'MODERATE' or 'AGGRESSIVE' based on recent trade history.

        AGGRESSIVE: last 3 of 5 trades profitable AND drawdown < 5% AND VIX < threshold.
        MODERATE:   default; also forced when 2+ consecutive losses, drawdown > 8%,
                    VIX elevated, or fewer than 10 live trades in the log.

        Reads mode_switching config block; if the block is absent or enable=false,
        always returns 'MODERATE'.
        """
        cfg = (self.config.get('mode_switching') or {})
        if not cfg.get('enable', True):
            return 'MODERATE'

        lookback     = int(cfg.get('lookback_trades', 5))
        agg_wins     = int(cfg.get('aggressive_trigger_wins', 3))
        mod_losses   = int(cfg.get('moderate_trigger_losses', 2))
        agg_max_dd   = float(cfg.get('aggressive_max_drawdown', 0.05))
        mod_dd       = float(cfg.get('moderate_drawdown', 0.08))
        mod_vix      = float(cfg.get('moderate_vix_threshold', 18.0))
        min_trades   = 10  # need live data before trusting the equity curve

        if vix_value > 0 and vix_value > mod_vix:
            logging.debug(f"[Mode] MODERATE forced: VIX {vix_value:.1f} > threshold {mod_vix:.1f}")
            return 'MODERATE'

        df = self._recent_trades_df()
        if df is None or 'ProfitLoss' not in df.columns or len(df) < min_trades:
            logging.debug(f"[Mode] MODERATE: insufficient trade history ({len(df) if df is not None else 0} < {min_trades} trades).")
            return 'MODERATE'

        # Consecutive losses at the tail of the full log.
        pnls = df['ProfitLoss'].tolist()
        consecutive_losses = 0
        for pnl in reversed(pnls):
            if float(pnl) < 0:
                consecutive_losses += 1
            else:
                break

        # Drawdown from cumulative equity peak.
        cum = df['ProfitLoss'].cumsum()
        peak = cum.cummax()
        with_peak = peak[peak > 0]
        if not with_peak.empty:
            current_dd = float((peak - cum).iloc[-1] / peak.iloc[-1]) if peak.iloc[-1] > 0 else 0.0
        else:
            current_dd = 0.0

        # Force MODERATE conditions.
        if consecutive_losses >= mod_losses:
            logging.debug(f"[Mode] MODERATE forced: {consecutive_losses} consecutive losses >= {mod_losses}.")
            return 'MODERATE'
        if current_dd > mod_dd:
            logging.debug(f"[Mode] MODERATE forced: drawdown {current_dd:.1%} > {mod_dd:.1%}.")
            return 'MODERATE'

        # Upgrade to AGGRESSIVE if recent performance warrants it.
        recent = df.tail(lookback)
        wins = int((recent['ProfitLoss'] > 0).sum())
        if wins >= agg_wins and current_dd < agg_max_dd:
            logging.debug(f"[Mode] AGGRESSIVE: {wins}/{lookback} recent wins, drawdown {current_dd:.1%}.")
            return 'AGGRESSIVE'

        logging.debug(f"[Mode] MODERATE: {wins}/{lookback} recent wins (need {agg_wins}), drawdown {current_dd:.1%}.")
        return 'MODERATE'

    def retrieve_context_for_loss_analysis(self, losing_trade_details: dict):
        """
        Retrieves context about past failures of the same strategy.
        """
        logging.info("[RAGService] Retrieving context for loss analysis...")
        strategy_name = losing_trade_details.get('Strategy')
        if not strategy_name:
            return "No strategy name provided for analysis."

        df = self._load_data(self.trade_log_path)
        if df is None:
            return "No historical trade data available to analyze."
        
        # Filter for past losing trades of the same strategy
        past_losses = df[(df['Strategy'] == strategy_name) & (df['ProfitLoss'] < 0)].tail(5)

        if past_losses.empty:
            return f"No prior losing trades found for the '{strategy_name}' strategy."

        context_lines = [f"Context on past losses for the '{strategy_name}' strategy:"]
        for _, row in past_losses.iterrows():
            rationale = row.get('Rationale', 'N/A')
            pnl = row.get('ProfitLoss', 0)
            timestamp = pd.to_datetime(row.get('Timestamp')).strftime('%Y-%m-%d')
            context_lines.append(f"- On {timestamp}, a similar trade lost {pnl:,.2f}. Rationale: {rationale}")
        
        context = "\n".join(context_lines)
        logging.info(f"[RAGService] Generated context for loss analysis:\n{context}")
        return context