"""
FII (foreign institutional) derivative-positioning bias — best-effort.

The playbook reads FII net index-futures + option OI each morning as the
institutional footprint. NSE's participant-wise data is notoriously hard to
scrape reliably (cookies / anti-bot), so this module is intentionally
best-effort and NEVER gates trades — it only provides an informational bias
that's surfaced on the dashboard / journal and can break ties in the regime
direction when sentiment is neutral.

Sources, in order:
  1. A local override file (state/fii_bias.json or fii.local_file) — the robust
     path. Populate it from ANY source you trust (a separate cron that scrapes
     NSE, a paid feed, or manual entry):
         {"net_index_futures": -12000, "bias": "BEARISH", "note": "FII net short"}
  2. A best-effort NSE fetch (may be blocked; degrades to None silently).

Returns a dict {bias, net_index_futures?, note?, source} or None. Never raises.
"""
from __future__ import annotations

import logging

from infra import read_json, state_path


def _bias_from_net(net) -> str:
    try:
        n = float(net)
    except (TypeError, ValueError):
        return "NEUTRAL"
    if n > 0:
        return "BULLISH"
    if n < 0:
        return "BEARISH"
    return "NEUTRAL"


def fetch_fii_bias(config: dict) -> "dict | None":
    """Best-effort FII bias. Local file first, then NSE. None when unavailable."""
    cfg = ((config or {}).get("fii") or {})
    if not cfg.get("enable", False):
        return None

    # 1) Local override file — reliable, user/cron-populated.
    path = cfg.get("local_file") or state_path("fii_bias.json")
    data = read_json(path, default=None)
    if isinstance(data, dict):
        bias = data.get("bias") or _bias_from_net(data.get("net_index_futures"))
        if bias:
            return {
                "bias": str(bias).upper(),
                "net_index_futures": data.get("net_index_futures"),
                "note": data.get("note", ""),
                "source": data.get("source", "local_file"),
            }

    # 2) Best-effort NSE fetch (fragile — wrapped so failure is silent).
    if cfg.get("try_nse_fetch", False):
        try:
            import requests
            sess = requests.Session()
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
            # Prime cookies, then hit the participant-OI report.
            sess.get("https://www.nseindia.com", headers=headers, timeout=8)
            url = cfg.get(
                "nse_url",
                "https://www.nseindia.com/api/fiidiiTradeReact",
            )
            r = sess.get(url, headers=headers, timeout=8)
            r.raise_for_status()
            payload = r.json()
            # Shape varies; pull the FII net derivative number defensively.
            net = None
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict) and "FII" in str(row.get("category", "")).upper():
                        net = row.get("netValue") or row.get("net")
                        break
            if net is not None:
                return {"bias": _bias_from_net(net),
                        "net_index_futures": net, "note": "NSE fiidii",
                        "source": "nse"}
        except Exception as e:
            logging.debug(f"[FII] NSE fetch failed (non-fatal): {e}")

    return None
