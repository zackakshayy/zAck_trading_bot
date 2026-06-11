"""
Push notifications to your phone via ntfy (https://ntfy.sh).

Fire-and-forget by design: every call runs in a daemon thread, swallows every
exception, and NEVER blocks or crashes the trading path. If ntfy is disabled,
unreachable, or unconfigured, the bot behaves exactly as before.

Privacy / security notes:
  • The topic name IS the credential — use a long random string and keep it
    out of git (config.yaml is gitignored; config.example.yaml ships empty).
  • Messages deliberately contain no account numbers or API keys — only
    trade symbols, P&L and operational events.
  • Self-hosting ntfy later only needs the `server` key changed.

Config (config.yaml):
    ntfy:
      enable: true
      topic: "zack-bot-<long-random-suffix>"
      server: "https://ntfy.sh"
"""
from __future__ import annotations

import logging
import threading


def _post(server: str, topic: str, title: str, message: str,
          priority: str, tags: str) -> None:
    try:
        import requests
        requests.post(
            f"{server.rstrip('/')}/{topic}",
            data=message.encode("utf-8"),
            headers={
                "Title": title.encode("utf-8").decode("latin-1", "replace"),
                "Priority": priority,
                "Tags": tags,
            },
            timeout=6,
        )
    except Exception as e:
        logging.debug(f"[ntfy] push failed (non-fatal): {e}")


def send_push(config: dict, title: str, message: str,
              priority: str = "default", tags: str = "robot") -> None:
    """Send a push notification. Never raises, never blocks (daemon thread)."""
    try:
        cfg = (config or {}).get("ntfy") or {}
        if not cfg.get("enable", False):
            return
        topic = str(cfg.get("topic") or "").strip()
        if not topic:
            return
        server = str(cfg.get("server") or "https://ntfy.sh")
        threading.Thread(
            target=_post, args=(server, topic, title, message, priority, tags),
            daemon=True,
        ).start()
    except Exception as e:
        logging.debug(f"[ntfy] dispatch failed (non-fatal): {e}")
