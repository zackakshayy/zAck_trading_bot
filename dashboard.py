"""
zAck — mobile trading-bot dashboard (FastAPI + installable PWA).

A single-file, no-build-step progressive web app with an Apple-style UI that
works identically on iPhone (Safari → Share → Add to Home Screen) and Android
(Chrome → Install app). It reads the trading process's state files — it shares
no memory with the bot, so it can never destabilise the trading loop.

Endpoints
  /                    the app shell (no data inside — data is fetched authed)
  /api/status          live snapshot (token-authed)
  /api/logs            log tail (token-authed)
  /api/control/arm     start the bot's systemd user unit   (token + PIN)
  /api/control/kill    engage the kill switch              (token + PIN)
  /api/control/clear   disengage a pending kill switch     (token + PIN)
  /kite/login          302 → Zerodha login page (morning token flow)
  /kite/callback       Zerodha redirects here; exchanges request_token for the
                       daily access token, persists it for the bot, pushes ntfy
  /manifest.webmanifest, /sw.js, /icon-*.png, /healthz

Run (local preview):
    DASHBOARD_TOKEN=mysecret uvicorn dashboard:app --host 127.0.0.1 --port 8000
    → open http://127.0.0.1:8000/?token=mysecret

Security model
  • NETWORK is the outer wall: bind to 127.0.0.1 or the Tailscale interface
    only — never 0.0.0.0 on a public box. No port is ever exposed publicly.
  • TOKEN (env DASHBOARD_TOKEN or config dashboard.token) gates every data
    endpoint; compared with hmac.compare_digest (no timing leaks). The client
    stores it in localStorage after first entry — never in the URL afterwards.
  • PIN (config dashboard.control_pin) additionally gates the two control
    actions, with a 15-minute lockout after 5 wrong attempts.
  • /kite/callback carries no token (Zerodha's redirect can't), so it relies on
    the network wall; it never echoes secrets and the api_secret never leaves
    the server.
  • The API never returns keys, account numbers or config contents.
"""
from __future__ import annotations

import datetime
import hmac
import logging
import os
import struct
import subprocess
import time
import zlib

import yaml
from fastapi import FastAPI, Request, Query
from fastapi.responses import (HTMLResponse, JSONResponse, PlainTextResponse,
                               RedirectResponse, Response)

from infra import state_path, read_json, atomic_write_json
from notify import send_push

app = FastAPI(title="zAck", docs_url=None, redoc_url=None)

LOG_FILE = os.environ.get("BOT_LOG_FILE", "output/bot.log")


# --------------------------------------------------------------------------- #
# Config (read once; only the keys this process needs)
# --------------------------------------------------------------------------- #
def _load_config() -> dict:
    try:
        with open("config.yaml") as f:
            raw = yaml.safe_load(f) or {}
        # ${VAR} substitution for the zerodha block (mirrors trading_bot).
        def _sub(node):
            if isinstance(node, dict):
                return {k: _sub(v) for k, v in node.items()}
            if isinstance(node, list):
                return [_sub(v) for v in node]
            if isinstance(node, str) and node.startswith("${") and node.endswith("}"):
                return os.environ.get(node[2:-1], "")
            return node
        return _sub(raw)
    except Exception as e:
        logging.warning(f"[dashboard] config.yaml unreadable ({e}); running data-only.")
        return {}


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

CONFIG = _load_config()
_DASH = (CONFIG.get("dashboard") or {})
_TOKEN = (os.environ.get("DASHBOARD_TOKEN") or str(_DASH.get("token") or "")).strip()
_PIN = (os.environ.get("DASHBOARD_PIN") or str(_DASH.get("control_pin") or "")).strip()
_BOT_UNIT = str(_DASH.get("bot_service") or "zack-bot.service")
# Defence-in-depth: the unit name reaches subprocess argv — even with shell=False,
# constrain it to a sane systemd unit name so a poisoned config can't smuggle flags.
import re as _re
if not _re.fullmatch(r"[A-Za-z0-9@._-]{1,64}\.(service|timer)", _BOT_UNIT):
    logging.warning(f"[dashboard] invalid bot_service name {_BOT_UNIT!r}; controls disabled.")
    _BOT_UNIT = ""

KITE_AUTH_FILE = state_path("kite_auth.json")
KILL_FILE = state_path("kill_switch.json")


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def _ct_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def _authorized(request: Request, token: str | None) -> bool:
    if not _TOKEN:
        return True  # no token configured → open (localhost/tailnet preview only)
    for cand in (token,
                 request.headers.get("x-auth"),
                 request.headers.get("authorization", "").removeprefix("Bearer ").strip() or None):
        if cand and _ct_eq(str(cand), _TOKEN):
            return True
    return False


def _deny() -> JSONResponse:
    return JSONResponse({"error": "unauthorized"}, status_code=401)


# PIN with lockout: 5 wrong attempts → 15-minute freeze. Failure timestamps are
# PERSISTED to a state file so restarting the dashboard can't reset the counter
# (otherwise an attacker who can crash the process gets unlimited attempts).
_PIN_LOCK_FILE = state_path("dashboard_pin_lockout.json")


def _pin_fails_load() -> list[float]:
    d = read_json(_PIN_LOCK_FILE, default=None) or {}
    now = time.time()
    return [t for t in d.get("fails", []) if isinstance(t, (int, float)) and now - t < 900]


def _pin_ok(pin: str | None) -> tuple[bool, str]:
    if not _PIN:
        return False, "controls disabled — set dashboard.control_pin in config.yaml"
    fails = _pin_fails_load()
    if len(fails) >= 5:
        return False, "locked: too many wrong PINs — try again in 15 minutes"
    if pin and _ct_eq(str(pin), _PIN):
        atomic_write_json(_PIN_LOCK_FILE, {"fails": []})
        return True, ""
    fails.append(time.time())
    atomic_write_json(_PIN_LOCK_FILE, {"fails": fails})
    return False, "wrong PIN"


# --------------------------------------------------------------------------- #
# Data assembly (read-only over the bot's state files)
# --------------------------------------------------------------------------- #
def _today_str() -> str:
    return datetime.date.today().isoformat()


def _build_status() -> dict:
    status = read_json(state_path("bot_status.json"), default=None) or {}
    daily = read_json(state_path("daily_pnl.json"), default={}) or {}
    weekly = read_json(state_path("weekly_pnl.json"), default={}) or {}
    ledger = read_json(state_path(f"trade_ledger_{_today_str()}.json"), default={}) or {}

    age = None
    ts = status.get("ts")
    if ts:
        try:
            age = (datetime.datetime.now()
                   - datetime.datetime.fromisoformat(ts)).total_seconds()
        except Exception:
            age = None
    status["snapshot_age_seconds"] = round(age, 1) if age is not None else None
    status["online"] = (age is not None and age < 30)

    if not status.get("completed_trades") and ledger.get("completed_trades"):
        status["completed_trades"] = ledger["completed_trades"]
        status.setdefault("wins", ledger.get("wins", 0))
        status.setdefault("losses", ledger.get("losses", 0))
        status.setdefault("trades_today_count", ledger.get("trades_today_count", 0))

    status["daily_pnl_persisted"] = daily.get(_today_str())
    status["weekly_pnl_persisted"] = next(iter(weekly.values()), None) if weekly else None

    kill = read_json(KILL_FILE, default=None) or {}
    status["kill_pending"] = bool(kill.get("active") and kill.get("date") == _today_str())
    auth = read_json(KITE_AUTH_FILE, default=None) or {}
    status["kite_token_today"] = (auth.get("date") == _today_str())
    status["controls_available"] = bool(_PIN)
    # What config.yaml says RIGHT NOW (vs `paper`, which is what the running
    # bot loaded at startup). Differing values → "applies at next start".
    status["configured_paper"] = _configured_paper()
    return status


def _configured_paper() -> bool | None:
    """Read trading_flags.paper_trading fresh from config.yaml each call."""
    try:
        with open("config.yaml") as f:
            c = yaml.safe_load(f) or {}
        return bool((c.get("trading_flags") or {}).get("paper_trading", True))
    except Exception:
        return None


def _set_paper_mode(paper: bool) -> tuple[bool, str]:
    """
    Flip trading_flags.paper_trading in config.yaml via a single-line rewrite —
    the rest of the file (comments, formatting, secrets) stays byte-identical.
    Atomic write + post-parse verification; restores the original on any error.
    """
    import re
    try:
        with open("config.yaml") as f:
            original = f.read()
        new_text, n = re.subn(
            r"^(\s*paper_trading\s*:\s*)(true|false)\b",
            lambda m: m.group(1) + ("true" if paper else "false"),
            original, count=1, flags=re.MULTILINE,
        )
        if n != 1:
            return False, "could not find a unique paper_trading line in config.yaml"
        parsed = yaml.safe_load(new_text)
        if bool((parsed.get("trading_flags") or {}).get("paper_trading")) is not paper:
            return False, "post-edit verification failed; config.yaml left untouched"
        tmp = "config.yaml.tmp"
        with open(tmp, "w") as f:
            f.write(new_text)
        os.chmod(tmp, 0o600)   # os.replace adopts tmp's perms — keep 600
        os.replace(tmp, "config.yaml")
        return True, ""
    except Exception as e:
        return False, str(e)


def _tail_log(lines: int) -> str:
    if not os.path.exists(LOG_FILE):
        return "(no log file yet — start the bot to generate output/bot.log)"
    try:
        with open(LOG_FILE, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 64 * 1024
            data = b""
            while size > 0 and data.count(b"\n") <= lines:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
        text = data.decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])
    except Exception as e:
        return f"(log read error: {e})"


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/status")
def api_status(request: Request, token: str | None = Query(default=None)):
    if not _authorized(request, token):
        return _deny()
    return JSONResponse(_build_status())


_REDACT = None  # compiled lazily


def _redact(text: str) -> str:
    """Scrub anything credential-shaped from log output before serving it."""
    global _REDACT
    import re
    if _REDACT is None:
        _REDACT = re.compile(
            r"(access_token|request_token|api_key|api_secret|token)"
            r"(\s*[=:]\s*)([A-Za-z0-9._~+/-]{8,})", re.IGNORECASE)
    return _REDACT.sub(lambda m: m.group(1) + m.group(2) + "[redacted]", text)


@app.get("/api/logs", response_class=PlainTextResponse)
def api_logs(request: Request, token: str | None = Query(default=None),
             lines: int = Query(default=200, ge=10, le=2000)):
    if not _authorized(request, token):
        return PlainTextResponse("unauthorized", status_code=401)
    return _redact(_tail_log(lines))


@app.post("/api/control/arm")
def api_arm(request: Request, token: str | None = Query(default=None)):
    if not _authorized(request, token):
        return _deny()
    ok, why = _pin_ok(request.headers.get("x-pin"))
    if not ok:
        return JSONResponse({"error": why}, status_code=403)
    try:
        r = subprocess.run(["systemctl", "--user", "start", _BOT_UNIT],
                           capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            send_push(CONFIG, "zAck armed", "Bot service started from your phone.",
                      priority="default", tags="rocket")
            return JSONResponse({"ok": True, "message": f"{_BOT_UNIT} started"})
        return JSONResponse({"error": (r.stderr or r.stdout or "systemctl failed").strip()},
                            status_code=500)
    except FileNotFoundError:
        return JSONResponse({"error": "systemctl not available on this host "
                                      "(arm works on the Linux server)"}, status_code=501)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/control/kill")
def api_kill(request: Request, token: str | None = Query(default=None)):
    if not _authorized(request, token):
        return _deny()
    ok, why = _pin_ok(request.headers.get("x-pin"))
    if not ok:
        return JSONResponse({"error": why}, status_code=403)
    atomic_write_json(KILL_FILE, {
        "active": True, "date": _today_str(),
        "requested_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": "dashboard",
    })
    send_push(CONFIG, "zAck KILL SWITCH", "Flattening any open position and stopping.",
              priority="urgent", tags="octagonal_sign")
    return JSONResponse({"ok": True, "message": "kill switch engaged — the bot will "
                                                "flatten and stop within one tick"})


@app.post("/api/control/mode")
async def api_mode(request: Request, token: str | None = Query(default=None)):
    """Switch paper/live trading. PIN-gated; LIVE direction is the sensitive
    one (real money), but both directions require the PIN for consistency.
    Takes effect at the bot's NEXT start — the running session keeps the mode
    it booted with (the trading loop reads the flag once at startup)."""
    if not _authorized(request, token):
        return _deny()
    ok, why = _pin_ok(request.headers.get("x-pin"))
    if not ok:
        return JSONResponse({"error": why}, status_code=403)
    try:
        body = await request.json()
        paper = bool(body.get("paper"))
    except Exception:
        return JSONResponse({"error": "body must be JSON: {\"paper\": true|false}"},
                            status_code=400)
    ok, why = _set_paper_mode(paper)
    if not ok:
        return JSONResponse({"error": why}, status_code=500)
    mode = "PAPER" if paper else "LIVE"
    send_push(CONFIG, f"zAck — switched to {mode}",
              f"Trading mode set to {mode} in config. Applies when the bot "
              f"next starts.", priority="high" if not paper else "default",
              tags="warning" if not paper else "page_facing_up")
    return JSONResponse({"ok": True,
                         "message": f"{mode} mode saved — applies at the bot's next start"})


@app.post("/api/control/clear")
def api_clear_kill(request: Request, token: str | None = Query(default=None)):
    if not _authorized(request, token):
        return _deny()
    ok, why = _pin_ok(request.headers.get("x-pin"))
    if not ok:
        return JSONResponse({"error": why}, status_code=403)
    atomic_write_json(KILL_FILE, {"active": False, "date": _today_str(),
                                  "cleared_at": datetime.datetime.now().isoformat(timespec="seconds")})
    return JSONResponse({"ok": True, "message": "kill switch cleared"})


# --------------------------------------------------------------------------- #
# Zerodha morning-login flow (headless server mode)
# --------------------------------------------------------------------------- #
@app.get("/kite/login")
def kite_login():
    api_key = ((CONFIG.get("zerodha") or {}).get("api_key") or "").strip()
    if not api_key:
        return PlainTextResponse("zerodha.api_key missing in config.yaml", status_code=500)
    return RedirectResponse(f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}")


def _persist_env_token(token: str, env_path: str = ".env") -> None:
    lines = []
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("ZERODHA_ACCESS_TOKEN="):
            lines[i] = f"ZERODHA_ACCESS_TOKEN={token}\n"
            break
    else:
        lines.append(f"ZERODHA_ACCESS_TOKEN={token}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)
    os.chmod(env_path, 0o600)  # owner-only: .env holds live credentials


_RESULT_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:-apple-system,system-ui,sans-serif;background:#000;color:#fff;
display:flex;align-items:center;justify-content:center;min-height:90vh;margin:0}
.c{text-align:center;padding:32px}.i{font-size:56px}.t{font-size:22px;font-weight:600;margin:14px 0 6px}
.s{color:#98989f;font-size:15px;line-height:1.5}</style></head><body><div class="c">
<div class="i">__ICON__</div><div class="t">__TITLE__</div><div class="s">__SUB__</div>
</div></body></html>"""


def _result_html(icon: str, title: str, sub: str, code: int = 200) -> HTMLResponse:
    page = (_RESULT_PAGE.replace("__ICON__", icon)
            .replace("__TITLE__", title).replace("__SUB__", sub))
    return HTMLResponse(page, status_code=code)


@app.get("/kite/callback")
def kite_callback(request_token: str | None = None, status: str | None = None):
    z = CONFIG.get("zerodha") or {}
    if status == "cancelled" or not request_token:
        return _result_html("&#10060;", "Login cancelled",
                            "No token received from Zerodha. Close this and retry.", 400)
    if not z.get("api_key") or not z.get("api_secret"):
        return _result_html("&#9888;", "Server not configured",
                            "zerodha.api_key / api_secret missing in config.yaml.", 500)
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=z["api_key"])
        data = kite.generate_session(request_token, api_secret=z["api_secret"])
        access_token = data["access_token"]
        _persist_env_token(access_token)
        atomic_write_json(KITE_AUTH_FILE, {
            "access_token": access_token,
            "date": _today_str(),
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "user": data.get("user_name") or data.get("user_id") or "",
        })
        send_push(CONFIG, "zAck — Kite login OK",
                  "Daily token captured. The bot can trade today.", tags="white_check_mark")
        logging.info("[kite/callback] daily access token captured and persisted.")
        return _result_html("&#9989;", "You're logged in",
                            "Daily token captured — the bot is good to go. "
                            "You can close this tab.")
    except Exception as e:
        logging.error(f"[kite/callback] session exchange failed: {e}")
        return _result_html("&#10060;", "Login failed",
                            "Token exchange failed. Open the dashboard and try "
                            "Kite login again.", 500)


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


# --------------------------------------------------------------------------- #
# PWA plumbing: manifest, service worker, icons (pure-python PNG, no deps)
# --------------------------------------------------------------------------- #
def _png(width: int, bg=(10, 132, 255), fg=(255, 255, 255)) -> bytes:
    """Flat app icon: solid iOS-blue square with a geometric white 'z'."""
    s = width / 512.0
    left, right = int(144 * s), int(368 * s)
    top, bot = int(150 * s), int(362 * s)
    bar = int(46 * s)
    band = int(66 * s)
    d_top, d_bot = top + bar, bot - bar
    rows = []
    for y in range(width):
        row = bytearray()
        if top <= y < top + bar or bot - bar <= y < bot:
            in_z = lambda x: left <= x < right
        elif d_top <= y < d_bot:
            t = (y - d_top) / max(d_bot - d_top - 1, 1)
            xl = int((right - band) - t * ((right - band) - left))
            in_z = lambda x, xl=xl: xl <= x < xl + band
        else:
            in_z = lambda x: False
        for x in range(width):
            row += bytes(fg if in_z(x) else bg) + b"\xff"
        rows.append(b"\x00" + bytes(row))
    raw = zlib.compress(b"".join(rows), 9)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, width, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


_ICON_512 = _png(512)
_ICON_180 = _png(180)


@app.get("/icon-512.png")
def icon_512():
    return Response(_ICON_512, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/icon-180.png")
def icon_180():
    return Response(_ICON_180, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/manifest.webmanifest")
def manifest():
    return JSONResponse({
        "name": "zAck Trading Bot",
        "short_name": "zAck",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "background_color": "#000000",
        "theme_color": "#000000",
        "icons": [
            {"src": "/icon-180.png", "sizes": "180x180", "type": "image/png"},
            {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png",
             "purpose": "any maskable"},
        ],
    }, media_type="application/manifest+json")


_SW = r"""self.addEventListener('install',e=>self.skipWaiting());
self.addEventListener('activate',e=>e.waitUntil((async()=>{
  const keep='zack-v2';
  for(const k of await caches.keys()){if(k!==keep)await caches.delete(k);}
  await clients.claim();
})()));
self.addEventListener('fetch',e=>{
  const u=new URL(e.request.url);
  if(e.request.method!=='GET'||u.pathname.startsWith('/api/')||u.pathname.startsWith('/kite/'))return;
  e.respondWith(caches.open('zack-v2').then(async c=>{
    const hit=await c.match(e.request);
    const net=fetch(e.request).then(r=>{if(r&&r.ok)c.put(e.request,r.clone());return r;}).catch(()=>hit);
    return hit||net;
  }));
});"""


@app.get("/sw.js")
def sw():
    return Response(_SW, media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_PAGE)


# --------------------------------------------------------------------------- #
# The app shell — Apple-style, light+dark, installable on iOS and Android
# --------------------------------------------------------------------------- #
_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>zAck Trading Bot</title>
<meta name="description" content="zAck — personal NIFTY options trading terminal">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="zAck">
<meta name="theme-color" content="#0a0c0f" media="(prefers-color-scheme: dark)">
<meta name="theme-color" content="#f4f1ea" media="(prefers-color-scheme: light)">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="apple-touch-icon" href="/icon-180.png">
<style>
/* ============ Obsidian desk ============ */
*{box-sizing:border-box;margin:0;padding:0}
:root{
  color-scheme:dark;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --bg:#0a0c0f;
  --surface:#101317;
  --surface2:#14181d;
  --surface3:#1a1f25;
  --ink:#e8ebee;
  --ink2:#a6adb4;
  --ink3:#70787f;
  --line:rgba(232,235,238,.07);
  --line2:rgba(232,235,238,.15);
  --pos:#3eb585;
  --pos-soft:rgba(62,181,133,.10);
  --pos-line:rgba(62,181,133,.38);
  --pos-btn:#1e7c55;
  --neg:#cd6b72;
  --neg-soft:rgba(205,107,114,.10);
  --neg-line:rgba(205,107,114,.38);
  --neg-btn:#a8454e;
  --gold:#c2a26a;
  --glow-flat:rgba(232,235,238,.045);
  --glow-pos:rgba(62,181,133,.07);
  --glow-neg:rgba(205,107,114,.06);
  --shadow:none;
  --logbg:#0c0e11;
  --toastbg:#1e242b;
  --toastink:#e8ebee;
  --press:1.16;
  --pulseA:rgba(62,181,133,.4);
}
@media (prefers-color-scheme: light){
  :root{
    color-scheme:light;
    --bg:#f4f1ea;
    --surface:#fcfaf5;
    --surface2:#f3efe7;
    --surface3:#ebe7dc;
    --ink:#22262a;
    --ink2:#595f66;
    --ink3:#82888e;
    --line:rgba(34,38,42,.09);
    --line2:rgba(34,38,42,.17);
    --pos:#17714c;
    --pos-soft:rgba(23,113,76,.08);
    --pos-line:rgba(23,113,76,.35);
    --pos-btn:#1b7550;
    --neg:#a23b43;
    --neg-soft:rgba(162,59,67,.08);
    --neg-line:rgba(162,59,67,.35);
    --neg-btn:#a23b43;
    --gold:#9a7a42;
    --glow-flat:rgba(34,38,42,.035);
    --glow-pos:rgba(23,113,76,.06);
    --glow-neg:rgba(162,59,67,.05);
    --shadow:0 1px 2px rgba(44,38,28,.05);
    --logbg:#f0ece1;
    --toastbg:#24272b;
    --toastink:#f2f0ea;
    --press:.95;
    --pulseA:rgba(23,113,76,.35);
  }
}
html{-webkit-text-size-adjust:100%}
body{
  font-family:var(--sans);
  background:var(--bg);
  color:var(--ink);
  min-height:100dvh;
  line-height:1.4;
  -webkit-tap-highlight-color:transparent;
  overscroll-behavior-y:contain;
  -webkit-font-smoothing:antialiased;
}
[hidden]{display:none!important}
button{font:inherit;color:inherit;-webkit-tap-highlight-color:transparent}
:focus-visible{outline:2px solid var(--ink3);outline-offset:2px;border-radius:4px}
::selection{background:var(--surface3)}

/* ---- header ---- */
header.top{
  position:sticky;top:0;z-index:30;
  background:var(--bg);
  border-bottom:1px solid var(--line);
  padding:calc(env(safe-area-inset-top) + 10px) 0 10px;
}
.hrow{max-width:600px;margin:0 auto;padding:0 14px;display:flex;align-items:center;gap:10px}
.brand{display:flex;align-items:center;gap:9px;user-select:none}
.mark{width:26px;height:26px;color:var(--ink);flex:none;display:block}
.bname{font-size:16px;font-weight:750;letter-spacing:.01em;line-height:1.05}
.bsub{font-size:8.5px;letter-spacing:.22em;color:var(--ink3);font-weight:650;margin-top:2px}
.hspace{flex:1}
.conn{display:flex;align-items:center;gap:7px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ink3);flex:none}
.dot.run{background:var(--pos)}
.dot.idle{background:var(--ink3)}
.dot.off{background:transparent;border:1.5px solid var(--neg)}
.clbl{font-size:9.5px;letter-spacing:.14em;font-weight:750;color:var(--ink2)}
.fresh{font-size:10.5px;color:var(--ink3);font-family:var(--mono);font-variant-numeric:tabular-nums;min-width:84px;text-align:right}

/* ---- layout ---- */
.wrap{max-width:600px;margin:0 auto;padding:0 14px calc(env(safe-area-inset-bottom) + 44px)}
main{display:flex;flex-direction:column;gap:12px;padding-top:14px}
#banners{display:flex;flex-direction:column;gap:12px}
#banners:empty{display:none}
.card{
  background:var(--surface);
  border:1px solid var(--line);
  border-radius:14px;
  padding:16px;
  box-shadow:var(--shadow);
}
.lbl{font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--ink3);font-weight:650}
.lblrow{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:12px}
.lblrow .lbl{margin:0}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.hint{font-size:12px;color:var(--ink3);line-height:1.55;margin-top:9px}
.cpos{color:var(--pos)}
.cneg{color:var(--neg)}
.foot{text-align:center;font-size:9.5px;letter-spacing:.2em;text-transform:uppercase;color:var(--ink3);padding:16px 0 4px;user-select:none}

/* ---- hero ---- */
.hero{
  text-align:center;
  padding:20px 16px 16px;
  --glow:var(--glow-flat);
  background:radial-gradient(130% 95% at 50% 0%, var(--glow), transparent 62%), var(--surface);
}
.hero.pos{--glow:var(--glow-pos)}
.hero.neg{--glow:var(--glow-neg)}
.hero::before{content:"";display:block;width:52px;height:1px;margin:0 auto 14px;background:var(--gold);opacity:.78}
.heroNum{
  font-family:var(--mono);
  font-size:44px;font-weight:600;
  letter-spacing:-.02em;line-height:1.05;
  font-variant-numeric:tabular-nums;
  margin-top:9px;
  min-height:48px;
}
.hero.pos .heroNum{color:var(--pos)}
.hero.neg .heroNum{color:var(--neg)}
.heroWeek{margin-top:8px;font-size:12px;color:var(--ink3)}
.hstats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;border-top:1px solid var(--line);margin-top:18px;padding-top:13px}
.hstat .lbl{margin-bottom:5px}
.hstat .val{font-family:var(--mono);font-size:13.5px;font-weight:600;font-variant-numeric:tabular-nums}

/* ---- pills / badges / chips ---- */
.pill{
  font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;font-weight:750;
  padding:4px 9px;border-radius:99px;
  border:1px solid var(--line2);color:var(--ink2);
  white-space:nowrap;
}
.pill.live{color:var(--neg);border-color:var(--neg-line);background:var(--neg-soft)}
.pill.exp{color:var(--neg);border-color:var(--neg-line);background:var(--neg-soft)}
.bdg{font-size:10px;font-weight:750;letter-spacing:.1em;padding:2px 7px;border-radius:6px;border:1px solid var(--line2);color:var(--ink2)}
.bdg.ce{color:var(--pos);border-color:var(--pos-line);background:var(--pos-soft)}
.bdg.pe{color:var(--neg);border-color:var(--neg-line);background:var(--neg-soft)}
.chip{
  font-family:var(--mono);font-size:12px;font-variant-numeric:tabular-nums;
  padding:3px 8px;border-radius:7px;border:1px solid var(--line2);color:var(--ink2);
}
.chip.res{color:var(--neg);border-color:var(--neg-line);background:var(--neg-soft)}
.chip.sup{color:var(--pos);border-color:var(--pos-line);background:var(--pos-soft)}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.nochip{color:var(--ink3);font-size:12px}

/* ---- buttons ---- */
.btn{
  display:inline-flex;align-items:center;justify-content:center;gap:8px;
  width:100%;height:48px;padding:0 18px;
  border-radius:12px;border:1px solid var(--line2);
  background:var(--surface3);color:var(--ink);
  font-weight:650;font-size:15px;cursor:pointer;user-select:none;
}
.btn:disabled{opacity:.42;cursor:default}
.btn:active:not(:disabled){filter:brightness(var(--press))}
.btn-pos{background:var(--pos-btn);border-color:transparent;color:#f0fbf6}
.btn-neg{background:var(--neg-soft);border-color:var(--neg-line);color:var(--neg)}
.btn-negsolid{background:var(--neg-btn);border-color:transparent;color:#fdf3f3}
.btn-ink{background:var(--ink);border-color:transparent;color:var(--bg)}
.btn-ghost{background:transparent;border-color:var(--line2);color:var(--ink2)}
.btn-sm{width:auto;height:34px;font-size:12.5px;border-radius:9px;padding:0 13px;flex:none}

/* ---- mode segment ---- */
.seg{display:flex;background:var(--surface3);border:1px solid var(--line);border-radius:11px;padding:3px;gap:3px}
.seg button{
  flex:1;height:38px;border:1px solid transparent;background:transparent;border-radius:8px;
  color:var(--ink3);font-weight:650;font-size:13px;letter-spacing:.07em;text-transform:uppercase;cursor:pointer;
}
.seg button.on{background:var(--surface);color:var(--ink);border-color:var(--line2)}
.seg button.live.on{color:var(--neg)}
.seg button:disabled{opacity:.45;cursor:default}

/* ---- banners ---- */
.banner{display:flex;align-items:center;gap:12px;border-radius:12px;padding:12px 14px;border:1px solid var(--line);background:var(--surface)}
.banner .btxt{flex:1;min-width:0}
.banner b{display:block;font-size:13.5px;font-weight:700}
.banner span{display:block;font-size:12px;color:var(--ink2);margin-top:1px;line-height:1.45}
.b-kill{background:var(--neg-soft);border-color:var(--neg-line)}
.b-kill b{color:var(--neg)}
.b-warn b{color:var(--ink2)}

/* ---- flow / steps ---- */
.runrow{display:flex;align-items:center;gap:13px;padding:2px 0}
.runDot{width:10px;height:10px;border-radius:50%;background:var(--pos);flex:none}
.runTitle{font-size:15px;font-weight:750}
.runSub{font-size:12.5px;color:var(--ink2);margin-top:2px}
.step{display:flex;gap:12px;align-items:flex-start}
.step + .step{margin-top:15px}
.snum{
  width:22px;height:22px;border-radius:50%;border:1px solid var(--line2);
  display:flex;align-items:center;justify-content:center;flex:none;margin-top:11px;
  font-family:var(--mono);font-size:11px;color:var(--ink2);
}
.step.done .snum{background:var(--pos-soft);border-color:var(--pos-line);color:var(--pos)}
.sbody{flex:1;min-width:0}
.sbody .hint{margin-top:7px}
.tokenok{display:flex;gap:8px;align-items:center;color:var(--pos);font-size:13px;font-weight:650;margin-bottom:12px}
.offmsg{font-size:13px;color:var(--ink2);line-height:1.5}

/* ---- stat cells ---- */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:13px 10px}
.cell .lbl{margin-bottom:4px}
.cell .val{font-size:13.5px;font-weight:550;color:var(--ink);overflow-wrap:anywhere}
.cell .val.num{font-weight:600;font-size:14px}

/* ---- market extras ---- */
.mhdr{display:flex;gap:8px;align-items:center}
.snap{font-size:10px;color:var(--ink3)}
.walls{border-top:1px solid var(--line);margin-top:15px;padding-top:13px;display:flex;flex-direction:column;gap:11px}
.wallrow .lbl{margin-bottom:6px}
.funnel{display:flex;align-items:center;gap:10px;border-top:1px solid var(--line);margin-top:15px;padding-top:13px}
.fcell{flex:1}
.fcell .lbl{margin-bottom:4px}
.fcell .val{font-family:var(--mono);font-size:15px;font-weight:650;font-variant-numeric:tabular-nums}
.farr{color:var(--ink3);font-size:12px;flex:none;padding-top:11px}

/* ---- position ---- */
.posSym{font-size:16px;font-weight:700;letter-spacing:.01em;margin:2px 0 14px}
.scale{position:relative;height:26px;margin-top:22px}
.sc-track{position:absolute;left:0;right:0;top:50%;height:4px;margin-top:-2px;border-radius:99px;background:var(--surface3)}
.sc-fill{position:absolute;left:0;top:50%;height:4px;margin-top:-2px;border-radius:99px;background:var(--pos);opacity:.32}
.sc-entry{position:absolute;top:50%;width:2px;height:14px;margin-top:-7px;background:var(--ink3);transform:translateX(-50%);border-radius:1px}
.sc-trail{position:absolute;top:50%;width:3px;height:17px;margin-top:-8.5px;border-radius:2px;background:var(--pos);transform:translateX(-50%)}
.sc-tlab{position:absolute;top:-10px;transform:translateX(-50%);font-size:9.5px;color:var(--pos);font-family:var(--mono);white-space:nowrap;font-variant-numeric:tabular-nums}
.sc-ends{display:flex;justify-content:space-between;margin-top:3px}
.target{margin-top:15px;padding-top:13px;border-top:1px solid var(--line)}
.target .lbl{margin-bottom:5px}
.tval{display:flex;align-items:baseline;gap:9px}
.tval .num{font-size:16px;font-weight:650}
.tlbl{font-size:12px;color:var(--ink2)}

/* ---- why no trade ---- */
.verdict{
  display:flex;gap:10px;align-items:baseline;
  background:var(--surface2);border:1px solid var(--line);border-left:2px solid var(--ink2);
  border-radius:9px;padding:10px 12px;margin-bottom:10px;
  font-size:13.5px;font-weight:550;line-height:1.5;
}
.vtag{font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:var(--ink3);font-weight:750;flex:none;position:relative;top:-1px}
.why-item{display:flex;gap:9px;font-size:13px;color:var(--ink2);padding:4.5px 0;line-height:1.5}
.why-item::before{content:"–";color:var(--ink3);flex:none}

/* ---- trades ---- */
.trade{border-top:1px solid var(--line);padding:12px 2px;cursor:pointer}
.trade:first-child{border-top:none;padding-top:2px}
.trade:last-child{padding-bottom:2px}
.trow{display:flex;align-items:center;gap:10px}
.tleft{flex:1;min-width:0}
.tsymrow{display:flex;align-items:center;gap:8px}
.tsym{font-weight:650;font-size:13.5px}
.tmeta{font-size:11px;color:var(--ink3);margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tnet{font-weight:650;font-size:14px;flex:none}
.chev{color:var(--ink3);font-size:11px;flex:none}
.trade.open .chev{transform:rotate(180deg)}
.tx{display:none;margin-top:11px;background:var(--surface2);border:1px solid var(--line);border-radius:10px;padding:9px 12px}
.trade.open .tx{display:block}
.tx .row{display:flex;justify-content:space-between;font-size:12.5px;padding:3.5px 0;color:var(--ink2)}
.tx .row .num{font-size:12.5px}
.tx .rownet{border-top:1px solid var(--line);margin-top:3px;padding-top:7px;color:var(--ink);font-weight:600}

/* ---- logs ---- */
.logsBtn{display:flex;width:100%;align-items:center;justify-content:space-between;background:none;border:none;padding:0;cursor:pointer;font-weight:650;font-size:14px}
.logsBtn .chev2{color:var(--ink3);font-size:12px}
.logsBtn .chev2.up{transform:rotate(180deg)}
#logsBody{margin-top:13px}
.logbox{background:var(--logbg);border:1px solid var(--line);border-radius:10px;max-height:340px;overflow:auto;padding:10px 12px;overscroll-behavior:contain}
.logbox pre{font-family:var(--mono);font-size:10.5px;line-height:1.6;white-space:pre-wrap;word-break:break-word;color:var(--ink2)}

/* ---- sheet ---- */
#sheet{position:fixed;inset:0;z-index:60;display:none}
#sheet.open{display:block}
.sheetBack{position:absolute;inset:0;background:rgba(0,0,0,.55)}
.sheetPanel{
  position:absolute;left:0;right:0;bottom:0;max-width:600px;margin:0 auto;
  background:var(--surface);border:1px solid var(--line2);border-bottom:none;
  border-radius:18px 18px 0 0;
  padding:14px 18px calc(env(safe-area-inset-bottom) + 20px);
}
.sheetGrab{width:36px;height:4px;border-radius:99px;background:var(--line2);margin:0 auto 16px}
#sheetTitle{font-size:17px;font-weight:750}
#sheetBody{font-size:13.5px;color:var(--ink2);line-height:1.55;margin:8px 0 16px}
.pinInput{
  width:100%;height:52px;border-radius:12px;border:1px solid var(--line2);
  background:var(--surface3);color:var(--ink);
  font-family:var(--mono);font-size:20px;letter-spacing:.45em;text-align:center;outline:none;
}
.pinInput::placeholder{letter-spacing:.18em;font-size:13px;color:var(--ink3)}
.pinInput:focus{border-color:var(--ink3)}
#sheetErr{color:var(--neg);font-size:12.5px;margin-top:9px;min-height:17px;line-height:1.4}
.sheetBtns{display:flex;gap:10px;margin-top:12px}
.sheetBtns .btn-ghost{flex:1}
#sheetGo{flex:1.7}

/* ---- gate ---- */
#gate{position:fixed;inset:0;z-index:80;background:var(--bg);display:none;align-items:center;justify-content:center;padding:24px}
#gate.open{display:flex}
.gateBox{width:100%;max-width:320px;text-align:center}
.gateBox .mark{width:44px;height:44px;margin:0 auto 14px}
.gateTitle{font-size:19px;font-weight:750}
.gateSub{font-size:12.5px;color:var(--ink3);margin:7px 0 20px;line-height:1.5}
.tokInput{
  width:100%;height:48px;border-radius:12px;border:1px solid var(--line2);
  background:var(--surface3);color:var(--ink);
  font-family:var(--mono);font-size:16px;text-align:center;outline:none;margin-bottom:12px;
}
.tokInput:focus{border-color:var(--ink3)}
#gateErr{color:var(--neg);font-size:12px;margin-top:11px}

/* ---- toasts ---- */
#toasts{
  position:fixed;left:0;right:0;bottom:calc(env(safe-area-inset-bottom) + 14px);
  display:flex;flex-direction:column;gap:8px;align-items:center;z-index:70;
  pointer-events:none;padding:0 16px;
}
.toast{
  pointer-events:auto;display:flex;gap:10px;align-items:center;
  background:var(--toastbg);color:var(--toastink);border:1px solid rgba(255,255,255,.09);
  border-radius:11px;padding:11px 15px;font-size:13px;font-weight:550;
  max-width:560px;box-shadow:0 8px 28px rgba(0,0,0,.3);line-height:1.4;
}
.tdot{width:7px;height:7px;border-radius:50%;flex:none;background:var(--ink3)}
.t-ok .tdot{background:var(--pos)}
.t-err .tdot{background:var(--neg)}

/* ---- skeleton ---- */
#skel{display:flex;flex-direction:column;gap:12px;padding-top:14px}
.sk{background:var(--surface2);border:1px solid var(--line);border-radius:14px}

/* ---- motion (opt-in only) ---- */
@media (prefers-reduced-motion: no-preference){
  .dot.run,.runDot{animation:pulse 2.4s ease-out infinite}
  @keyframes pulse{0%{box-shadow:0 0 0 0 var(--pulseA)}75%{box-shadow:0 0 0 8px transparent}100%{box-shadow:0 0 0 0 transparent}}
  .sk{animation:skp 1.5s ease-in-out infinite}
  @keyframes skp{50%{opacity:.55}}
  .btn{transition:transform .09s ease,filter .12s ease}
  .btn:active:not(:disabled){transform:scale(.98)}
  .seg button{transition:color .15s ease,background-color .15s ease}
  .chev,.chev2{transition:transform .18s ease}
  #sheet.open .sheetPanel{animation:sheetUp .28s cubic-bezier(.32,.72,.27,1)}
  #sheet.open .sheetBack{animation:fadeIn .22s ease-out}
  @keyframes sheetUp{from{transform:translateY(48px);opacity:.4}to{transform:translateY(0);opacity:1}}
  @keyframes fadeIn{from{opacity:0}to{opacity:1}}
  .toast{animation:toastIn .24s ease-out}
  @keyframes toastIn{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:translateY(0)}}
  .tx{animation:fadeIn .18s ease-out}
}

/* ---- desktop composure ---- */
@media (min-width:700px){
  .heroNum{font-size:52px}
  main{gap:14px}
  .card{padding:20px}
  .wrap,.hrow{padding-left:0;padding-right:0}
}
</style>
</head>
<body>

<!-- token gate -->
<div id="gate" role="dialog" aria-modal="true" aria-label="Access token required">
  <div class="gateBox">
    <svg class="mark" viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="14" fill="none" stroke="var(--gold)" stroke-width="1.1" opacity=".9"/>
      <path d="M10.8 11.1h10.4L10.8 20.9h10.4" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <div class="gateTitle">zAck Trading Bot</div>
    <div class="gateSub">This terminal is private. Enter your access token to continue.</div>
    <input id="gateInput" class="tokInput" type="password" autocomplete="current-password" autocapitalize="off" autocorrect="off" spellcheck="false" placeholder="Access token" aria-label="Access token">
    <button id="gateBtn" type="button" class="btn btn-ink">Unlock</button>
    <div id="gateErr" hidden>Token rejected — check and try again.</div>
  </div>
</div>

<header class="top">
  <div class="hrow">
    <div class="brand">
      <svg class="mark" viewBox="0 0 32 32" aria-hidden="true">
        <circle cx="16" cy="16" r="14" fill="none" stroke="var(--gold)" stroke-width="1.1" opacity=".9"/>
        <path d="M10.8 11.1h10.4L10.8 20.9h10.4" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <div>
        <div class="bname">zAck</div>
        <div class="bsub">TRADING BOT</div>
      </div>
    </div>
    <div class="hspace"></div>
    <div class="conn"><span class="dot idle" id="dot"></span><span class="clbl" id="dotLbl">&nbsp;</span></div>
    <span class="fresh" id="fresh"></span>
  </div>
</header>

<div class="wrap">

  <!-- skeleton first paint -->
  <div id="skel" aria-hidden="true">
    <div class="sk" style="height:172px"></div>
    <div class="sk" style="height:118px"></div>
    <div class="sk" style="height:96px"></div>
    <div class="sk" style="height:236px"></div>
    <div class="sk" style="height:84px"></div>
  </div>

  <main id="app" hidden>
    <div id="banners"></div>

    <!-- P&L hero -->
    <section class="card hero" id="hero" aria-label="Profit and loss today">
      <div class="lbl" id="heroLabel">P&amp;L · today</div>
      <div class="heroNum" id="heroNum">—</div>
      <div class="heroWeek num" id="heroWeek" hidden></div>
      <div class="hstats">
        <div class="hstat"><div class="lbl">Record</div><div class="val" id="stWL">—</div></div>
        <div class="hstat"><div class="lbl">Trades</div><div class="val" id="stTrades">—</div></div>
        <div class="hstat"><div class="lbl" id="stBalLbl">Balance</div><div class="val" id="stBal">—</div></div>
      </div>
    </section>

    <!-- open position -->
    <section class="card" id="posCard" hidden aria-label="Open position"></section>

    <!-- morning flow / running -->
    <section class="card" id="flow" aria-label="Bot control"></section>

    <!-- mode -->
    <section class="card" id="modeCard" aria-label="Execution mode">
      <div class="lblrow">
        <span class="lbl">Execution mode</span>
        <span class="pill" id="modePending" hidden></span>
      </div>
      <div class="seg" role="group" aria-label="Paper or live mode">
        <button type="button" id="segPaper" data-act="mode-paper">Paper</button>
        <button type="button" id="segLive" class="live" data-act="mode-live">Live</button>
      </div>
      <p class="hint" id="modeHint">Live places real orders. A change applies at the bot&rsquo;s next start.</p>
    </section>

    <!-- market -->
    <section class="card" id="marketCard" aria-label="Market panel"></section>

    <!-- why no trade -->
    <section class="card" id="whyCard" hidden aria-label="Why no trade yet"></section>

    <!-- closed trades -->
    <section class="card" id="tradesCard" hidden aria-label="Closed trades">
      <div class="lblrow">
        <span class="lbl">Closed trades</span>
        <span class="pill" id="tradesCount"></span>
      </div>
      <div id="tradesList"></div>
    </section>

    <!-- logs -->
    <section class="card" aria-label="Session logs">
      <button type="button" class="logsBtn" id="logsBtn" data-act="logs" aria-expanded="false" aria-controls="logsBody">
        <span>Session logs</span><span class="chev2" id="logsChev" aria-hidden="true">▾</span>
      </button>
      <div id="logsBody" hidden>
        <div class="logbox" id="logsBox"><pre id="logsPre">—</pre></div>
      </div>
    </section>

    <!-- risk controls -->
    <section class="card" id="riskCard" aria-label="Risk controls">
      <div class="lblrow"><span class="lbl">Risk controls</span></div>
      <button type="button" class="btn btn-neg" id="killBtn" data-act="kill">Kill switch — flatten &amp; stop</button>
      <p class="hint">Closes any open position at market and stops the bot. PIN required.</p>
      <p class="hint" id="ctrlNote" hidden>Controls are unavailable from this host.</p>
    </section>

    <footer class="foot">zAck · private terminal</footer>
  </main>
</div>

<div id="toasts" role="status" aria-live="polite"></div>

<!-- PIN sheet -->
<div id="sheet" role="dialog" aria-modal="true" aria-labelledby="sheetTitle">
  <div class="sheetBack" id="sheetBack"></div>
  <div class="sheetPanel">
    <div class="sheetGrab" aria-hidden="true"></div>
    <h2 id="sheetTitle">—</h2>
    <p id="sheetBody">—</p>
    <input id="sheetPin" class="pinInput" type="password" inputmode="numeric" pattern="[0-9]*" autocomplete="one-time-code" maxlength="10" placeholder="PIN" aria-label="Control PIN">
    <div id="sheetErr" role="alert"></div>
    <div class="sheetBtns">
      <button type="button" class="btn btn-ghost" id="sheetCancel">Cancel</button>
      <button type="button" class="btn btn-ink" id="sheetGo">Confirm</button>
    </div>
  </div>
</div>

<noscript><div style="padding:24px;text-align:center;font-family:sans-serif">zAck needs JavaScript to run.</div></noscript>

<script>
'use strict';
/* ================= helpers ================= */
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const inr0 = new Intl.NumberFormat('en-IN', {maximumFractionDigits:0});
const inr2 = new Intl.NumberFormat('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2});
const RM = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : {matches:false};

function money(v, dp, signed){
  if(v == null || !isFinite(v)) return '—';
  const f = dp === 2 ? inr2 : inr0;
  const s = v < 0 ? '−' : (signed && v > 0 ? '+' : '');
  return s + '₹' + f.format(Math.abs(v));
}
function rupee2(v){ return (v == null || !isFinite(v)) ? '—' : '₹' + inr2.format(v); }
function fmtN(v, dp){ return (v == null || !isFinite(v)) ? '—' : Number(v).toFixed(dp); }
function fmtAge(v){
  if(v == null || !isFinite(v)) return '—';
  if(v < 60) return Math.round(v) + 's';
  if(v < 3600) return Math.round(v/60) + 'm';
  return (v/3600).toFixed(1) + 'h';
}
function pretty(v){
  if(v == null || v === '') return '—';
  const s = String(v).replace(/_/g,' ').trim();
  if(!s) return '—';
  if(/[a-z]/.test(s)) return esc(s);
  const low = s.toLowerCase();
  return esc(low.charAt(0).toUpperCase() + low.slice(1));
}
function setText(el, v){ if(el.textContent !== v) el.textContent = v; }
function setSection(el, key, html){
  if(el.dataset.k === key) return false;
  el.dataset.k = key;
  el.innerHTML = (typeof html === 'function') ? html() : html;
  return true;
}
function cell(label, value, extra){
  return '<div class="cell"><div class="lbl">' + label + '</div><div class="val ' + (extra || '') + '">' + value + '</div></div>';
}

/* ================= state ================= */
let token = localStorage.getItem('zt') || '';
let st = null;            // last /api/status payload
let reachable = true;     // did last fetch succeed
let lastOkAt = 0;
let firstPaint = false;
let gated = false;
let gateAttempted = false;
let sheetCtx = null;
let sheetBusy = false;
const expanded = new Set();
let logsOpen = false, logsTimer = null;
let heroVal = null, heroAnim = null;

/* token bootstrap from ?token= */
(function(){
  try{
    const u = new URL(location.href);
    const t = u.searchParams.get('token');
    if(t){
      token = t;
      localStorage.setItem('zt', t);
      u.searchParams.delete('token');
      const qs = u.searchParams.toString();
      history.replaceState(null, '', u.pathname + (qs ? '?' + qs : '') + u.hash);
    }
  }catch(e){}
})();

function authHeaders(extra){
  const h = {'X-Auth': token};
  if(extra) for(const k in extra) h[k] = extra[k];
  return h;
}
function ctrlOK(){ return !(st && st.controls_available === false); }
function cfgPaper(){
  if(!st) return true;
  if(st.configured_paper != null) return !!st.configured_paper;
  if(st.paper != null) return !!st.paper;
  return true;
}

/* ================= polling ================= */
async function poll(){
  if(gated) return;
  if(document.hidden && firstPaint) return;
  const ctl = new AbortController();
  const tm = setTimeout(() => ctl.abort(), 3500);
  try{
    const r = await fetch('/api/status', {headers: authHeaders(), cache:'no-store', signal: ctl.signal});
    clearTimeout(tm);
    if(r.status === 401){ reveal(); openGate(); return; }
    if(!r.ok) throw new Error('http ' + r.status);
    st = await r.json();
    reachable = true;
    lastOkAt = Date.now();
  }catch(e){
    clearTimeout(tm);
    reachable = false;
  }
  reveal();
  renderAll();
}
function reveal(){
  if(firstPaint) return;
  firstPaint = true;
  $('#skel').hidden = true;
  $('#app').hidden = false;
}

/* ================= render ================= */
function renderAll(){
  const online = !!(st && st.online);
  document.body.classList.toggle('offline', !online);

  const dot = $('#dot');
  const cls = reachable ? (online ? 'dot run' : 'dot idle') : 'dot off';
  if(dot.className !== cls) dot.className = cls;
  setText($('#dotLbl'), reachable ? (online ? 'RUNNING' : 'STOPPED') : 'OFFLINE');

  renderBanners();
  renderHero(online);
  renderPos();
  renderFlow(online);
  renderMode(online);
  renderMarket();
  renderWhy();
  renderTrades();
  renderRisk();
}

function renderBanners(){
  let h = '';
  if(!reachable){
    h += '<div class="banner b-warn"><div class="btxt"><b>Server unreachable</b><span>Retrying every few seconds — figures below may be stale.</span></div></div>';
  }
  if(st && st.kill_pending){
    h += '<div class="banner b-kill"><div class="btxt"><b>Kill pending</b><span>The bot is flagged to stop and will not trade.</span></div>' +
         '<button type="button" class="btn btn-sm" data-act="clear"' + (ctrlOK() ? '' : ' disabled') + '>Clear kill</button></div>';
  }
  setSection($('#banners'), h, h);
}

function renderHero(online){
  let effPaper = null;
  if(st){
    effPaper = online
      ? (st.paper != null ? st.paper : st.configured_paper)
      : (st.configured_paper != null ? st.configured_paper : st.paper);
  }
  const label = (effPaper == null ? 'P&L' : (effPaper ? 'Paper P&L' : 'Real P&L')) + ' · today';
  setText($('#heroLabel'), label);

  let v = null;
  if(st){
    v = online
      ? st.realized_pnl_today
      : (st.daily_pnl_persisted != null ? st.daily_pnl_persisted : st.realized_pnl_today);
  }
  setHero(v == null || !isFinite(v) ? null : Number(v));

  const wk = st ? st.weekly_pnl_persisted : null;
  const wkEl = $('#heroWeek');
  if(wk != null && isFinite(wk)){
    wkEl.hidden = false;
    setText(wkEl, 'Week ' + money(wk, 0, true));
  } else { wkEl.hidden = true; }

  const w = st ? st.wins : null, l = st ? st.losses : null, sc = st ? st.scratch : null;
  const rec = (w == null && l == null && sc == null) ? '—'
    : '<span class="cpos">' + (w ?? 0) + 'W</span> · <span class="cneg">' + (l ?? 0) + 'L</span> · <span>' + (sc ?? 0) + 'S</span>';
  const wlEl = $('#stWL');
  if(wlEl.dataset.k !== rec){ wlEl.dataset.k = rec; wlEl.innerHTML = rec; }

  const tc = st ? st.trades_today_count : null, mt = st ? st.max_trades : null;
  setText($('#stTrades'), (tc == null ? '—' : tc) + ' / ' + (mt == null ? '—' : mt));

  setText($('#stBal'), st ? money(st.balance, 0) : '—');
  setText($('#stBalLbl'), (st && st.balance_label) ? String(st.balance_label) : 'Balance');
}

function setHero(v){
  const el = $('#heroNum');
  const hero = $('#hero');
  hero.classList.toggle('pos', v != null && v > 0);
  hero.classList.toggle('neg', v != null && v < 0);
  if(v == null){
    if(heroAnim) cancelAnimationFrame(heroAnim);
    heroVal = null; el.textContent = '—'; return;
  }
  const target = Math.round(v);
  if(heroVal === null || RM.matches || Math.abs(target - heroVal) < 1){
    if(heroAnim) cancelAnimationFrame(heroAnim);
    heroVal = target; el.textContent = money(target, 0, true); return;
  }
  if(heroVal === target) return;
  if(heroAnim) cancelAnimationFrame(heroAnim);
  const from = heroVal, to = target, t0 = performance.now(), dur = 480;
  const step = t => {
    const p = Math.min(1, (t - t0) / dur);
    const e = 1 - Math.pow(1 - p, 3);
    const cur = Math.round(from + (to - from) * e);
    el.textContent = money(cur, 0, true);
    if(p < 1){ heroAnim = requestAnimationFrame(step); }
    else { heroVal = to; heroAnim = null; }
  };
  heroAnim = requestAnimationFrame(step);
}

function renderFlow(online){
  const el = $('#flow');
  const kite = !!(st && st.kite_token_today);
  const ctrl = ctrlOK();
  const stateLine = st ? (pretty(st.state) + (st.strategy ? ' · ' + esc(String(st.strategy)) : '')) : '—';
  const modeWord = cfgPaper() ? 'paper' : 'live';
  const key = [online, kite, ctrl, !!st, reachable, stateLine, modeWord].join('|');

  setSection(el, key, function(){
    if(!st && !reachable){
      return '<div class="lblrow"><span class="lbl">Bot</span></div>' +
             '<div class="offmsg">Waiting for first contact with the server…</div>';
    }
    if(online){
      return '<div class="lblrow"><span class="lbl">Bot</span></div>' +
        '<div class="runrow"><span class="runDot" aria-hidden="true"></span><div>' +
        '<div class="runTitle">Running</div>' +
        '<div class="runSub">' + stateLine + '</div></div></div>';
    }
    if(kite){
      return '<div class="lblrow"><span class="lbl">Morning start</span></div>' +
        '<div class="tokenok">✓&nbsp; Kite token captured for today</div>' +
        '<button type="button" class="btn btn-pos" data-act="arm"' + (ctrl ? '' : ' disabled') + '>Start bot</button>' +
        '<p class="hint">Starts the service in ' + modeWord.toUpperCase() + ' mode. You&rsquo;ll confirm with your PIN.</p>' +
        (ctrl ? '' : '<p class="hint">Controls are unavailable from this host.</p>');
    }
    return '<div class="lblrow"><span class="lbl">Morning start</span></div>' +
      '<div class="step"><div class="snum">1</div><div class="sbody">' +
      '<button type="button" class="btn btn-ghost" data-act="kite">Login with Kite&nbsp;&nbsp;↗</button>' +
      '<p class="hint">Opens Zerodha&rsquo;s official login in a new tab. Once you sign in, you&rsquo;re sent back and the daily token is captured automatically.</p>' +
      '</div></div>' +
      '<div class="step"><div class="snum">2</div><div class="sbody">' +
      '<button type="button" class="btn btn-pos" data-act="arm" disabled>Start bot</button>' +
      '<p class="hint">Unlocks once today&rsquo;s Kite token is in — this page updates by itself.</p>' +
      '</div></div>';
  });
}

function renderMode(online){
  const conf = cfgPaper();
  $('#segPaper').classList.toggle('on', conf === true);
  $('#segLive').classList.toggle('on', conf === false);
  const dis = !ctrlOK();
  $('#segPaper').disabled = dis;
  $('#segLive').disabled = dis;

  const pendEl = $('#modePending');
  const pending = !!(online && st && st.configured_paper != null && st.paper != null && st.configured_paper !== st.paper);
  if(pending){
    const liveNext = st.configured_paper === false;
    pendEl.hidden = false;
    setText(pendEl, (liveNext ? 'Live' : 'Paper') + ' at next start');
    pendEl.classList.toggle('live', liveNext);
  } else {
    pendEl.hidden = true;
  }

  const hint = pending
    ? 'Running session is ' + (st.paper ? 'PAPER' : 'LIVE') + ' — the change takes effect at the next start.'
    : (dis ? 'Mode switching is unavailable from this host.'
           : 'Live places real orders. A change applies at the bot’s next start.');
  setText($('#modeHint'), hint);
}

function posHTML(t){
  const dir = t.type === 'BUY' ? 'CE' : (t.type === 'SELL' ? 'PE' : esc(String(t.type || '—')));
  const bcls = t.type === 'SELL' ? 'pe' : (t.type === 'BUY' ? 'ce' : '');
  let h = '<div class="lblrow"><span class="lbl">Open position</span><span class="bdg ' + bcls + '">' + dir + '</span></div>';
  h += '<div class="posSym num">' + esc(t.symbol || '—') + '</div>';
  h += '<div class="grid2">' +
    cell('Qty', t.qty != null ? inr0.format(t.qty) : '—', 'num') +
    cell('Entry', rupee2(t.entry), 'num') +
    cell('Trail SL', rupee2(t.trail), 'num') +
    cell('Hard SL', rupee2(t.sl), 'num') +
    cell('High-water', rupee2(t.hwm), 'num') +
    cell('Setup score', t.score != null ? fmtN(t.score, 1) : '—', 'num') +
    '</div>';

  /* premium scale: SL → entry → HWM, marker = trailing stop */
  const e = t.entry, s = t.sl, hw = t.hwm, tr = t.trail;
  if(typeof e === 'number' && typeof s === 'number' && typeof hw === 'number'){
    const lo = Math.min(s, e), hi = Math.max(hw, e);
    if(hi > lo){
      const P = v => Math.max(0, Math.min(100, (v - lo) / (hi - lo) * 100));
      const pe = P(e);
      let marks = '';
      let fillW = pe;
      if(typeof tr === 'number'){
        const pt = P(tr); fillW = pt;
        const clab = Math.max(8, Math.min(92, pt));
        marks = '<div class="sc-trail" style="left:' + pt.toFixed(2) + '%"></div>' +
                '<div class="sc-tlab" style="left:' + clab.toFixed(2) + '%">trail ' + rupee2(tr) + '</div>';
      }
      h += '<div class="scale" aria-hidden="true">' +
        '<div class="sc-track"></div>' +
        '<div class="sc-fill" style="width:' + fillW.toFixed(2) + '%"></div>' +
        '<div class="sc-entry" style="left:' + pe.toFixed(2) + '%"></div>' +
        marks + '</div>' +
        '<div class="sc-ends"><span class="lbl">SL ' + rupee2(s) + '</span><span class="lbl">HWM ' + rupee2(hw) + '</span></div>';
    }
  }

  if(t.target_spot != null || t.target_label){
    h += '<div class="target"><div class="lbl">Structure target · index level</div><div class="tval">' +
      '<span class="num">' + (t.target_spot != null ? inr0.format(t.target_spot) : '—') + '</span>' +
      (t.target_label ? '<span class="tlbl">' + esc(t.target_label) + '</span>' : '') +
      '</div></div>';
  }
  return h;
}

function renderPos(){
  const el = $('#posCard');
  const t = (st && st.active_trade) ? st.active_trade : null;
  el.hidden = !t;
  setSection(el, t ? JSON.stringify(t) : '0', t ? function(){ return posHTML(t); } : '');
}

function wallRow(label, arr, cls){
  const chips = (Array.isArray(arr) && arr.length)
    ? arr.slice(0, 6).map(v => '<span class="chip ' + cls + '">' + (typeof v === 'number' ? inr0.format(v) : esc(String(v))) + '</span>').join('')
    : '<span class="nochip">—</span>';
  return '<div class="wallrow"><div class="lbl">' + label + '</div><div class="chips">' + chips + '</div></div>';
}

function renderMarket(){
  const el = $('#marketCard');
  const s = st || {};
  const key = JSON.stringify([s.state, s.strategy, s.regime, s.sentiment, s.day_quality, s.conviction, s.mode,
    s.vix, s.pcr, s.iv_percentile, s.max_pain, s.fii_bias, s.snapshot_age_seconds, s.is_expiry_day,
    s.call_walls, s.put_walls, s.setups_scored, s.setups_passed, s.best_setup_score]);
  setSection(el, key, function(){
    let h = '<div class="lblrow"><span class="lbl">Market</span><span class="mhdr">' +
      (s.is_expiry_day ? '<span class="pill exp">Expiry day</span>' : '') +
      (s.snapshot_age_seconds != null ? '<span class="snap num">snap ' + fmtAge(s.snapshot_age_seconds) + '</span>' : '') +
      '</span></div>';
    h += '<div class="grid2">' +
      cell('State', pretty(s.state)) +
      cell('Strategy', s.strategy ? esc(String(s.strategy)) : '—') +
      cell('Regime', pretty(s.regime)) +
      cell('Sentiment', pretty(s.sentiment)) +
      cell('Day quality', pretty(s.day_quality)) +
      cell('Conviction', pretty(s.conviction)) +
      cell('Risk mode', pretty(s.mode)) +
      cell('FII bias', pretty(s.fii_bias)) +
      cell('VIX', fmtN(s.vix, 2), 'num') +
      cell('PCR', fmtN(s.pcr, 2), 'num') +
      cell('IV percentile', s.iv_percentile != null && isFinite(s.iv_percentile) ? Math.round(s.iv_percentile) + '%' : '—', 'num') +
      cell('Max pain', s.max_pain != null && isFinite(s.max_pain) ? inr0.format(s.max_pain) : '—', 'num') +
      '</div>';
    h += '<div class="walls">' +
      wallRow('Call walls · resistance', s.call_walls, 'res') +
      wallRow('Put walls · support', s.put_walls, 'sup') +
      '</div>';
    h += '<div class="funnel">' +
      '<div class="fcell"><div class="lbl">Scored</div><div class="val num">' + (s.setups_scored != null ? esc(String(s.setups_scored)) : '—') + '</div></div>' +
      '<div class="farr" aria-hidden="true">→</div>' +
      '<div class="fcell"><div class="lbl">Passed</div><div class="val num">' + (s.setups_passed != null ? esc(String(s.setups_passed)) : '—') + '</div></div>' +
      '<div class="farr" aria-hidden="true">→</div>' +
      '<div class="fcell"><div class="lbl">Best score</div><div class="val num">' + (s.best_setup_score != null ? fmtN(s.best_setup_score, 1) : '—') + '</div></div>' +
      '</div>';
    return h;
  });
}

function renderWhy(){
  const el = $('#whyCard');
  const list = (st && Array.isArray(st.why_no_trade))
    ? st.why_no_trade.filter(x => typeof x === 'string' && x.trim()) : [];
  const tradedToday = !!(st && ((st.trades_today_count || 0) > 0 ||
    (Array.isArray(st.completed_trades) && st.completed_trades.length > 0)));
  const show = list.length > 0 && !tradedToday;
  el.hidden = !show;
  if(!show){ el.dataset.k = ''; el.innerHTML = ''; return; }
  setSection(el, JSON.stringify(list), function(){
    let h = '<div class="lblrow"><span class="lbl">Why no trade yet</span></div>';
    for(const line of list){
      const m = line.match(/^\s*verdict\s*:\s*(.*)$/i);
      if(m){
        h += '<div class="verdict"><span class="vtag">Verdict</span><span>' + esc(m[1] || '—') + '</span></div>';
      } else {
        h += '<div class="why-item"><span>' + esc(line) + '</span></div>';
      }
    }
    return h;
  });
}

function tradeHTML(t, i, open){
  const dir = t.type === 'BUY' ? 'CE' : (t.type === 'SELL' ? 'PE' : esc(String(t.type || '—')));
  const bcls = t.type === 'SELL' ? 'pe' : (t.type === 'BUY' ? 'ce' : '');
  const net = (t.net != null && isFinite(t.net)) ? Number(t.net) : null;
  const nc = net == null ? '' : (net > 0 ? 'cpos' : (net < 0 ? 'cneg' : ''));
  return '<div class="trade' + (open ? ' open' : '') + '" data-i="' + i + '" role="button" tabindex="0" aria-expanded="' + open + '">' +
    '<div class="trow"><div class="tleft">' +
    '<div class="tsymrow"><span class="tsym num">' + esc(t.symbol || '—') + '</span><span class="bdg ' + bcls + '">' + dir + '</span></div>' +
    '<div class="tmeta">' + esc(t.strategy || '—') + (t.regime ? ' · ' + esc(t.regime) : '') + '</div>' +
    '</div>' +
    '<div class="tnet num ' + nc + '">' + money(net, 0, true) + '</div>' +
    '<span class="chev" aria-hidden="true">▾</span></div>' +
    '<div class="tx">' +
    '<div class="row"><span>Gross</span><span class="num">' + money(t.gross, 2, true) + '</span></div>' +
    '<div class="row"><span>Costs</span><span class="num">' + (t.costs != null && isFinite(t.costs) ? '−₹' + inr2.format(Math.abs(t.costs)) : '—') + '</span></div>' +
    '<div class="row rownet"><span>Net</span><span class="num ' + nc + '">' + money(net, 2, true) + '</span></div>' +
    '</div></div>';
}

function renderTrades(){
  const list = (st && Array.isArray(st.completed_trades)) ? st.completed_trades : [];
  $('#tradesCard').hidden = !list.length;
  setText($('#tradesCount'), String(list.length));
  const el = $('#tradesList');
  const key = JSON.stringify(list);
  if(el.dataset.k === key) return;
  el.dataset.k = key;
  for(const i of Array.from(expanded)) if(i >= list.length) expanded.delete(i);
  el.innerHTML = list.map((t, i) => tradeHTML(t, i, expanded.has(i))).join('');
}

function renderRisk(){
  const ok = ctrlOK();
  $('#killBtn').disabled = !ok;
  $('#ctrlNote').hidden = ok;
}

/* ================= controls / PIN sheet ================= */
async function control(path, pin, body){
  const h = authHeaders({'X-PIN': pin});
  if(body) h['Content-Type'] = 'application/json';
  const r = await fetch(path, {method:'POST', headers: h, body: body ? JSON.stringify(body) : undefined});
  if(r.status === 401){ closeSheet(); openGate(); throw new Error('Session expired — enter your token.'); }
  let j = null;
  try{ j = await r.json(); }catch(e){}
  if(r.ok && j && j.ok) return j;
  throw new Error((j && (j.error || j.message)) || ('Request failed (HTTP ' + r.status + ')'));
}

function sheetCopy(kind, extra){
  const conf = cfgPaper();
  if(kind === 'arm') return {
    t: 'Start bot',
    b: 'Boots the trading service in ' + (conf ? 'PAPER' : 'LIVE') + ' mode' +
       (conf ? ' — simulated fills, no real orders.' : ' — it will place real orders with real money.') +
       ' Enter your PIN to confirm.',
    btn: 'Start bot', cls: 'btn-pos'
  };
  if(kind === 'kill') return {
    t: 'Kill switch',
    b: 'Flattens any open position at market and stops the bot for the day. This is immediate and cannot be taken back.',
    btn: 'Flatten & stop', cls: 'btn-negsolid'
  };
  if(kind === 'clear') return {
    t: 'Clear pending kill',
    b: 'Removes the kill flag so the bot is allowed to run again.',
    btn: 'Clear kill', cls: 'btn-ink'
  };
  if(extra && extra.paper) return {
    t: 'Switch to Paper',
    b: 'Paper mode simulates fills — no real orders are sent. The change takes effect the next time the bot starts; a running session is unaffected.',
    btn: 'Switch to Paper', cls: 'btn-ink'
  };
  return {
    t: 'Switch to Live',
    b: 'Live mode places real orders with real money at your broker. The change takes effect at the bot’s NEXT start — the currently running session keeps its mode.',
    btn: 'Switch to Live', cls: 'btn-negsolid'
  };
}

function openSheet(kind, extra){
  sheetCtx = {kind, extra: extra || {}};
  const c = sheetCopy(kind, extra);
  setText($('#sheetTitle'), c.t);
  setText($('#sheetBody'), c.b);
  const go = $('#sheetGo');
  go.className = 'btn ' + c.cls;
  go.textContent = c.btn;
  go.disabled = false;
  $('#sheetErr').textContent = '';
  $('#sheetPin').value = '';
  $('#sheet').classList.add('open');
  $('#sheet').setAttribute('aria-hidden', 'false');
  setTimeout(() => { try{ $('#sheetPin').focus(); }catch(e){} }, 60);
}
function closeSheet(){
  if(sheetBusy) return;
  sheetCtx = null;
  $('#sheet').classList.remove('open');
  $('#sheet').setAttribute('aria-hidden', 'true');
}
function sheetErr(msg){ $('#sheetErr').textContent = msg || ''; }

async function confirmSheet(){
  if(!sheetCtx || sheetBusy) return;
  const pin = $('#sheetPin').value.trim();
  if(!pin){ sheetErr('Enter your PIN.'); return; }
  const go = $('#sheetGo');
  const prev = go.textContent;
  sheetBusy = true; go.disabled = true; go.textContent = 'Working…'; sheetErr('');
  try{
    const k = sheetCtx.kind;
    let j;
    if(k === 'arm')        j = await control('/api/control/arm', pin);
    else if(k === 'kill')  j = await control('/api/control/kill', pin);
    else if(k === 'clear') j = await control('/api/control/clear', pin);
    else                   j = await control('/api/control/mode', pin, {paper: !!sheetCtx.extra.paper});
    sheetBusy = false;
    closeSheet();
    toast(j.message || 'Done.', 'ok');
    setTimeout(poll, 400);
  }catch(e){
    sheetErr(e && e.message ? e.message : 'Failed.');
  }finally{
    sheetBusy = false;
    go.disabled = false;
    go.textContent = prev;
  }
}

function trySwitchMode(toPaper){
  if(cfgPaper() === toPaper) return;
  if(!ctrlOK()){ toast('Controls are unavailable from this host.', 'err'); return; }
  openSheet('mode', {paper: toPaper});
}

/* ================= toasts ================= */
function toast(msg, kind){
  const c = $('#toasts');
  const d = document.createElement('div');
  d.className = 'toast t-' + (kind || 'info');
  const dot = document.createElement('span'); dot.className = 'tdot';
  const tx = document.createElement('span'); tx.textContent = msg;
  d.appendChild(dot); d.appendChild(tx);
  c.appendChild(d);
  while(c.children.length > 3) c.firstChild.remove();
  const rm = () => d.remove();
  d.addEventListener('click', rm);
  setTimeout(rm, 4200);
}

/* ================= gate ================= */
function openGate(){
  gated = true;
  $('#gateErr').hidden = !gateAttempted;
  $('#gate').classList.add('open');
  setTimeout(() => { try{ $('#gateInput').focus(); }catch(e){} }, 60);
}
function saveGate(){
  const v = $('#gateInput').value.trim();
  if(!v) return;
  token = v;
  localStorage.setItem('zt', v);
  gateAttempted = true;
  gated = false;
  $('#gate').classList.remove('open');
  $('#gateInput').value = '';
  poll();
}

/* ================= logs ================= */
function toggleLogs(){
  logsOpen = !logsOpen;
  $('#logsBody').hidden = !logsOpen;
  $('#logsBtn').setAttribute('aria-expanded', String(logsOpen));
  $('#logsChev').classList.toggle('up', logsOpen);
  if(logsOpen){
    $('#logsPre').textContent = 'Loading…';
    loadLogs(true);
    logsTimer = setInterval(() => { if(!document.hidden) loadLogs(false); }, 8000);
  } else if(logsTimer){
    clearInterval(logsTimer);
    logsTimer = null;
  }
}
async function loadLogs(first){
  try{
    const r = await fetch('/api/logs?lines=250', {headers: authHeaders(), cache:'no-store'});
    if(r.status === 401){ openGate(); return; }
    const t = await r.text();
    const box = $('#logsBox');
    const pinned = first || (box.scrollTop + box.clientHeight >= box.scrollHeight - 48);
    $('#logsPre').textContent = t && t.trim() ? t : '(no log output)';
    if(pinned) box.scrollTop = box.scrollHeight;
  }catch(e){
    if(first) $('#logsPre').textContent = 'Could not load logs — will retry.';
  }
}

/* ================= events ================= */
document.addEventListener('click', e => {
  const b = e.target.closest('[data-act]');
  if(!b || b.disabled) return;
  const a = b.dataset.act;
  if(a === 'kite'){
    window.open('/kite/login', '_blank', 'noopener');
    toast('Finish the Zerodha login in the new tab — this page updates automatically.', 'info');
  }
  else if(a === 'arm')  openSheet('arm');
  else if(a === 'kill') openSheet('kill');
  else if(a === 'clear') openSheet('clear');
  else if(a === 'mode-paper') trySwitchMode(true);
  else if(a === 'mode-live')  trySwitchMode(false);
  else if(a === 'logs') toggleLogs();
});

$('#tradesList').addEventListener('click', e => {
  const row = e.target.closest('.trade');
  if(row) toggleTrade(row);
});
$('#tradesList').addEventListener('keydown', e => {
  if(e.key !== 'Enter' && e.key !== ' ') return;
  const row = e.target.closest('.trade');
  if(row){ e.preventDefault(); toggleTrade(row); }
});
function toggleTrade(row){
  const i = Number(row.dataset.i);
  const open = !row.classList.contains('open');
  row.classList.toggle('open', open);
  row.setAttribute('aria-expanded', String(open));
  if(open) expanded.add(i); else expanded.delete(i);
}

$('#sheetBack').addEventListener('click', closeSheet);
$('#sheetCancel').addEventListener('click', closeSheet);
$('#sheetGo').addEventListener('click', confirmSheet);
$('#sheetPin').addEventListener('keydown', e => { if(e.key === 'Enter'){ e.preventDefault(); confirmSheet(); } });
document.addEventListener('keydown', e => {
  if(e.key === 'Escape' && $('#sheet').classList.contains('open')) closeSheet();
});
$('#gateBtn').addEventListener('click', saveGate);
$('#gateInput').addEventListener('keydown', e => { if(e.key === 'Enter'){ e.preventDefault(); saveGate(); } });

document.addEventListener('visibilitychange', () => {
  if(!document.hidden){
    poll();
    if(logsOpen) loadLogs(false);
  }
});

/* freshness ticker */
setInterval(() => {
  const el = $('#fresh');
  if(!lastOkAt){ el.textContent = reachable ? '' : 'no data'; return; }
  const n = Math.max(0, Math.round((Date.now() - lastOkAt) / 1000));
  el.textContent = 'updated ' + (n <= 1 ? 'now' : (n < 60 ? n + 's ago' : Math.floor(n / 60) + 'm ago'));
}, 1000);

/* ================= boot ================= */
window.addEventListener('load', () => {
  if('serviceWorker' in navigator){
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  }
});
poll();
setInterval(poll, 4000);
</script>
</body>
</html>
"""
