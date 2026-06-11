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


# PIN with lockout: 5 wrong attempts → 15-minute freeze.
_pin_fails: list[float] = []


def _pin_ok(pin: str | None) -> tuple[bool, str]:
    if not _PIN:
        return False, "controls disabled — set dashboard.control_pin in config.yaml"
    now = time.time()
    recent = [t for t in _pin_fails if now - t < 900]
    _pin_fails[:] = recent
    if len(recent) >= 5:
        return False, "locked: too many wrong PINs — try again in 15 minutes"
    if pin and _ct_eq(str(pin), _PIN):
        _pin_fails.clear()
        return True, ""
    _pin_fails.append(now)
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
    return status


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


@app.get("/api/logs", response_class=PlainTextResponse)
def api_logs(request: Request, token: str | None = Query(default=None),
             lines: int = Query(default=200, ge=10, le=2000)):
    if not _authorized(request, token):
        return PlainTextResponse("unauthorized", status_code=401)
    return _tail_log(lines)


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
        "name": "zAck",
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
self.addEventListener('activate',e=>e.waitUntil(clients.claim()));
self.addEventListener('fetch',e=>{
  const u=new URL(e.request.url);
  if(e.request.method!=='GET'||u.pathname.startsWith('/api/')||u.pathname.startsWith('/kite/'))return;
  e.respondWith(caches.open('zack-v1').then(async c=>{
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
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" media="(prefers-color-scheme: light)" content="#f2f2f7">
<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#000000">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="apple-touch-icon" href="/icon-180.png">
<title>zAck</title>
<style>
:root{
  --bg:#f2f2f7; --card:#ffffff; --fg:#000; --sec:#6e6e73; --ter:#aeaeb2;
  --sep:rgba(60,60,67,.10); --green:#34c759; --red:#ff3b30; --blue:#007aff;
  --amber:#ff9500; --fill:rgba(120,120,128,.12);
}
@media (prefers-color-scheme: dark){:root{
  --bg:#000; --card:#1c1c1e; --fg:#fff; --sec:#98989f; --ter:#636366;
  --sep:rgba(84,84,88,.36); --green:#30d158; --red:#ff453a; --blue:#0a84ff;
  --amber:#ff9f0a; --fill:rgba(120,120,128,.18);
}}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,Inter,sans-serif;
  padding-bottom:calc(28px + env(safe-area-inset-bottom))}
.hdr{position:sticky;top:0;z-index:10;display:flex;align-items:center;gap:10px;
  padding:calc(10px + env(safe-area-inset-top)) 20px 10px;
  background:color-mix(in srgb,var(--bg) 72%,transparent);
  -webkit-backdrop-filter:saturate(180%) blur(20px);backdrop-filter:saturate(180%) blur(20px)}
.hdr h1{font-size:28px;font-weight:700;letter-spacing:-.02em;margin:0}
.spacer{flex:1}
.pill{font-size:12px;font-weight:600;padding:4px 10px;border-radius:999px;
  background:var(--fill);color:var(--sec);letter-spacing:.02em}
.pill.live{color:#fff;background:var(--green)} .pill.paper{color:#fff;background:var(--amber)}
.pill.off{color:#fff;background:var(--ter)} .pill.killed{color:#fff;background:var(--red)}
.wrap{max-width:560px;margin:0 auto;padding:4px 16px;display:flex;flex-direction:column;gap:12px}
.card{background:var(--card);border-radius:16px;padding:16px}
.lbl{font-size:12px;font-weight:600;color:var(--sec);text-transform:uppercase;letter-spacing:.05em}
.hero{font-size:44px;font-weight:700;letter-spacing:-.03em;margin:4px 0 2px;
  font-variant-numeric:tabular-nums}
.sub{font-size:14px;color:var(--sec)}
.pos{color:var(--green)}.neg{color:var(--red)}.neu{color:var(--fg)}
.row{display:flex;justify-content:space-between;align-items:center;padding:10px 0;
  border-bottom:.5px solid var(--sep);font-size:15px}
.row:last-child{border-bottom:none;padding-bottom:0}
.row .k{color:var(--sec)} .row .v{font-weight:500;font-variant-numeric:tabular-nums;text-align:right}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip{font-size:12px;font-weight:500;padding:3px 9px;border-radius:8px;background:var(--fill);color:var(--sec)}
.chip.r{color:var(--red)} .chip.g{color:var(--green)}
.sym{font-size:20px;font-weight:600;letter-spacing:-.01em}
.tgrid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:12px}
.tcell .lbl{font-size:11px}.tcell .tv{font-size:17px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
.trade{display:flex;align-items:center;gap:10px;padding:11px 0;border-bottom:.5px solid var(--sep)}
.trade:last-child{border-bottom:none;padding-bottom:2px}
.trade .ic{width:32px;height:32px;border-radius:9px;display:flex;align-items:center;justify-content:center;
  font-size:12px;font-weight:700;color:#fff;flex-shrink:0}
.trade .m{flex:1;min-width:0}.trade .m .s1{font-size:15px;font-weight:500}
.trade .m .s2{font-size:12px;color:var(--sec);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.trade .pl{font-size:15px;font-weight:600;font-variant-numeric:tabular-nums}
ul.why{margin:8px 0 0;padding-left:18px}
ul.why li{font-size:14px;color:var(--sec);margin:5px 0;line-height:1.45}
.verdict{margin-top:10px;font-size:14px;color:var(--amber);line-height:1.45}
.btnrow{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px}
.btn{appearance:none;border:0;border-radius:13px;padding:14px;font-size:16px;font-weight:600;
  font-family:inherit;color:#fff;cursor:pointer;transition:opacity .15s}
.btn:active{opacity:.6}
.btn.blue{background:var(--blue)}.btn.red{background:var(--red)}
.btn.gray{background:var(--fill);color:var(--fg)}
.btn[disabled]{opacity:.4;pointer-events:none}
details{border-radius:16px;background:var(--card);padding:14px 16px}
summary{font-size:15px;font-weight:600;cursor:pointer;list-style:none;display:flex;align-items:center}
summary::after{content:"›";margin-left:auto;color:var(--ter);font-size:20px;transform:rotate(90deg)}
details[open] summary::after{transform:rotate(-90deg)}
pre{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:11px;line-height:1.5;
  background:var(--bg);border-radius:10px;padding:10px;overflow:auto;max-height:46vh;
  white-space:pre-wrap;word-break:break-word;margin:12px 0 0}
.sheetbg{position:fixed;inset:0;background:rgba(0,0,0,.45);display:none;z-index:50;
  align-items:flex-end;justify-content:center}
.sheetbg.show{display:flex}
.sheet{background:var(--card);border-radius:20px 20px 0 0;width:100%;max-width:560px;
  padding:22px 20px calc(24px + env(safe-area-inset-bottom))}
.sheet h3{margin:0 0 4px;font-size:19px;font-weight:600}
.sheet p{margin:0 0 14px;font-size:14px;color:var(--sec);line-height:1.45}
.pinbox{width:100%;font-size:24px;letter-spacing:10px;text-align:center;padding:12px;
  border-radius:12px;border:1px solid var(--sep);background:var(--bg);color:var(--fg);
  font-family:inherit;font-variant-numeric:tabular-nums;outline:none;margin-bottom:12px}
.gate{position:fixed;inset:0;background:var(--bg);z-index:100;display:none;
  align-items:center;justify-content:center;padding:24px}
.gate.show{display:flex}
.gatecard{max-width:380px;width:100%;text-align:center}
.gatecard .logo{width:72px;height:72px;border-radius:18px;margin:0 auto 18px;display:block}
.gatecard h2{font-size:24px;font-weight:700;margin:0 0 6px}
.gatecard p{font-size:14px;color:var(--sec);margin:0 0 18px;line-height:1.5}
.gatecard input{width:100%;font-size:16px;padding:13px 14px;border-radius:12px;
  border:1px solid var(--sep);background:var(--card);color:var(--fg);font-family:inherit;
  outline:none;margin-bottom:12px;text-align:center}
.toast{position:fixed;left:50%;transform:translateX(-50%);bottom:calc(30px + env(safe-area-inset-bottom));
  background:var(--fg);color:var(--bg);font-size:14px;font-weight:500;padding:11px 18px;
  border-radius:999px;opacity:0;transition:opacity .25s;pointer-events:none;z-index:200;max-width:86vw;text-align:center}
.toast.show{opacity:.95}
</style>
</head>
<body>

<header class="hdr">
  <h1>zAck</h1>
  <span class="spacer"></span>
  <span id="killPill" class="pill killed" style="display:none">KILL PENDING</span>
  <span id="modePill" class="pill off">—</span>
</header>

<main class="wrap">

  <section class="card">
    <div class="lbl">Today</div>
    <div id="pnl" class="hero neu">—</div>
    <div class="sub" id="pnlSub">connecting…</div>
    <div class="tgrid">
      <div class="tcell"><div class="lbl">Wins</div><div class="tv" id="wins">—</div></div>
      <div class="tcell"><div class="lbl">Losses</div><div class="tv" id="losses">—</div></div>
      <div class="tcell"><div class="lbl">Trades</div><div class="tv" id="trades">—</div></div>
    </div>
  </section>

  <section class="card" id="activeCard" style="display:none">
    <div class="lbl">Open position</div>
    <div style="display:flex;align-items:center;gap:8px;margin-top:6px">
      <span class="sym" id="atSym">—</span>
      <span class="pill" id="atType"></span>
    </div>
    <div class="tgrid">
      <div class="tcell"><div class="lbl">Entry</div><div class="tv" id="atEntry">—</div></div>
      <div class="tcell"><div class="lbl">Trail SL</div><div class="tv" id="atTrail">—</div></div>
      <div class="tcell"><div class="lbl">Qty</div><div class="tv" id="atQty">—</div></div>
      <div class="tcell"><div class="lbl">Hard SL</div><div class="tv" id="atSL">—</div></div>
      <div class="tcell"><div class="lbl">Target</div><div class="tv" id="atTgt">—</div></div>
      <div class="tcell"><div class="lbl">Score</div><div class="tv" id="atScore">—</div></div>
    </div>
  </section>

  <section class="card">
    <div class="lbl">Market</div>
    <div class="row"><span class="k">State</span><span class="v" id="state">—</span></div>
    <div class="row"><span class="k">Strategy</span><span class="v" id="strategy">—</span></div>
    <div class="row"><span class="k">Regime</span><span class="v" id="regime">—</span></div>
    <div class="row"><span class="k">Sentiment</span><span class="v" id="sentiment">—</span></div>
    <div class="row"><span class="k">Day quality</span><span class="v" id="dayq">—</span></div>
    <div class="row"><span class="k">VIX · PCR</span><span class="v" id="vixpcr">—</span></div>
    <div class="row"><span class="k">IV %ile · Max pain</span><span class="v" id="ivmp">—</span></div>
    <div class="row"><span class="k">FII bias</span><span class="v" id="fii">—</span></div>
    <div class="row"><span class="k">Balance</span><span class="v" id="bal">—</span></div>
    <div class="chips" id="walls"></div>
  </section>

  <section class="card" id="whyCard" style="display:none">
    <div class="lbl">Why no trade yet</div>
    <ul class="why" id="whyList"></ul>
    <div class="verdict" id="verdict"></div>
  </section>

  <section class="card" id="tradesCard" style="display:none">
    <div class="lbl" style="margin-bottom:4px">Today's trades</div>
    <div id="tradesList"></div>
  </section>

  <section class="card">
    <div class="lbl" style="margin-bottom:10px">Operations</div>
    <div class="btnrow">
      <button class="btn blue" id="armBtn" onclick="askPin('arm')">Arm bot</button>
      <button class="btn red" onclick="askPin('kill')">Kill switch</button>
    </div>
    <div class="btnrow" style="margin-top:10px">
      <button class="btn gray" onclick="window.open(api('/kite/login'),'_blank')">Kite login</button>
      <button class="btn gray" id="clearBtn" style="display:none" onclick="askPin('clear')">Clear kill</button>
    </div>
    <div class="sub" id="opsSub" style="margin-top:10px"></div>
  </section>

  <details>
    <summary>Logs</summary>
    <pre id="logs">…</pre>
  </details>

</main>

<div class="sheetbg" id="sheetBg" onclick="if(event.target===this)hideSheet()">
  <div class="sheet">
    <h3 id="sheetTitle">Enter PIN</h3>
    <p id="sheetSub">This action talks to the live bot.</p>
    <input class="pinbox" id="pin" inputmode="numeric" autocomplete="one-time-code"
           maxlength="8" placeholder="••••">
    <div class="btnrow">
      <button class="btn gray" onclick="hideSheet()">Cancel</button>
      <button class="btn blue" id="sheetGo" onclick="doAction()">Confirm</button>
    </div>
  </div>
</div>

<div class="gate" id="gate">
  <div class="gatecard">
    <img class="logo" src="/icon-180.png" alt="">
    <h2>zAck</h2>
    <p>Enter the dashboard token to connect. It's stored only on this device.</p>
    <input id="tokIn" type="password" placeholder="Dashboard token"
           onkeydown="if(event.key==='Enter')saveTok()">
    <button class="btn blue" style="width:100%" onclick="saveTok()">Connect</button>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const qs=new URLSearchParams(location.search);
if(qs.get('token')){localStorage.setItem('zt',qs.get('token'));history.replaceState(null,'','/');}
let TOK=localStorage.getItem('zt')||'';
const api=p=>p;
const H=()=>TOK?{'X-Auth':TOK}:{};
const inr=n=>n==null?'—':'₹'+Number(n).toLocaleString('en-IN',{maximumFractionDigits:0});
const inr2=n=>n==null?'—':'₹'+Number(n).toLocaleString('en-IN',{maximumFractionDigits:2});
const sgn=n=>(n>0?'+':'')+inr(n);
const cls=n=>n>0?'pos':(n<0?'neg':'neu');
const set=(id,v)=>{document.getElementById(id).textContent=(v==null||v==='')?'—':v;};
let toastT;function toast(m){const t=document.getElementById('toast');t.textContent=m;
  t.classList.add('show');clearTimeout(toastT);toastT=setTimeout(()=>t.classList.remove('show'),2600);}
function gate(on){document.getElementById('gate').classList.toggle('show',on);}
function saveTok(){TOK=document.getElementById('tokIn').value.trim();
  localStorage.setItem('zt',TOK);gate(false);tick();loadLogs();}

async function tick(){
  let s;
  try{
    const r=await fetch('/api/status',{headers:H(),cache:'no-store'});
    if(r.status===401){gate(true);return;}
    s=await r.json();
  }catch(e){
    document.getElementById('modePill').className='pill off';
    set('pnlSub','cannot reach server');return;
  }
  const pill=document.getElementById('modePill');
  if(!s.online){pill.className='pill off';pill.textContent='OFFLINE';}
  else if(s.paper){pill.className='pill paper';pill.textContent='PAPER';}
  else{pill.className='pill live';pill.textContent='LIVE';}
  document.getElementById('killPill').style.display=s.kill_pending?'':'none';
  document.getElementById('clearBtn').style.display=s.kill_pending?'':'none';

  const pnl=s.realized_pnl_today??s.daily_pnl_persisted??0;
  const pe=document.getElementById('pnl');pe.textContent=sgn(pnl);pe.className='hero '+cls(pnl);
  set('pnlSub', s.online?('live · updated '+(s.snapshot_age_seconds??'?')+'s ago')
                :'bot offline — last saved data');
  set('wins',s.wins??0);set('losses',s.losses??0);
  set('trades',(s.trades_today_count??0)+'/'+(s.max_trades??'—'));

  set('state',s.state);set('strategy',(s.manual_mode?'[M] ':'')+(s.strategy||'—'));
  set('regime',s.regime);set('sentiment',s.sentiment);
  set('dayq',s.day_quality);
  set('vixpcr',(s.vix?Number(s.vix).toFixed(1):'—')+' · '+(s.pcr!=null?Number(s.pcr).toFixed(2):'—'));
  set('ivmp',(s.iv_percentile!=null?Math.round(s.iv_percentile)+'%':'—')+' · '+
             (s.max_pain!=null?Number(s.max_pain).toLocaleString('en-IN'):'—'));
  set('fii',s.fii_bias);
  set('bal',inr(s.balance)+(s.balance_label?(' · '+s.balance_label):''));
  const w=document.getElementById('walls');w.innerHTML='';
  (s.call_walls||[]).forEach(x=>{const c=document.createElement('span');
    c.className='chip r';c.textContent='CE '+Number(x).toLocaleString('en-IN');w.appendChild(c);});
  (s.put_walls||[]).forEach(x=>{const c=document.createElement('span');
    c.className='chip g';c.textContent='PE '+Number(x).toLocaleString('en-IN');w.appendChild(c);});

  const a=s.active_trade,ac=document.getElementById('activeCard');
  if(a&&a.symbol){ac.style.display='';
    set('atSym',a.symbol);
    const tp=document.getElementById('atType');tp.textContent=a.type||'';tp.className='pill';
    set('atEntry',inr2(a.entry));set('atTrail',inr2(a.trail));set('atQty',a.qty);
    set('atSL',inr2(a.sl));
    set('atTgt',a.target_spot?(Number(a.target_spot).toLocaleString('en-IN')+
        (a.target_label?(' '+a.target_label):'')):'—');
    set('atScore',a.score!=null?a.score:'—');
  } else ac.style.display='none';

  const why=s.why_no_trade||[],wc=document.getElementById('whyCard');
  if(why.length&&!(s.trades_today_count>0)){wc.style.display='';
    const ul=document.getElementById('whyList');ul.innerHTML='';let v='';
    why.forEach(l=>{if(String(l).startsWith('Verdict:'))v=l;
      else{const li=document.createElement('li');li.textContent=l;ul.appendChild(li);}});
    document.getElementById('verdict').textContent=v;
  } else wc.style.display='none';

  const t=s.completed_trades||[],tc=document.getElementById('tradesCard');
  if(t.length){tc.style.display='';const L=document.getElementById('tradesList');L.innerHTML='';
    t.forEach(x=>{const d=document.createElement('div');d.className='trade';
      const up=(x.net??0)>=0;
      d.innerHTML='<div class="ic" style="background:'+(up?'var(--green)':'var(--red)')+'">'+
        (x.type==='SELL'?'PE':'CE')+'</div>'+
        '<div class="m"><div class="s1"></div><div class="s2"></div></div>'+
        '<div class="pl '+cls(x.net)+'"></div>';
      d.querySelector('.s1').textContent=x.symbol||'';
      d.querySelector('.s2').textContent=(x.strategy||'')+(x.regime?(' · '+x.regime):'');
      d.querySelector('.pl').textContent=sgn(x.net);
      L.appendChild(d);});
  } else tc.style.display='none';

  set('opsSub',(s.kite_token_today?'Kite token: captured today ✓':'Kite token: not captured today')+
      (s.controls_available?'':' · controls disabled (no PIN set)'));
  document.getElementById('armBtn').disabled=!s.controls_available;
}

async function loadLogs(){
  try{const r=await fetch('/api/logs?lines=250',{headers:H(),cache:'no-store'});
    if(r.status===401)return;
    const t=await r.text();const el=document.getElementById('logs');
    el.textContent=t;el.scrollTop=el.scrollHeight;
  }catch(e){}
}

let pendingAction=null;
const ACT={arm:{t:'Arm the bot',s:'Starts the bot service on the server.',btn:'Arm'},
  kill:{t:'Kill switch',s:'Flattens any open position at market and stops the bot. This trades real money if live.',btn:'Kill'},
  clear:{t:'Clear kill switch',s:'Allows the bot to run again.',btn:'Clear'}};
function askPin(a){pendingAction=a;const m=ACT[a];
  document.getElementById('sheetTitle').textContent=m.t;
  document.getElementById('sheetSub').textContent=m.s;
  document.getElementById('sheetGo').textContent=m.btn;
  document.getElementById('sheetGo').className='btn '+(a==='kill'?'red':'blue');
  document.getElementById('pin').value='';
  document.getElementById('sheetBg').classList.add('show');
  setTimeout(()=>document.getElementById('pin').focus(),60);}
function hideSheet(){document.getElementById('sheetBg').classList.remove('show');}
async function doAction(){
  const pin=document.getElementById('pin').value.trim();
  if(!pin){toast('Enter the PIN');return;}
  const ep={arm:'/api/control/arm',kill:'/api/control/kill',clear:'/api/control/clear'}[pendingAction];
  try{
    const r=await fetch(ep,{method:'POST',headers:{...H(),'X-PIN':pin}});
    const j=await r.json();
    toast(j.ok?(j.message||'done'):(j.error||'failed'));
    if(j.ok){hideSheet();tick();}
  }catch(e){toast('request failed');}
}

if('serviceWorker' in navigator){navigator.serviceWorker.register('/sw.js').catch(()=>{});}
tick();loadLogs();
setInterval(tick,4000);
setInterval(()=>{const d=document.querySelector('details');if(d&&d.open)loadLogs();},8000);
</script>
</body>
</html>"""
