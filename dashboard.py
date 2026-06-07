"""
zAck Bot — read-only monitoring dashboard (FastAPI + embedded PWA).

Serves a mobile-friendly dashboard that shows live bot status, today's P&L,
win/loss, the active trade, completed trades, the "why no trade" rationale, and
a live log tail — all by reading the trading process's existing state files
(state/bot_status.json, state/daily_pnl.json, state/trade_ledger_<date>.json)
and output/bot.log. It does NOT control the bot and never touches secrets, so
it's safe to run alongside the trading process.

Run (local test):
    uvicorn dashboard:app --host 127.0.0.1 --port 8000

Production (behind WireGuard — bind to the VPN interface only):
    DASHBOARD_TOKEN=<long-random> uvicorn dashboard:app --host 10.10.0.1 --port 8000

Then open http://<host>:8000/  (append ?token=... if DASHBOARD_TOKEN is set).
On Android Chrome: ⋮ → "Add to Home screen" to install it like an app.

Security:
  • Bind to the WireGuard interface (or 127.0.0.1), never 0.0.0.0 on a public box.
  • Optional bearer token: set env DASHBOARD_TOKEN to require ?token= / Authorization.
  • Read-only: no start/stop/secret access here.
"""
from __future__ import annotations

import datetime
import os

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from infra import state_path, read_json

app = FastAPI(title="zAck Bot Dashboard", docs_url=None, redoc_url=None)

_TOKEN = os.environ.get("DASHBOARD_TOKEN", "").strip()
LOG_FILE = os.environ.get("BOT_LOG_FILE", "output/bot.log")


# --------------------------------------------------------------------------- #
# Auth (optional, token-based)
# --------------------------------------------------------------------------- #
def _authorized(request: Request, token: str | None) -> bool:
    if not _TOKEN:
        return True  # no token configured → open (use only on localhost/VPN)
    if token and token == _TOKEN:
        return True
    auth = request.headers.get("authorization", "")
    return auth == f"Bearer {_TOKEN}"


def _deny() -> JSONResponse:
    return JSONResponse({"error": "unauthorized"}, status_code=401)


# --------------------------------------------------------------------------- #
# Data assembly from the bot's state files
# --------------------------------------------------------------------------- #
def _today_str() -> str:
    return datetime.date.today().isoformat()


def _build_status() -> dict:
    status = read_json(state_path("bot_status.json"), default=None) or {}
    daily = read_json(state_path("daily_pnl.json"), default={}) or {}
    weekly = read_json(state_path("weekly_pnl.json"), default={}) or {}
    ledger = read_json(state_path(f"trade_ledger_{_today_str()}.json"), default={}) or {}

    # Freshness: how long since the bot last wrote a snapshot.
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

    # Prefer live snapshot; fall back to persisted ledger if the bot is offline.
    if not status.get("completed_trades") and ledger.get("completed_trades"):
        status["completed_trades"] = ledger["completed_trades"]
        status.setdefault("wins", ledger.get("wins", 0))
        status.setdefault("losses", ledger.get("losses", 0))
        status.setdefault("trades_today_count", ledger.get("trades_today_count", 0))

    status["daily_pnl_persisted"] = daily.get(_today_str())
    status["weekly_pnl_persisted"] = next(iter(weekly.values()), None) if weekly else None
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


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


@app.get("/manifest.webmanifest")
def manifest():
    return JSONResponse({
        "name": "zAck Bot Dashboard",
        "short_name": "zAck Bot",
        "start_url": ".",
        "display": "standalone",
        "background_color": "#0b1220",
        "theme_color": "#0b1220",
        "icons": [],
    })


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_PAGE)


# --------------------------------------------------------------------------- #
# Embedded single-page PWA (vanilla JS, no build step, mobile-first)
# --------------------------------------------------------------------------- #
_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b1220">
<link rel="manifest" href="manifest.webmanifest">
<title>zAck Bot</title>
<style>
  :root{--bg:#0b1220;--card:#131c2e;--mut:#8aa0c0;--fg:#e8eefc;--green:#27c08a;--red:#ff5d6c;--amber:#ffb454;--line:#22304a;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:env(safe-area-inset-top) 12px 24px}
  h1{font-size:18px;margin:14px 4px 4px;display:flex;align-items:center;gap:8px}
  .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
  .sub{color:var(--mut);font-size:12px;margin:0 4px 12px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 14px}
  .card.full{grid-column:1/-1}
  .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
  .v{font-size:20px;font-weight:700;margin-top:2px}
  .v.sm{font-size:15px;font-weight:600}
  .pos{color:var(--green)} .neg{color:var(--red)} .neu{color:var(--amber)}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th,td{text-align:left;padding:6px 4px;border-bottom:1px solid var(--line)}
  th{color:var(--mut);font-weight:600}
  .why li{margin:4px 0;font-size:13px;color:var(--fg)}
  .verdict{margin-top:8px;font-size:13px;color:var(--amber)}
  pre{background:#0a1322;border:1px solid var(--line);border-radius:10px;padding:10px;font-size:11px;line-height:1.45;overflow:auto;max-height:46vh;white-space:pre-wrap;word-break:break-word}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;border:1px solid var(--line);color:var(--mut)}
  .row{display:flex;justify-content:space-between;align-items:center;gap:8px}
  a.refresh{color:var(--mut);font-size:12px;text-decoration:none}
</style>
</head>
<body>
  <h1><span id="dot" class="dot" style="background:#555"></span> zAck Bot
      <span id="paper" class="pill" style="margin-left:auto"></span></h1>
  <p class="sub" id="sub">connecting…</p>

  <div class="grid">
    <div class="card"><div class="k">State</div><div class="v sm" id="state">—</div>
        <div class="k" style="margin-top:8px">Strategy</div><div class="v sm" id="strategy">—</div></div>
    <div class="card"><div class="k">Net P&amp;L today</div><div class="v" id="pnl">—</div>
        <div class="k" style="margin-top:8px">W / L · Trades</div><div class="v sm" id="wl">—</div></div>

    <div class="card"><div class="k">Sentiment / Bias</div><div class="v sm" id="sentiment">—</div>
        <div class="k" style="margin-top:8px">Regime · Day quality</div><div class="v sm" id="dayq">—</div></div>
    <div class="card"><div class="k">Balance</div><div class="v sm" id="bal">—</div>
        <div class="k" style="margin-top:8px">Mode · VIX/IV</div><div class="v sm" id="mode">—</div></div>

    <div class="card full">
      <div class="k">Market flow (institutional footprint)</div>
      <table><tr><th>VIX</th><th>IV %ile</th><th>PCR</th><th>Max-pain</th><th>FII</th></tr>
        <tr><td id="mfVix">—</td><td id="mfIvp">—</td><td id="mfPcr">—</td><td id="mfMp">—</td><td id="mfFii">—</td></tr></table>
      <div class="k" style="margin-top:6px">Call walls (resistance) · Put walls (support)</div>
      <div class="v sm" id="mfWalls">—</div>
    </div>

    <div class="card full" id="activeCard" style="display:none">
      <div class="row"><div class="k">Open position</div><span class="pill" id="atType"></span></div>
      <div class="v sm" id="atSym">—</div>
      <table><tr><th>Entry</th><th>Trail</th><th>Hard SL</th><th>Qty</th></tr>
        <tr><td id="atEntry">—</td><td id="atTrail">—</td><td id="atSL">—</td><td id="atQty">—</td></tr></table>
    </div>

    <div class="card full" id="whyCard" style="display:none">
      <div class="k">Why no trade yet</div>
      <ul class="why" id="whyList"></ul>
      <div class="verdict" id="verdict"></div>
    </div>

    <div class="card full" id="tradesCard" style="display:none">
      <div class="k">Today's trades</div>
      <table id="tradesTbl"><thead><tr><th>#</th><th>Dir</th><th>Symbol</th><th>Net</th><th>Strategy</th></tr></thead>
        <tbody id="tradesBody"></tbody></table>
    </div>

    <div class="card full">
      <div class="row"><div class="k">Live logs</div><a class="refresh" href="#" onclick="loadLogs();return false">refresh</a></div>
      <pre id="logs">loading…</pre>
    </div>
  </div>

<script>
const qs=new URLSearchParams(location.search); const TOKEN=qs.get('token')||'';
const tp=p=>TOKEN?(p+(p.includes('?')?'&':'?')+'token='+encodeURIComponent(TOKEN)):p;
const f=n=>n==null?'—':'₹'+Number(n).toLocaleString('en-IN',{maximumFractionDigits:2});
const cls=n=>n>0?'pos':(n<0?'neg':'neu');
const sgn=n=>(n>0?'+':'')+f(n);

async function loadStatus(){
  try{
    const r=await fetch(tp('/api/status'),{cache:'no-store'}); if(!r.ok)throw 0;
    const s=await r.json();
    const online=s.online;
    document.getElementById('dot').style.background=online?'#27c08a':'#ff5d6c';
    document.getElementById('sub').textContent=online
      ? ('live · updated '+(s.snapshot_age_seconds??'?')+'s ago')
      : 'bot offline (showing last saved data)';
    document.getElementById('paper').textContent=s.paper?'PAPER':'LIVE';
    document.getElementById('paper').style.color=s.paper?'#ffb454':'#ff5d6c';
    document.getElementById('state').textContent=s.state||'—';
    document.getElementById('strategy').textContent=(s.manual_mode?'[M] ':'')+(s.strategy||'—');
    const pnl=s.realized_pnl_today??0;
    const pe=document.getElementById('pnl'); pe.textContent=sgn(pnl); pe.className='v '+cls(pnl);
    document.getElementById('wl').textContent='W '+(s.wins??0)+' · L '+(s.losses??0)+'  ·  '+(s.trades_today_count??0)+'/'+(s.max_trades??'—')+'T';
    document.getElementById('sentiment').textContent=s.sentiment||'—';
    document.getElementById('dayq').textContent=(s.regime||'—')+(s.day_quality?(' · '+s.day_quality):'');
    document.getElementById('bal').textContent=f(s.balance)+(s.balance_label?(' ('+s.balance_label+')'):'');
    document.getElementById('mode').textContent=(s.mode||'—')+(s.conditions&&s.conditions.length?(' · '+s.conditions.join(', ')):'');

    // market flow
    const fx=(id,v)=>document.getElementById(id).textContent=(v==null||v===''?'—':v);
    fx('mfVix', s.vix? s.vix.toFixed? s.vix.toFixed(1): s.vix : null);
    fx('mfIvp', s.iv_percentile!=null? Math.round(s.iv_percentile)+'%': null);
    fx('mfPcr', s.pcr!=null? Number(s.pcr).toFixed(2): null);
    fx('mfMp',  s.max_pain!=null? Number(s.max_pain).toLocaleString('en-IN'): null);
    fx('mfFii', s.fii_bias||null);
    const cw=(s.call_walls||[]).join(', '), pw=(s.put_walls||[]).join(', ');
    document.getElementById('mfWalls').textContent=(cw||pw)?('CE: '+(cw||'—')+'   ·   PE: '+(pw||'—')):'—';

    // active trade
    const a=s.active_trade, ac=document.getElementById('activeCard');
    if(a&&a.symbol){ac.style.display='';
      document.getElementById('atType').textContent=a.type||'';
      document.getElementById('atSym').textContent=a.symbol;
      document.getElementById('atEntry').textContent=f(a.entry);
      document.getElementById('atTrail').textContent=f(a.trail);
      document.getElementById('atSL').textContent=f(a.sl);
      document.getElementById('atQty').textContent=a.qty??'—';
    } else ac.style.display='none';

    // why-no-trade
    const why=s.why_no_trade||[], wc=document.getElementById('whyCard');
    if(why.length){wc.style.display='';const ul=document.getElementById('whyList');ul.innerHTML='';
      let verdict='';
      why.forEach(l=>{ if(l.startsWith('Verdict:')){verdict=l;} else {const li=document.createElement('li');li.textContent=l;ul.appendChild(li);} });
      document.getElementById('verdict').textContent=verdict;
    } else wc.style.display='none';

    // trades
    const t=s.completed_trades||[], tc=document.getElementById('tradesCard');
    if(t.length){tc.style.display='';const b=document.getElementById('tradesBody');b.innerHTML='';
      t.forEach((x,i)=>{const tr=document.createElement('tr');
        tr.innerHTML='<td>'+(i+1)+'</td><td>'+(x.type||'')+'</td><td>'+(x.symbol||'')+'</td>'+
          '<td class="'+cls(x.net)+'">'+sgn(x.net)+'</td><td>'+(x.strategy||'')+'</td>';
        b.appendChild(tr);});
    } else tc.style.display='none';
  }catch(e){ document.getElementById('sub').textContent='cannot reach API'; document.getElementById('dot').style.background='#ff5d6c'; }
}
async function loadLogs(){
  try{const r=await fetch(tp('/api/logs?lines=200'),{cache:'no-store'});const t=await r.text();
    const el=document.getElementById('logs');el.textContent=t;el.scrollTop=el.scrollHeight;
  }catch(e){document.getElementById('logs').textContent='cannot load logs';}
}
loadStatus();loadLogs();
setInterval(loadStatus,3000);
setInterval(loadLogs,6000);
</script>
</body>
</html>"""
