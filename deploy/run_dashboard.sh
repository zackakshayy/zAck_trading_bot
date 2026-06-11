#!/usr/bin/env bash
# Launch the dashboard bound to the TAILSCALE interface only — never 0.0.0.0.
# If Tailscale isn't up yet, fall back to loopback (still private).
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="127.0.0.1"
if command -v tailscale >/dev/null 2>&1; then
    TS_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
    [ -n "${TS_IP}" ] && HOST="${TS_IP}"
fi

echo "dashboard binding to ${HOST}:8000 (Tailscale-private)"
exec ./venv/bin/uvicorn dashboard:app --host "${HOST}" --port 8000 --no-access-log
