#!/usr/bin/env bash
# Fired by systemd OnFailure= when the bot service crashes hard (after its
# restart budget). Reads the ntfy topic from config.yaml; silent if unset.
set -u
cd "$(dirname "$0")/.."

read -r SERVER TOPIC <<EOF
$(./venv/bin/python - <<'PY'
import yaml
c = yaml.safe_load(open("config.yaml")) or {}
n = c.get("ntfy") or {}
print((n.get("server") or "https://ntfy.sh").strip(), (n.get("topic") or "").strip())
PY
)
EOF

[ -z "${TOPIC}" ] && exit 0
curl -s -m 8 \
    -H "Title: zAck — BOT SERVICE FAILED" \
    -H "Priority: urgent" \
    -H "Tags: rotating_light" \
    -d "The bot service crashed and stopped. Check the dashboard logs. If a position is open, the broker SL-M still protects it — verify in the Kite app." \
    "${SERVER%/}/${TOPIC}" >/dev/null || true
