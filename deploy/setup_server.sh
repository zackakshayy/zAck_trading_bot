#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# zAck one-shot server setup — Ubuntu 22.04/24.04 (AWS Lightsail, Mumbai).
#
# Run as the default 'ubuntu' user AFTER:
#   1. Tailscale is installed and up:  curl -fsSL https://tailscale.com/install.sh | sh
#                                      sudo tailscale up --ssh
#   2. The repo is cloned to:          ~/zack_trading_bot
#   3. config.yaml is copied over:     (from your Mac, via Tailscale)
#      scp config.yaml ubuntu@<server-tailscale-name>:~/zack_trading_bot/
#
# Then:  cd ~/zack_trading_bot && bash deploy/setup_server.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
APP="$HOME/zack_trading_bot"
cd "$APP"

echo "── [1/7] System packages ──────────────────────────────────────────────"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-dev build-essential \
    curl wget ufw

echo "── [2/7] Timezone → Asia/Kolkata (market hours + systemd timer) ──────"
sudo timedatectl set-timezone Asia/Kolkata

echo "── [3/7] TA-Lib C library (official .deb) ────────────────────────────"
if ! ldconfig -p | grep -q libta-lib; then
    ARCH="$(dpkg --print-architecture)"   # amd64 | arm64
    TA_VER="0.6.4"
    wget -q "https://github.com/ta-lib/ta-lib/releases/download/v${TA_VER}/ta-lib_${TA_VER}_${ARCH}.deb" \
        -O /tmp/ta-lib.deb
    sudo dpkg -i /tmp/ta-lib.deb || sudo apt-get -f install -y
    rm -f /tmp/ta-lib.deb
fi

echo "── [4/7] Python venv + dependencies ──────────────────────────────────"
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install --upgrade pip wheel
./venv/bin/pip install -r requirements.txt

echo "── [5/7] Firewall: deny ALL inbound except the Tailscale interface ───"
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow in on tailscale0
sudo ufw --force enable

echo "── [6/7] systemd user services ───────────────────────────────────────"
mkdir -p "$HOME/.config/systemd/user"
cp deploy/zack-bot.service deploy/zack-bot.timer \
   deploy/zack-dashboard.service deploy/zack-fail-notify.service \
   "$HOME/.config/systemd/user/"
chmod +x deploy/run_dashboard.sh deploy/notify_fail.sh
# Let user services run without an active SSH session (survives logout/reboot).
sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now zack-dashboard.service
systemctl --user enable --now zack-bot.timer

echo "── [7/7] Sanity check ────────────────────────────────────────────────"
sleep 2
systemctl --user --no-pager status zack-dashboard.service | head -5 || true
TS_IP="$(tailscale ip -4 2>/dev/null | head -1 || echo 127.0.0.1)"
curl -s "http://${TS_IP}:8000/healthz" && echo "  ← dashboard healthy"

cat <<DONE

──────────────────────────────────────────────────────────────────────────────
 Setup complete.

 Dashboard:   http://${TS_IP}:8000/        (Tailscale devices only)
 Bot starts:  Mon–Fri 08:45 IST automatically (zack-bot.timer);
              it waits for your phone Kite login, then trades.
 Manual ops:  systemctl --user start|stop|status zack-bot.service
              journalctl --user -u zack-bot.service -f

 REMINDERS
  • Set dashboard.token + dashboard.control_pin + ntfy.topic in config.yaml,
    and server.headless: true — then: systemctl --user restart zack-dashboard
  • In the Kite developer console, set the app's Redirect URL to:
        http://${TS_IP}:8000/kite/callback
  • On your phone: install Tailscale (sign in), install ntfy (subscribe to
    your topic), open the dashboard URL → Add to Home Screen.
──────────────────────────────────────────────────────────────────────────────
DONE
