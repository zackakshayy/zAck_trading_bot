# Deploying zAck to AWS (Lightsail + Tailscale + PWA)

Total cost ≈ **$5/month**. Total time ≈ **45–60 minutes**. Nothing is ever
exposed to the public internet; your phone reaches the bot only through your
private Tailscale network.

```
iPhone/Android (PWA + ntfy)
        │  Tailscale (WireGuard, no open ports)
        ▼
AWS Lightsail · Mumbai · $5/mo
  ├─ zack-bot.service       (systemd --user, starts 08:45 IST Mon–Fri)
  ├─ zack-dashboard.service (FastAPI PWA, bound to the Tailscale IP)
  └─ state/ + output/       (status, journal, logs)
        │
        ▼
Zerodha Kite API  (orders; broker-side SL-M protects open positions)
```

## 1. Create the server (10 min)

1. AWS console → **Lightsail** → *Create instance*.
2. Region: **Mumbai (ap-south-1)** · Platform: **Linux** · Blueprint: **Ubuntu 24.04 LTS**.
3. Plan: **$5/mo (1 GB RAM)** — the 512 MB tier will swap, don't.
4. Name it `zack-bot`, create. Connect via the browser SSH for the next steps.

## 2. Private network (5 min)

On the server:
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --ssh
```
Open the printed link on your phone/laptop to authorise the machine into your
tailnet. From now on use Tailscale SSH; the Lightsail firewall ports (22/80)
can be closed in the Lightsail console — the setup script also installs a
host firewall (ufw) that drops everything except the Tailscale interface.

On your **phone**: install the Tailscale app and sign in to the same account.

## 3. Code + secrets (10 min)

On the server:
```bash
git clone <your-repo-url> ~/zack_trading_bot
```
(For a private repo, create a read-only deploy key: `ssh-keygen -t ed25519`,
add the public key in GitHub → repo → Settings → Deploy keys.)

From your **Mac** (Tailscale makes the hostname work):
```bash
scp config.yaml ubuntu@zack-bot:~/zack_trading_bot/
```
`config.yaml` travels over the encrypted tailnet, never through git.

Edit it on the server — the deployment-specific keys:
```yaml
server:
  headless: true              # phone login instead of terminal input()
ntfy:
  enable: true
  topic: "zack-<long-random-string>"   # invent it; subscribe in the ntfy app
dashboard:
  token: "<long-random-string>"        # the dashboard password
  control_pin: "<6-digits>"            # arms the Arm/Kill buttons
```

## 4. One-shot setup (10 min)

```bash
cd ~/zack_trading_bot && bash deploy/setup_server.sh
```
Installs TA-Lib, the venv, the firewall, the timezone, and the systemd user
services (dashboard now; bot weekday mornings at 08:45 IST). The script ends
by printing your dashboard URL.

## 5. Kite redirect URL (2 min)

In the [Kite developer console](https://developers.kite.trade/apps), set your
app's **Redirect URL** to:
```
http://<server-tailscale-ip>:8000/kite/callback
```
The redirect happens inside YOUR phone's browser, which is on the tailnet —
Zerodha's servers never need to reach this address.

## 6. Phone install (3 min)

1. Open `http://<server-tailscale-ip>:8000/` in the phone browser
   (Tailscale app must be connected). Enter the dashboard token once.
2. **iPhone:** Safari → Share → *Add to Home Screen*.
   **Android:** Chrome → ⋮ → *Install app*.
3. ntfy app → subscribe to your topic. Done.

## A normal morning

| Time | What happens |
|---|---|
| 08:45 | Timer starts the bot. No valid token → push: **“login needed”** |
| 08:46 | You tap **Kite login** in the PWA → fingerprint → “You're logged in” |
| 08:46 | Push: **“Kite login OK”** → bot runs pre-market prep + brief |
| 09:15 | Bot trades. Every entry/exit lands as a push with net P&L |
| 15:20 | Hard exit; bot stops cleanly. Daily report email as usual |

Skipped the login? Reminder every 10 min until ~10:15, then it gives up for
the day and tells you so.

## Emergencies

- **Kill switch** (PWA → Kill, PIN): flattens any position at market, stops
  the bot, confirms by push. If the flatten fails it screams at you to open
  the Kite app — the broker-side SL-M is still working regardless.
- **Bot crash**: systemd restarts it (it reconciles the open position on
  startup). A hard repeated failure fires an urgent ntfy alert.
- **Lost phone**: `tailscale logout` the device from any other Tailscale
  login, or remove it in the Tailscale admin console. The dashboard token
  alone is useless outside the tailnet.

## Updating the bot

```bash
ssh ubuntu@zack-bot
cd ~/zack_trading_bot && git pull && systemctl --user restart zack-dashboard
```
(The bot itself picks up changes at the next morning start.)
