# Crypto Low-Cap Scanner — Always-On + Telegram Top 10

This version runs independently of Streamlit using GitHub Actions and sends a compact **Top 10 report on every scheduled run**, plus a separate message when a new qualifying signal appears.

## Included
- CoinGecko market-cap/volume screening
- Low-cap universe plus **ZEN always included**
- BTC-relative 7D strength
- Volume / market cap
- 1D / 1W / 1M / 3M RSI
- BTC downside beta and BTC-down-day relative performance
- 15-point resilience component
- 0–100 unified score
- RISK-ON / MIXED / RISK-OFF regime penalty
- Top 10 saved to `data/latest_scan.json` and CSV
- Telegram Top 10 report every run
- New breakout / high-score / extended signal alerts
- Four scheduled runs per day at 06:07, 12:07, 18:07 and 23:07 Europe/London

## GitHub secrets
Create these repository secrets:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

The Telegram Bot API uses HTTPS requests to `sendMessage`; keep the bot token in GitHub Secrets rather than source code.

## Manual run
GitHub → Actions → Crypto Scanner → Run workflow.

## Streamlit
The Streamlit app can remain a dashboard only. The background scan does not depend on the Streamlit app staying awake.
