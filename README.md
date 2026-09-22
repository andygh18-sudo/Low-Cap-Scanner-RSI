# Crypto Low-Cap Scanner — TRUE BREAKOUT PRO v8.1

The repository contains the Streamlit dashboard plus a separate headless GitHub Actions engine.

## Background engine

`github_actions_engine_v8_1.py` is the actual scheduled scanner. It implements the v8.1 RSI, cache, pre-screen and breakout logic and sends Telegram alerts.

Key layers:
- CoinGecko Top 500
- 50–100 candidate pre-screen (default 60)
- ZEN/Horizen always retained
- cached CCXT exchange instances and OHLCV
- 1H/4H/1D/1W/1M/3M RSI
- weighted RSI 5/10/20/25/20/20%
- daily + weekly breakout structure
- volume, candle quality, ATR, OBV and EMA confirmation
- StochRSI and ADX/DMI
- BTC-relative 7D strength
- BTC-down-day resilience
- breakout lifecycle states
- Telegram alerts
- data/latest_scan.json for the dashboard

## Actions

Workflow: `.github/workflows/crypto_scanner_v8_1.yml`

- Manual: Actions → Crypto Low-Cap Scanner v8.1 → Run workflow
- Scheduled: 07:07 UK time daily
- Uses GitHub Actions secrets for Telegram credentials
- Commits the latest scan JSON back to the repository

## Required secrets

Create these at Settings → Secrets and variables → Actions:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `COINGECKO_API_KEY` (optional)

Do not put Telegram credentials in source code.
