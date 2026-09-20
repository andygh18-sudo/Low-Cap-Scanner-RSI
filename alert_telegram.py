import json, os, requests

API = 'https://api.telegram.org/bot{}/sendMessage'


def send_telegram(text: str) -> None:
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print('Telegram secrets not configured; skipping alert.')
        return
    # Telegram sendMessage accepts up to 4096 characters.
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        r = requests.post(API.format(token), json={
            'chat_id': chat_id,
            'text': chunk,
            'disable_web_page_preview': True,
        }, timeout=30)
        r.raise_for_status()


def n(row, key, default=0.0):
    try:
        v = row.get(key, default)
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def top10_text(payload):
    meta = payload.get('meta', {})
    top = payload.get('top10', [])
    lines = [
        '📊 CRYPTO SCANNER — TOP 10',
        f"Regime: {meta.get('regime', 'N/A')}",
        f"BTC: {n(meta,'btc_24h'):.2f}% 24h | {n(meta,'btc_7d'):.2f}% 7d | Breadth {n(meta,'breadth'):.0f}%",
        '',
    ]
    for i, row in enumerate(top[:10], 1):
        lines.append(
            f"{i}. {row.get('ticker','?')} {n(row,'score'):.1f}/100 — {row.get('signal','') }\n"
            f"   BTC-rel {n(row,'btc_rel_7d_pct'):+.1f}% | Vol/MCap {n(row,'vol_mcap_pct'):.1f}% | "
            f"RSI {n(row,'weighted_rsi',50):.1f} | Downβ {n(row,'downside_beta',0):.2f} | "
            f"Res {n(row,'resilience_score'):.1f}/15"
        )
    return '\n'.join(lines)


def new_alert_text(payload, new):
    if not new:
        return ''
    meta = payload.get('meta', {})
    lines = [
        '🚨 NEW CRYPTO SIGNAL',
        f"Regime: {meta.get('regime','N/A')}",
        f"BTC 24h {n(meta,'btc_24h'):.2f}% | BTC 7d {n(meta,'btc_7d'):.2f}%",
        '',
    ]
    for x in new:
        lines.append(f"{x['ticker']} — {x['score']}/100 — {x['signal']}")
    return '\n'.join(lines)


def main():
    with open('data/latest_scan.json', encoding='utf-8') as f:
        payload = json.load(f)
    new = []
    if os.path.exists('data/new_alerts.json'):
        with open('data/new_alerts.json', encoding='utf-8') as f:
            new = json.load(f)
    # Always send the latest Top 10 report; add a separate section when a new signal appears.
    send_telegram(top10_text(payload))
    extra = new_alert_text(payload, new)
    if extra:
        send_telegram(extra)


if __name__ == '__main__':
    main()
