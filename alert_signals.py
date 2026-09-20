import json, os

STATE = 'data/alert_state.json'


def qualifying(payload):
    out = []
    for row in payload.get('top10', []):
        score = float(row.get('score') or 0)
        signal = row.get('signal', '')
        if score >= 75 or signal.startswith('🚀') or signal.startswith('🟠'):
            out.append({
                'id': row.get('id') or row.get('ticker'),
                'ticker': row.get('ticker'),
                'score': round(score, 1),
                'signal': signal,
            })
    return out


def main():
    with open('data/latest_scan.json', 'r', encoding='utf-8') as f:
        payload = json.load(f)
    current = qualifying(payload)
    try:
        with open(STATE, 'r', encoding='utf-8') as f:
            raw = f.read().strip()
            old = json.loads(raw) if raw else []
            if not isinstance(old, list): old = []
    except (FileNotFoundError, json.JSONDecodeError):
        old = []
    old_keys = {(x.get('id'), x.get('signal')) for x in old}
    new = [x for x in current if (x.get('id'), x.get('signal')) not in old_keys]
    with open(STATE, 'w', encoding='utf-8') as f:
        json.dump(current, f, indent=2)
    with open('data/new_alerts.json', 'w', encoding='utf-8') as f:
        json.dump(new, f, indent=2)
    print(f'New qualifying alerts: {len(new)}')
    for x in new:
        print(x)


if __name__ == '__main__':
    main()
