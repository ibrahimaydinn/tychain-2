#!/usr/bin/env python3
"""
Quick connectivity test — run this in Terminal to diagnose download issues:
  python3 test_download.py
"""
import requests, pandas as pd, time

HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/121.0.0.0 Safari/537.36'),
    'Accept':          'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer':         'https://finance.yahoo.com/',
}

def test_symbol(symbol):
    sess = requests.Session()
    sess.headers.update(HEADERS)
    try:
        # Get cookies
        sess.get('https://finance.yahoo.com', timeout=15)
        # Get crumb
        crumb = None
        cr = sess.get('https://query1.finance.yahoo.com/v1/test/getcrumb', timeout=10)
        if cr.status_code == 200:
            crumb = cr.text.strip()
        print(f"  Crumb: {crumb[:20] if crumb else 'NONE'}")
        # Download
        start_ts = int(pd.Timestamp('2020-01-01').timestamp())
        end_ts   = int(pd.Timestamp('2026-03-09').timestamp()) + 86400
        params = {'period1': start_ts, 'period2': end_ts,
                  'interval': '1d', 'events': 'history',
                  'includeAdjustedClose': 'true'}
        if crumb:
            params['crumb'] = crumb
        resp = sess.get(f'https://query2.finance.yahoo.com/v8/finance/chart/{symbol}',
                        params=params, timeout=30)
        print(f"  HTTP status: {resp.status_code}")
        if resp.status_code == 200:
            data   = resp.json()
            result = data['chart']['result'][0]
            n      = len(result['timestamp'])
            close  = result['indicators']['quote'][0]['close']
            last   = next(x for x in reversed(close) if x is not None)
            print(f"  ✅ OK — {n} rows, last close = {last:.2f}")
            return True
        else:
            print(f"  ❌ FAILED: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

SYMBOLS = ['AKBNK.IS','YKBNK.IS','GARAN.IS','THYAO.IS','FROTO.IS']
for sym in SYMBOLS:
    print(f"\n{sym}")
    ok = test_symbol(sym)
    if not ok:
        time.sleep(3)
