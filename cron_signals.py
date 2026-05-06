#!/usr/bin/env python3
import os
import time
from datetime import datetime

import app
import email_helper

APP_URL = os.environ.get('APP_URL', 'http://localhost:8080')

def clog(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def run_cron():
    clog('=== Tychain Signal Scan Start ===')
    
    with app.app.app_context():
        tickers = app.db_all_unique_tickers()
        
        if not tickers:
            clog('No tracked stocks found. Add stocks on the dashboard first.')
            return {'scanned': 0, 'skipped': 0, 'failed': 0, 'alerts_sent': 0}
            
        clog(f'Unique tickers to check: {", ".join(tickers)}')
        
        scanned = 0
        skipped = 0
        failed = 0
        alerts_sent = 0
        
        for ticker in tickers:
            clog(f"→ {ticker}")
            
            if app.db_signal_is_fresh(ticker):
                clog("  ✓ Signal is fresh — skipping")
                skipped += 1
                continue
                
            data = app.run_analysis(ticker)
            if 'error' in data:
                clog(f"  ✗ Analysis failed for {ticker}: {data['error']}")
                failed += 1
                continue
                
            app.persist_signal(ticker, data)
            sig = data['signal']
            
            action = sig.get('action', 'HOLD')
            strength = int(sig.get('strength', 0))
            last_price = float(data.get('last_price', 0))
            next_price = float(data.get('next_day_price', 0))
            summary = sig.get('summary', '')
            currency_symbol = data.get('currency_symbol', '₺')
            
            clog(f"  ✓ Signal: {action} ({strength}%) | Price: {currency_symbol}{last_price}")

            # Only fire alerts for directional signals (BUY/SELL family) above threshold
            is_alertable = (
                action in app.ALERT_SIGNAL_TYPES
                and strength >= app.ALERT_THRESHOLD
            )
            if is_alertable:
                clog(f"  🔔 {action} ≥ {app.ALERT_THRESHOLD}% — checking for users to alert")
                users = app.db_users_tracking(ticker)
                clog(f"  Found {len(users)} user(s) tracking {ticker}")
                
                for u in users:
                    uid = u['id']
                    email = u['email']
                    
                    if app.db_already_notified(uid, ticker, action):
                        clog(f"  ↳ {email} — already notified today, skipping")
                        continue
                        
                    clog(f"  ↳ Sending alert to {email} …")
                    result = email_helper.email_signal_alert(
                        to_email=email,
                        to_name=email,
                        ticker=ticker,
                        signal_type=action,
                        strength=strength,
                        price=last_price,
                        next_price=next_price,
                        summary=summary,
                        currency_symbol=currency_symbol,
                        dashboard_url=APP_URL,
                    )
                    
                    if result is True:
                        app.db_log_email(uid, ticker, action, strength)
                        alerts_sent += 1
                        clog("  ↳ ✓ Email sent")
                    else:
                        clog(f"  ↳ ✗ Email failed: {result}")
                        
            scanned += 1
            if len(tickers) > 1:
                time.sleep(2)
                
        clog("=== Scan Complete ===")
        clog(f"Scanned: {scanned} | Skipped (fresh): {skipped} | Failed: {failed} | Alerts sent: {alerts_sent}")
        
        return {
            'scanned': scanned,
            'skipped': skipped,
            'failed': failed,
            'alerts_sent': alerts_sent
        }

if __name__ == '__main__':
    run_cron()
