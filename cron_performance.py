#!/usr/bin/env python3
import time
import sqlite3
import os
import yfinance as yf
import pandas as pd
from datetime import datetime

import app

def clog(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def run_performance_fetch():
    clog('=== Tychain Performance Fetch Start ===')
    
    bist_tickers = list(app.BIST30_STOCKS.keys())
    sp500_tickers = list(app.SP500_STOCKS.keys())
    
    # yfinance uses .IS for BIST
    yf_bist = [t + '.IS' for t in bist_tickers]
    yf_sp500 = sp500_tickers
    
    all_yf_tickers = yf_bist + yf_sp500
    
    # Fetch 2y so we always have ≥ 252 trading days to compute the 1Y change.
    # Previously period="1y" returned exactly ~252 rows, and the `len(df) > 252`
    # guard below failed → 1Y column rendered as 0.00% for every ticker.
    clog(f"Fetching 2y data for {len(all_yf_tickers)} tickers...")
    data = yf.download(" ".join(all_yf_tickers), period="2y", interval="1d", group_by="ticker", threads=True, progress=False)
    
    records = []
    
    for yf_tick in all_yf_tickers:
        is_bist = yf_tick.endswith('.IS')
        base_tick = yf_tick.replace('.IS', '') if is_bist else yf_tick
        market = 'BIST30' if is_bist else 'SP500'
        
        try:
            # yfinance returns multiindex columns if multiple tickers
            if len(all_yf_tickers) == 1:
                df = data
            else:
                df = data[yf_tick]
            
            df = df.dropna(subset=['Close'])
            if len(df) < 2:
                continue
                
            last_price = df['Close'].iloc[-1]
            
            def get_change(days):
                """Return (abs, pct) change vs. the price `days` trading-days
                ago. Falls back to the earliest available price for tickers
                with shorter listing history (so newly-IPO'd S&P names still
                get a 1Y number instead of 0.00%)."""
                if len(df) <= 1:
                    return 0.0, 0.0
                if len(df) > days:
                    past_price = df['Close'].iloc[-days - 1]
                else:
                    past_price = df['Close'].iloc[0]   # fallback: oldest we have
                if past_price == 0:
                    return 0.0, 0.0
                abs_chg = last_price - past_price
                pct_chg = (abs_chg / past_price) * 100
                return abs_chg, pct_chg
                
            abs_1d, pct_1d = get_change(1)
            abs_1w, pct_1w = get_change(5)
            abs_1m, pct_1m = get_change(21)
            abs_1y, pct_1y = get_change(252)
            
            # Use Python native types instead of pandas types
            records.append((
                base_tick, market, float(last_price), 
                float(abs_1d), float(pct_1d), 
                float(abs_1w), float(pct_1w), 
                float(abs_1m), float(pct_1m), 
                float(abs_1y), float(pct_1y)
            ))
            
        except Exception as e:
            clog(f"Error processing {yf_tick}: {e}")
            
    clog(f"Processed {len(records)} records. Saving to DB...")
    
    with app.get_db() as conn:
        conn.execute("DELETE FROM performance_analytics")
        conn.executemany("""
            INSERT INTO performance_analytics
            (ticker, market, price, abs_1d, pct_1d, abs_1w, pct_1w, abs_1m, pct_1m, abs_1y, pct_1y)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        
    clog('=== Performance Fetch Complete ===')

if __name__ == '__main__':
    # Initialize DB so table exists
    app.init_db()
    run_performance_fetch()
