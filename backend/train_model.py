#!/usr/bin/env python3
"""
BIST30 LSTM + HMM Financial Forecasting Backend
Usage: python3 train_model.py YKBNK.IS
Outputs a single JSON object to stdout.

Performance features:
  - 12-hour local CSV cache — repeat requests skip the download entirely
  - yf.Ticker().history() preferred over yf.download() (more reliable for Turkish stocks)
  - MC_SIMS = 40, EPOCHS = 30, batch = 32 for faster training
  - Vectorised Monte Carlo using raw NumPy arrays
"""

import sys, json, io, base64, warnings, os, time
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import requests
from curl_cffi import requests as cffi_requests
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ── config ────────────────────────────────────────────────────────────────────
SYMBOL      = sys.argv[1] if len(sys.argv) > 1 else 'YKBNK.IS'
START_DATE  = '2020-01-01'
END_DATE    = str(date.today())
SEQ_LEN     = 10
N_HMM       = 2
MC_WINDOW   = 60
MC_SIMS     = 40        # ↓ from 80 — faster, still statistically solid
MC_DAYS     = 10
SEED        = 42
EPOCHS      = 30        # ↓ from 50 — early stopping kicks in well before this
BATCH       = 32        # ↑ from 16 — larger batches train faster on CPU
TRAIN_RATIO = 0.80
MIN_ROWS    = 60
CACHE_HOURS = 12        # re-download only if cache is older than 12 hours

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── helpers ───────────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def monte_carlo_features(df, window=MC_WINDOW, num_sims=MC_SIMS, days=MC_DAYS):
    """Vectorised per-row Monte Carlo using raw NumPy — faster than pandas loops."""
    close = df['Close'].squeeze().values.astype(float)
    n     = len(close)
    ep    = np.full(n, np.nan)
    ev    = np.full(n, np.nan)
    ev95  = np.full(n, np.nan)
    for i in range(window, n):
        slc   = close[i - window:i]
        rets  = np.diff(slc) / (slc[:-1] + 1e-12)
        lr    = np.log1p(rets)
        mu, sigma = lr.mean(), lr.std() + 1e-9
        lp    = close[i - 1]
        rng   = np.random.default_rng(seed=i)
        steps = rng.normal(mu, sigma, (num_sims, days))
        sims  = lp * np.exp(np.cumsum(steps, axis=1))
        ep[i]   = sims[:, -1].mean()
        ev[i]   = sims[:, -1].std()
        ev95[i] = np.percentile(sims[:, -1], 5)
    out = df.copy()
    out['MC_Expected']   = ep
    out['MC_Volatility'] = ev
    out['MC_VaR95']      = ev95
    return out

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data

# ── chart theme ───────────────────────────────────────────────────────────────
BG    = '#0D1B2A'
CARD  = '#1A2B3C'
BLUE  = '#29B6F6'
ORANGE= '#FFA726'
GREEN = '#66BB6A'
RED   = '#EF5350'
WHITE = '#ECEFF1'
MUTED = '#78909C'
GRID  = '#1E3A5F'
PURPLE= '#AB47BC'

def style(fig, ax, title=''):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_title(title, fontsize=12, fontweight='bold', color=WHITE, pad=12)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, color=GRID, linestyle='--', alpha=0.4, linewidth=0.8)

def die(msg):
    print(json.dumps({'error': msg}))
    sys.exit(1)

# ── 1. data download with 12-hour cache ───────────────────────────────────────
cache_key  = SYMBOL.replace('.', '_')
cache_path = os.path.join(CACHE_DIR, f'{cache_key}.csv')
meta_path  = os.path.join(CACHE_DIR, f'{cache_key}.meta')


def cache_is_fresh():
    if not os.path.exists(cache_path) or not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path) as f:
            ts = float(f.read().strip())
        return (time.time() - ts) / 3600 < CACHE_HOURS
    except Exception:
        return False

def load_cache():
    try:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if len(df) >= MIN_ROWS:
            return df
    except Exception:
        pass
    return None

def save_cache(df):
    try:
        df.to_csv(cache_path)
        with open(meta_path, 'w') as f:
            f.write(str(time.time()))
    except Exception:
        pass

def _normalise(df):
    """Flatten MultiIndex columns and strip timezone."""
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = 'Date'
    return df if len(df) >= MIN_ROWS else None

# ── Method A: curl_cffi direct Yahoo Finance v8 (fastest, no yfinance overhead)
def direct_yahoo_cffi(symbol, start, end):
    """Use curl_cffi with Chrome impersonation to hit Yahoo's JSON API directly."""
    try:
        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts   = int(pd.Timestamp(end).timestamp()) + 86400

        sess = cffi_requests.Session(impersonate='chrome')
        sess.get('https://finance.yahoo.com', timeout=15)

        crumb = None
        for crumb_url in ['https://query1.finance.yahoo.com/v1/test/getcrumb',
                          'https://query2.finance.yahoo.com/v1/test/getcrumb']:
            try:
                cr = sess.get(crumb_url, timeout=10)
                if cr.status_code == 200 and cr.text.strip():
                    crumb = cr.text.strip()
                    break
            except Exception:
                pass

        params = {'period1': start_ts, 'period2': end_ts,
                  'interval': '1d', 'events': 'history',
                  'includeAdjustedClose': 'true'}
        if crumb:
            params['crumb'] = crumb

        for base in ['https://query2.finance.yahoo.com',
                     'https://query1.finance.yahoo.com']:
            try:
                resp = sess.get(f'{base}/v8/finance/chart/{symbol}',
                                params=params, timeout=30)
                if resp.status_code != 200:
                    continue
                data   = resp.json()
                result = data['chart']['result'][0]
                ts_raw = result['timestamp']
                quote  = result['indicators']['quote'][0]
                adj    = result['indicators'].get('adjclose', [{}])
                close  = adj[0].get('adjclose') if adj else None
                if close is None:
                    close = quote['close']

                df = pd.DataFrame({
                    'Open':   quote['open'],
                    'High':   quote['high'],
                    'Low':    quote['low'],
                    'Close':  close,
                    'Volume': quote['volume'],
                }, index=pd.to_datetime(ts_raw, unit='s'))
                df.dropna(subset=['Close'], inplace=True)
                return _normalise(df)
            except Exception:
                continue
    except Exception:
        pass
    return None

# ── Method B: yfinance Ticker.history() (reliable fallback) ──────────────────
def yf_ticker(symbol, start, end):
    """yfinance 1.2+ manages its own curl_cffi session — do not pass custom sessions."""
    try:
        t  = yf.Ticker(symbol)
        df = t.history(start=start, end=end, auto_adjust=True)
        return _normalise(df)
    except Exception:
        return None

# ── Method C: yf.download() ──────────────────────────────────────────────────
def yf_download(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        return _normalise(df)
    except Exception:
        return None

# ── Live price from Borsa Istanbul via bigpara ───────────────────────────────
def get_bist_live_price(symbol_short):
    """Fetch real-time price from Borsa Istanbul via bigpara API."""
    try:
        url = f'https://bigpara.hurriyet.com.tr/api/v1/borsa/hisseyuzeysel/{symbol_short}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://bigpara.hurriyet.com.tr/',
        }
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            info = data.get('data', {}).get('hisseYuzeysel', {})
            return {
                'live_price':  float(info.get('kapanis', 0)),
                'open':        float(info.get('acilis', 0)),
                'high':        float(info.get('yuksek', 0)),
                'low':         float(info.get('dusuk', 0)),
                'bid':         float(info.get('alis', 0)),
                'ask':         float(info.get('satis', 0)),
                'live_time':   info.get('tarih', ''),
            }
    except Exception:
        pass
    return None

# ── Download orchestrator ─────────────────────────────────────────────────────
def download_fresh():
    for attempt in range(1, 3):
        # A: curl_cffi direct Yahoo (fastest)
        raw = direct_yahoo_cffi(SYMBOL, START_DATE, END_DATE)
        if raw is not None:
            return raw
        # B: yfinance Ticker
        raw = yf_ticker(SYMBOL, START_DATE, END_DATE)
        if raw is not None:
            return raw
        # C: yfinance download
        raw = yf_download(SYMBOL, START_DATE, END_DATE)
        if raw is not None:
            return raw
        if attempt < 2:
            time.sleep(3)
    return None

# ── Cache check / load ────────────────────────────────────────────────────────
if cache_is_fresh():
    raw = load_cache()
    if raw is None:           # corrupted cache — re-download
        raw = download_fresh()
        if raw is not None:
            save_cache(raw)
else:
    raw = download_fresh()
    if raw is not None:
        save_cache(raw)

if raw is None:
    die(
        f"Could not download data for '{SYMBOL}'. "
        f"Please try again in 30 seconds. "
        f"If the error repeats, try a different stock."
    )

# Keep only OHLCV columns
needed = [c for c in ['Open','High','Low','Close','Volume'] if c in raw.columns]
df = raw[needed].copy()
df.dropna(inplace=True)

if len(df) < 200:
    die(f"Not enough historical data for {SYMBOL} ({len(df)} rows). Try a different stock.")

# ── 2. technical indicators ──────────────────────────────────────────────────
close = df['Close'].squeeze()
df['log_return'] = np.log(close / close.shift(1))
df['RSI']        = compute_rsi(close, 14)
df['EMA20']      = compute_ema(close, 20)
df['EMA50']      = compute_ema(close, 50)
df['EMA_diff']   = df['EMA20'] - df['EMA50']
df['Volatility'] = df['log_return'].rolling(10).std()
df['Log']        = np.log(close)
df['Returns']    = df['Log'].pct_change()
df['Range']      = (df['High'] / df['Low']) - 1
df.dropna(inplace=True)

# ── 3. monte carlo ────────────────────────────────────────────────────────────
df = monte_carlo_features(df)
df.dropna(inplace=True)

if len(df) < 100:
    die("Insufficient data after feature engineering.")

# ── 4. HMM ───────────────────────────────────────────────────────────────────
hmm_cols  = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
hmm_data  = df[hmm_cols].copy()
hmm_scaler = MinMaxScaler()
hmm_scaled = hmm_scaler.fit_transform(hmm_data)
try:
    hmm_model = hmm.GaussianHMM(n_components=N_HMM, n_iter=100,
                                  covariance_type='full', random_state=SEED)
    hmm_model.fit(hmm_scaled)
except Exception:
    # Fall back to diagonal covariance if full fails
    hmm_model = hmm.GaussianHMM(n_components=N_HMM, n_iter=100,
                                  covariance_type='diag', random_state=SEED)
    hmm_model.fit(hmm_scaled)

obs = df.copy()
obs['HMM_State']      = hmm_model.predict(hmm_scaled)
proba                  = hmm_model.predict_proba(hmm_scaled)
obs['HMM_Prob_0']     = proba[:, 0]
obs['HMM_Prob_1']     = proba[:, 1]
obs['Prev_HMM_State'] = obs['HMM_State'].shift(1)
obs['State_Change']   = (obs['HMM_State'] != obs['Prev_HMM_State']).astype(int)
obs.drop(columns=['Prev_HMM_State'], inplace=True)
obs.replace([np.inf, -np.inf], np.nan, inplace=True)
obs.dropna(inplace=True)

# ── 5. features ───────────────────────────────────────────────────────────────
SEL_WANTED = ['Close','HMM_Prob_1','Low','High','Log','RSI',
              'HMM_Prob_0','MC_VaR95','log_return','HMM_State']
SEL = [f for f in SEL_WANTED if f in obs.columns]
if 'Close' not in SEL:
    die("Close column missing from features.")

df_sel = obs[SEL].copy().dropna()
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_sel)
CI     = SEL.index('Close')

def make_sequences(data, sl=SEQ_LEN):
    X, y = [], []
    for i in range(len(data) - sl):
        X.append(data[i:i+sl])
        y.append(data[i+sl][CI])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled)
if len(X) < 40:
    die("Not enough sequence data for LSTM training.")

# ── 6. LSTM with 80/20 train/test split ───────────────────────────────────────
split    = int(len(X) * TRAIN_RATIO)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    LSTM(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train,
          epochs=EPOCHS, batch_size=BATCH, verbose=0,
          validation_data=(X_test, y_test), callbacks=[es])

# ── 7. out-of-sample metrics ──────────────────────────────────────────────────
test_preds    = model.predict(X_test, verbose=0)
pf_t          = np.zeros((len(test_preds), scaled.shape[1]))
pf_t[:, CI]   = test_preds[:, 0]
restored_test = scaler.inverse_transform(pf_t)[:, CI]
real_test     = df_sel['Close'].iloc[SEQ_LEN + split : SEQ_LEN + split + len(restored_test)].values
dates_test    = df_sel.index[SEQ_LEN + split : SEQ_LEN + split + len(restored_test)]

mae  = float(mean_absolute_error(real_test, restored_test))
mse  = float(mean_squared_error(real_test, restored_test))
rmse = float(np.sqrt(mse))
mape = float(mean_absolute_percentage_error(real_test, restored_test) * 100)

# Full predictions for charts
all_preds    = model.predict(X, verbose=0)
pf_a         = np.zeros((len(all_preds), scaled.shape[1]))
pf_a[:, CI]  = all_preds[:, 0]
restored     = scaler.inverse_transform(pf_a)[:, CI]
real         = df_sel['Close'].iloc[SEQ_LEN : SEQ_LEN + len(restored)].values
dates        = df_sel.index[SEQ_LEN : SEQ_LEN + len(restored)]

# Next-day forecast
last_seq       = np.expand_dims(scaled[-SEQ_LEN:], axis=0)
next_s         = model.predict(last_seq, verbose=0)
dummy          = np.zeros((1, scaled.shape[1]))
dummy[0, CI]   = next_s[0, 0]
next_day_price = float(scaler.inverse_transform(dummy)[0, CI])
last_price     = float(df_sel['Close'].iloc[-1])
price_change   = ((next_day_price - last_price) / last_price) * 100

# ── 8. signal analysis ────────────────────────────────────────────────────────
current_rsi   = float(obs['RSI'].iloc[-1])
current_ema20 = float(obs['EMA20'].iloc[-1])
current_ema50 = float(obs['EMA50'].iloc[-1])
current_prob1 = float(obs['HMM_Prob_1'].iloc[-1])

score = 0
if   price_change >  1.5: score += 2
elif price_change >  0.3: score += 1
elif price_change < -1.5: score -= 2
elif price_change < -0.3: score -= 1
if   current_rsi < 30: score += 2
elif current_rsi < 45: score += 1
elif current_rsi > 70: score -= 2
elif current_rsi > 55: score -= 1
score += 1 if current_ema20 > current_ema50 else -1
if   current_prob1 > 0.60: score += 1
elif current_prob1 < 0.40: score -= 1

if   score >= 4:  action = "STRONG BUY"
elif score >= 2:  action = "BUY"
elif score >= -1: action = "HOLD"
elif score >= -3: action = "SELL"
else:             action = "STRONG SELL"

strength = min(100, max(0, int((score + 6) / 12 * 100)))

# RSI
if current_rsi > 70:
    rsi_label = f"Overbought ({current_rsi:.0f})"
    rsi_desc  = (f"RSI above 70 — the stock has risen a lot recently and may be due "
                 f"for a pullback. Buying now carries higher risk.")
elif current_rsi > 60:
    rsi_label = f"Strong Momentum ({current_rsi:.0f})"
    rsi_desc  = ("RSI 60–70 — solid upward momentum but approaching overbought territory. Watch carefully.")
elif current_rsi < 30:
    rsi_label = f"Oversold ({current_rsi:.0f})"
    rsi_desc  = (f"RSI below 30 — the stock has fallen sharply and may be undervalued. "
                 f"A recovery or bounce is possible.")
elif current_rsi < 40:
    rsi_label = f"Weak Momentum ({current_rsi:.0f})"
    rsi_desc  = ("RSI 30–40 — sellers are in control. Wait for RSI to stabilise above 40 before buying.")
else:
    rsi_label = f"Neutral ({current_rsi:.0f})"
    rsi_desc  = ("RSI 40–60 — balanced market. No strong buy or sell signal from momentum alone.")

# Trend
if current_ema20 > current_ema50:
    trend      = "Uptrend"
    trend_desc = ("The 20-day average is ABOVE the 50-day average — a 'golden cross' pattern. "
                  "Short-term buyers are stronger than long-term sellers. Bullish sign.")
else:
    trend      = "Downtrend"
    trend_desc = ("The 20-day average is BELOW the 50-day average — a 'death cross' pattern. "
                  "Short-term selling pressure is dominating. Bearish sign.")

# HMM regime
if current_prob1 > 0.60:
    hmm_label = "Bull Regime"
    hmm_desc  = (f"The AI is {current_prob1*100:.0f}% confident the market is in an uptrend regime — "
                 f"conditions generally favour buyers.")
elif current_prob1 < 0.40:
    hmm_label = "Bear Regime"
    hmm_desc  = (f"The AI is {(1-current_prob1)*100:.0f}% confident the market is in a downtrend regime. "
                 f"Proceed cautiously.")
else:
    hmm_label = "Transitioning"
    hmm_desc  = ("The market is switching between bull and bear regimes — uncertainty is elevated. "
                 "Smaller positions are advisable.")

# Key levels
if action in ("BUY", "STRONG BUY"):
    entry_low  = round(last_price * 0.990, 2)
    entry_high = round(last_price * 1.005, 2)
    target_prc = round(max(next_day_price, last_price * 1.015), 2)
    stop_loss  = round(last_price * 0.950, 2)
elif action in ("SELL", "STRONG SELL"):
    entry_low  = round(last_price * 0.995, 2)
    entry_high = round(last_price * 1.010, 2)
    target_prc = round(min(next_day_price, last_price * 0.980), 2)
    stop_loss  = round(last_price * 1.050, 2)
else:
    entry_low  = round(last_price * 0.975, 2)
    entry_high = round(last_price * 1.025, 2)
    target_prc = round(next_day_price, 2)
    stop_loss  = round(last_price * 0.950, 2)

# Summary
direction = "rise" if price_change > 0 else "fall"
summary   = (f"The AI model predicts {SYMBOL} will {direction} "
             f"{abs(price_change):.2f}% tomorrow — from {last_price:.2f} to "
             f"{next_day_price:.2f} TRY. ")
if current_rsi > 70:
    summary += (f"RSI at {current_rsi:.0f} warns the stock is overbought — "
                f"consider waiting for a dip before entering. ")
elif current_rsi < 30:
    summary += (f"RSI at {current_rsi:.0f} signals the stock is oversold — "
                f"a bounce back may be near. ")
else:
    summary += f"RSI at {current_rsi:.0f} is neutral. "
summary += ("Trend is bullish (EMA20 > EMA50). " if current_ema20 > current_ema50
            else "Trend is bearish (EMA20 < EMA50). ")
if action == "STRONG BUY":
    summary += ("All major indicators point up. Consider buying near the entry zone — always set a stop-loss to protect against surprises.")
elif action == "BUY":
    summary += ("Most indicators are positive. Consider buying near the entry zone with a stop-loss to limit downside risk.")
elif action == "HOLD":
    summary += ("Signals are mixed. If you already own this stock, hold for now. If not, wait for a clearer signal before entering.")
elif action == "SELL":
    summary += ("Most indicators suggest caution. If you own this stock, consider setting a stop-loss or reducing your position.")
else:
    summary += ("Multiple warning signs are active. If you hold this stock, consider a tight stop-loss or taking profits.")

# ── 9. backtest on test set ───────────────────────────────────────────────────
n_bt     = min(len(real_test), 100)
close_bt = real_test[-n_bt:]
pred_bt  = restored_test[-n_bt:]
capital  = 100_000.0
cash, position = capital, 0.0
trade_log = []

for i in range(len(close_bt) - 1):
    dn = close_bt[i]   - pred_bt[i]
    dx = close_bt[i+1] - pred_bt[i+1]
    if dn * dx < 0:
        a  = abs(dn) / (abs(dn) + abs(dx) + 1e-12)
        cp = float(close_bt[i] + a * (close_bt[i+1] - close_bt[i]))
        if dn < 0 and cash > 0:
            position = cash / cp; cash = 0.0
            trade_log.append(('Buy', i+a, cp))
        elif dn > 0 and position > 0:
            cash = position * cp; position = 0.0
            trade_log.append(('Sell', i+a, cp))

final_value  = cash if position == 0 else position * float(close_bt[-1])
total_profit = final_value - capital
profit_pct   = (total_profit / capital) * 100
num_trades   = sum(1 for t in trade_log if t[0] == 'Buy')
pairs        = [(trade_log[i], trade_log[i+1])
                for i in range(0, len(trade_log)-1, 2)
                if trade_log[i][0]=='Buy' and trade_log[i+1][0]=='Sell']
success      = sum(1 for b, s in pairs if s[2] > b[2])
success_rate = (success / len(pairs) * 100) if pairs else 0.0

# ── 10. charts ────────────────────────────────────────────────────────────────
test_start = dates_test[0] if len(dates_test) > 0 else None

# chart 1
fig1, ax1 = plt.subplots(figsize=(11, 5))
style(fig1, ax1, f'{SYMBOL} — Last 100 Days: Actual vs LSTM Prediction')
n_show = min(100, len(dates))
ax1.plot(dates[-n_show:], real[-n_show:],
         color=BLUE, lw=2, marker='o', ms=3, label='Actual Price')
ax1.plot(dates[-n_show:], restored[-n_show:],
         color=ORANGE, lw=2, linestyle='--', marker='x', ms=3, label='Model Prediction')
if test_start is not None:
    ax1.axvspan(test_start, dates[-1], alpha=0.08, color=GREEN, label='Out-of-sample zone')
ax1.set_xlabel('Date', color=MUTED); ax1.set_ylabel('Price (TRY)', color=MUTED)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.legend(facecolor=CARD, edgecolor=GRID, labelcolor=WHITE, fontsize=9)
plt.xticks(rotation=45); plt.tight_layout()
c1 = fig_to_b64(fig1)

# chart 2
fig2, ax2 = plt.subplots(figsize=(11, 5))
style(fig2, ax2, f'{SYMBOL} — Full History: Actual vs Predicted')
ax2.plot(dates, real,     color=BLUE,   lw=1.5, label='Actual Price',     alpha=0.9)
ax2.plot(dates, restored, color=ORANGE, lw=1.5, label='Model Prediction', alpha=0.85)
if test_start is not None:
    ax2.axvline(x=test_start, color=PURPLE, lw=1.5, linestyle=':', label='Train / Test split')
    ax2.axvspan(test_start, dates[-1], alpha=0.07, color=GREEN)
ax2.set_xlabel('Date', color=MUTED); ax2.set_ylabel('Price (TRY)', color=MUTED)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.legend(facecolor=CARD, edgecolor=GRID, labelcolor=WHITE, fontsize=9)
plt.xticks(rotation=45); plt.tight_layout()
c2 = fig_to_b64(fig2)

# chart 3
fig3, ax3 = plt.subplots(figsize=(11, 5))
style(fig3, ax3, f'{SYMBOL} — Trading Strategy Backtest (Out-of-Sample, {n_bt} Days)')
ax3.plot(close_bt, color=BLUE,  lw=2,   label='Actual Price')
ax3.plot(pred_bt,  color=WHITE, lw=1.5, linestyle='--', alpha=0.6, label='Model Prediction')
bp = sp = False
for act, idx, price in trade_log:
    if act == 'Buy':
        ax3.scatter(idx, price, marker='^', color=GREEN, s=130,
                    label='Buy signal' if not bp else '_nolegend_', zorder=5)
        bp = True
    else:
        ax3.scatter(idx, price, marker='v', color=RED, s=130,
                    label='Sell signal' if not sp else '_nolegend_', zorder=5)
        sp = True
ax3.set_xlabel('Day Index', color=MUTED); ax3.set_ylabel('Price (TRY)', color=MUTED)
ax3.legend(facecolor=CARD, edgecolor=GRID, labelcolor=WHITE, fontsize=9)
plt.tight_layout()
c3 = fig_to_b64(fig3)

# ── 11. output ────────────────────────────────────────────────────────────────
# Fetch live BIST price (non-blocking — skip on failure)
symbol_short = SYMBOL.replace('.IS', '')
bist_live    = get_bist_live_price(symbol_short)

output = {
    'symbol':         SYMBOL,
    'next_day_price': round(next_day_price, 2),
    'last_price':     round(last_price, 2),
    'price_change':   round(price_change, 2),
    'last_date':      str(df_sel.index[-1].date()),
    'metrics': {
        'mae':  round(mae,  4),
        'mse':  round(mse,  4),
        'rmse': round(rmse, 4),
        'mape': round(mape, 2),
    },
    'signal': {
        'action':     action,
        'strength':   strength,
        'score':      score,
        'rsi':        round(current_rsi, 1),
        'rsi_label':  rsi_label,
        'rsi_desc':   rsi_desc,
        'trend':      trend,
        'trend_desc': trend_desc,
        'hmm_label':  hmm_label,
        'hmm_desc':   hmm_desc,
        'entry_low':  entry_low,
        'entry_high': entry_high,
        'target':     target_prc,
        'stop_loss':  stop_loss,
        'summary':    summary,
    },
    'trading': {
        'final_value':  round(float(final_value), 2),
        'total_profit': round(float(total_profit), 2),
        'profit_pct':   round(float(profit_pct), 2),
        'num_trades':   num_trades,
        'success_rate': round(success_rate, 1),
        'test_days':    n_bt,
    },
    'charts': {
        'last_100_days':    c1,
        'full_range':       c2,
        'trading_strategy': c3,
    }
}

if bist_live:
    output['bist_live'] = bist_live

print(json.dumps(output))
