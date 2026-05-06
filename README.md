---
title: Tychain Pro — BIST30 & S&P 500 AI Forecasting
emoji: 🚀
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI stock forecasting for BIST30 and S&P 500
---

# 📈 Tychain Pro — AI Stock Forecasting

> **Live Demo:** [huggingface.co/spaces/nihans/Tychain-nihan](https://huggingface.co/spaces/nihans/Tychain-nihan)

Tychain is an AI-powered stock forecasting platform covering both **Borsa Istanbul (BIST30)** and **S&P 500** markets. It combines deep learning, statistical models, and Monte Carlo simulation to generate BUY / SELL / HOLD signals with confidence scores, price targets, and plain-English explanations.

---

## ✨ Features

### 📊 AI Analysis Engine
- **LSTM Neural Network** — sequence-based price prediction
- **Hidden Markov Model (HMM)** — market regime detection (Bull / Bear / Neutral)
- **Monte Carlo Simulation** — volatility-aware feature engineering
- **Technical Indicators** — RSI, EMA, Log Returns

### 🌍 Dual Market Coverage
| Market | Stocks | Currency |
|---|---|---|
| 🇹🇷 BIST30 (Turkey) | 33 stocks | ₺ TRY |
| 🇺🇸 S&P 500 (US) | 125 stocks across 11 sectors | $ USD |

**S&P 500 sectors:** 💻 Technology · 📡 Communication · 🛒 Consumer Discretionary · 🛍️ Consumer Staples · 🏥 Healthcare · 🏦 Financials · ⚡ Energy · 🏗️ Industrials · 🧱 Materials · 🏠 Real Estate · 💡 Utilities

### 📋 Signal Output
- BUY / SELL / HOLD signal with strength score (0–100%)
- Entry Zone, Price Target, Stop Loss levels
- 3 dark-themed charts: Last 100 Days, Full Range, Trading Strategy backtest
- Model performance metrics: RMSE, MAPE, Success Rate

### 👤 User Dashboard
- Create an account and track up to 10 stocks
- Automatic signal refresh with email alerts
- Stocks grouped by market (BIST30 / S&P 500 by sector)

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/nihansupadem/Bitirme.git
cd Bitirme
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file or export directly in your shell:
```bash
export SECRET_KEY="your-random-secret-key"   # required for sessions

# Optional — only needed for email alerts
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="you@gmail.com"
export SMTP_PASS="your-app-password"
export ALERT_FROM="you@gmail.com"
```

### 5. Run the app
```bash
flask run
```

Open your browser at **http://127.0.0.1:5000**

---

## 🐳 Docker (optional)

```bash
docker build -t tychain .
docker run -p 7860:7860 -e SECRET_KEY="your-secret" tychain
```

Open **http://localhost:7860**

---

## 🔔 Automated Signal Alerts (optional)

To run background signal scans and send email alerts when thresholds are crossed:

```bash
python cron_signals.py
```

Schedule this with cron or a task runner to run every few hours.

---

## 🗂️ Project Structure

```
├── app.py                  # Flask app — routes, auth, API endpoints
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker config for deployment
├── cron_signals.py         # Background signal scanner + email alerts
├── email_helper.py         # SMTP email utility
├── backend/
│   ├── train_model.py      # BIST30 LSTM + HMM model
│   └── train_model_sp500.py# S&P 500 LSTM + HMM model
└── templates/
    ├── index.html          # Homepage — stock selector + analysis view
    ├── dashboard.html      # User dashboard — tracked stocks
    └── auth.html           # Login / signup page
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask + Jinja2 |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Statistical ML | hmmlearn (HMM) |
| Data | yfinance, curl_cffi |
| Database | SQLite |
| Plotting | matplotlib |
| Server (prod) | Gunicorn |
| Deployment | Docker · Hugging Face Spaces |

---

## ⚠️ Disclaimer

This is an **educational AI research tool — not financial advice**.  
Predictions can be wrong. Always do your own research and consult a licensed financial adviser before investing real money.
