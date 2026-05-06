# BIST Project — VS Code Setup Guide

## Project: Stochastic-Driven Deep Learning for Financial Recommendation Systems
Borsa Istanbul (BIST30) Application — YKBNK.IS

---

## Folder Structure

```
BIST_PROJECT/
├── BIST_LSTM_HMM.ipynb      ← Main notebook (run this)
├── Final Raporu.ipynb        ← Original notebook (reference copy)
├── requirements.txt          ← Python dependencies
├── SETUP.md                  ← This file
├── backend/
│   ├── train_model.py        ← Flask backend entry point (optional)
│   └── frontend/
│       └── index.php
└── venv/                     ← Virtual environment (if already set up)
```

---

## Quick Start (VS Code)

### Step 1 — Create a virtual environment

```bash
# Navigate to project folder
cd BIST_PROJECT

# Create venv (Python 3.10+)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on TensorFlow:** If you have an NVIDIA GPU, install `tensorflow[and-cuda]` instead.
> For CPU-only machines `tensorflow` (CPU build) is installed by default.

> **Note on TA-Lib:** The original code used `talib`. This has been replaced with
> pure-pandas RSI/EMA implementations in `BIST_LSTM_HMM.ipynb`, so **no C library
> installation is required**.

### Step 3 — Open the notebook in VS Code

1. Open VS Code and install the **Jupyter** extension (if not already installed).
2. Open `BIST_LSTM_HMM.ipynb`.
3. In the top-right kernel picker, select the `venv` interpreter
   (`Python 3.x.x ('venv')`).
4. Run all cells with **Run All** (`Ctrl+Shift+P` → *Notebook: Run All Cells*).

---

## What the Notebook Does

| Cell | Description |
|------|-------------|
| 1 | Imports & configuration (symbol, dates, hyperparameters) |
| 2 | Download YKBNK.IS OHLCV from Yahoo Finance (2020–2025) |
| 3 | Technical indicators: RSI, EMA20/50, Volatility, Log Return, Range |
| 4 | Monte Carlo simulation features: MC_Expected, MC_Volatility, MC_VaR95 |
| 5 | HMM regime features: HMM_State, HMM_Prob_0/1, State_Change |
| 6 | Normalisation (MinMaxScaler) + sliding-window LSTM sequences |
| 7 | Train LSTM(64) model — full feature set |
| 8 | Predictions + MAE / MSE / RMSE / MAPE metrics |
| 9 | Full-history and last-100-days forecast plots |
| 10 | Next-day price prediction |
| 11 | Feature Elimination analysis (RMSE when each feature removed) |
| 12 | Feature elimination bar charts (RMSE, MAE, MSE, MAPE) |
| 13 | SHAP summary plot (XGBoost surrogate) |
| 14 | Permutation importance (LSTM) |
| 15 | Integrated Gradients attribution |
| 16 | Final optimised model — top 10 selected features |
| 17 | Final model plots (full history, last 100 & 200 days) |
| 18 | Crossover trading strategy backtest (200-day window) |
| 19 | Trade summary table (buy/sell pairs, P&L, success rate) |

---

## Expected Runtime

| Stage | Approximate time (CPU) |
|-------|------------------------|
| Monte Carlo simulation | ~30–60 s |
| LSTM training (Cell 7) | ~3–8 min |
| Feature elimination (Cell 11) | ~45–90 min |
| SHAP analysis (Cell 13) | ~1–2 min |
| Final model (Cell 16) | ~2–5 min |

> To speed up feature elimination, reduce `epochs=20` to `epochs=10` inside `evaluate_without_feature()`.

---

## Changing the Stock Symbol

Edit **Cell 1** (`cell-imports`):

```python
SYMBOL     = 'YKBNK.IS'   # change to any BIST30 ticker, e.g. 'FROTO.IS'
START_DATE = '2020-01-01'
END_DATE   = '2025-05-17'
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: yfinance` | Run `pip install -r requirements.txt` |
| `ModuleNotFoundError: talib` | Not needed — TA-Lib has been replaced with pure-pandas code |
| `ValueError: Input 0 is incompatible` | Ensure you run cells **in order** from top to bottom |
| TensorFlow CUDA warnings | Safe to ignore if running on CPU; install `tensorflow[and-cuda]` for GPU |
| yfinance columns issue | Already handled — code flattens multi-level column headers automatically |
