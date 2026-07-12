# -crypto-price-prediction-hybrid-dl
MSc Data Science Dissertation — Cryptocurrency Price Prediction Using Hybrid Deep Learning Models
# Cryptocurrency Price Prediction Using Hybrid Deep Learning Models

**MSc Data Science Dissertation**
**Student:** Abiodun Adeyinka Adetoro | A00065756
**University:** University of Roehampton London
**Supervisor:** Dr Michael

---

## Project Overview

This project investigates whether hybrid deep learning architectures
can produce statistically superior forecasts of cryptocurrency log
returns compared to standalone and naive baselines.

Six models implemented and evaluated:
- **Baselines:** ARIMA, Standalone LSTM, Standalone GRU
- **Hybrids:** CNN-LSTM, LSTM with Attention, Transformer-LSTM

Four cryptocurrencies studied:
Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB), Litecoin (LTC)

---

## Repository Contents

| File | Description |
|---|---|
| `crypto_dataset_pipeline_v2.py` | Downloads OHLCV data, engineers 16 features, walk-forward splits |
| `crypto_models_v3.py` | Trains and evaluates all 6 models with Diebold-Mariano testing |
| `crypto_charts_v2.py` | Generates observed vs predicted charts |
| `all_results_summary.csv` | Final RMSE and MAE for all models across all four coins |

---

## How to Run

**Install dependencies:**
pip install yfinance ta pandas numpy scikit-learn statsmodels
tensorflow matplotlib seaborn requests

**Run pipeline:**
python crypto_dataset_pipeline_v2.py

**Train and evaluate models:**
python crypto_models_v3.py

**Generate charts:**
python crypto_charts_v2.py

---

## Key Results

Transformer-LSTM achieved the lowest RMSE among hybrid architectures
across all four cryptocurrencies.

Diebold-Mariano test results:
- Bitcoin vs Persistence: p = 0.0034 (p < 0.01)
- Ethereum vs Persistence: p = 0.0005 (p < 0.01)

---

## Technologies

Python 3.10 | TensorFlow 2.x | Keras | scikit-learn |
yfinance | statsmodels | pandas | NumPy | Matplotlib
