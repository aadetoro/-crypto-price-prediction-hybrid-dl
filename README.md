# Cryptocurrency Price Prediction Using Hybrid Deep Learning Models

**MSc Data Science Dissertation**
**Student:** Abiodun Adeyinka Adetoro | A00065756
**University:** University of Roehampton London
**Supervisor:** Dr Michael

---

## Project Overview

This project investigates whether hybrid deep learning architectures can produce statistically superior forecasts of cryptocurrency log returns compared to standalone and naive baselines, using rigorous walk-forward validation and leakage-free preprocessing.

**Six models are implemented and evaluated:**
- **Baselines:** ARIMA, Standalone LSTM, Standalone GRU
- **Hybrids:** CNN-LSTM, LSTM with Attention, Transformer-LSTM

**Four cryptocurrencies are studied:**
Bitcoin (BTC-USD), Ethereum (ETH-USD), Binance Coin (BNB-USD), Litecoin (LTC-USD)

**Data:** Five years of daily OHLCV data (January 2019 – January 2024) from Yahoo Finance, enriched with 16 engineered features — momentum, trend, volatility, and volume indicators — plus the Crypto Fear & Greed Index as a sentiment proxy.

---

## Repository Structure

```
crypto-price-prediction-hybrid-dl/
├── README.md
├── requirements.txt
├── src/
│   ├── seed_control.py             # Reproducibility: fixed seed + deterministic GPU ops
│   ├── crypto_dataset_pipeline.py  # Data download, feature engineering, walk-forward splits
│   ├── crypto_eda.py               # Exploratory data analysis (ADF test, correlations, outliers)
│   ├── crypto_models.py         # All 6 models: training, evaluation, Diebold-Mariano testing
│   └── crypto_charts.py         # Observed vs predicted charts (train/test split visualisation)
├── notebooks/
│   └── CryptoPricePrediction_MSc.ipynb   # Step-by-step Colab notebook running all 5 scripts in order
├── data/
│   └── processed/                  # Processed OHLCV + feature CSVs for all 4 coins
└── results/
    ├── all_results_summary.csv     # Final RMSE/MAE/MAPE for all 8 models × 4 coins
    ├── eda_*.csv / eda_*.png       # EDA outputs (descriptive stats, ADF test, correlations, outliers)
    ├── *_model_results.png         # Per-coin model comparison bar charts
    └── *_actual_vs_predicted.png   # Per-coin observed vs predicted charts
```

---

## How to Run

### Option A — Step-by-step notebook (recommended)

Open `notebooks/CryptoPricePrediction_MSc.ipynb` in Google Colab and run each cell in order. It walks through all 10 steps — install, upload, seed control, pipeline, verification, EDA, model training, charts, download, and results summary — with explanations at each stage.

### Option B — Manual, in a fresh Colab runtime

```bash
pip install -r requirements.txt
```

Then, in this exact order (each step depends on the files the previous one creates):

```python
exec(open("src/seed_control.py").read())            # 1. Set reproducible seed — run FIRST
exec(open("src/crypto_dataset_pipeline.py").read())  # 2. Download data, engineer features
exec(open("src/crypto_eda.py").read())                # 3. Exploratory data analysis
exec(open("src/crypto_models_v3.py").read())          # 4. Train + evaluate all 6 models
exec(open("src/crypto_charts_v2.py").read())           # 5. Generate observed vs predicted charts
```

Expect **60–100 minutes total** (model training is the slow step). A GPU runtime (`Runtime → Change runtime type → T4 GPU`) is strongly recommended.

---

## Methodology Highlights

- **Walk-forward expanding-window validation** — stricter than a static train/test split; each fold trains on all data up to a point and tests on the following 90 days, evaluating models across bull, bear, and recovery market conditions.
- **Leakage-free preprocessing** — the `MinMaxScaler` is fit exclusively on each fold's training data and only ever applied, never re-fitted, to that fold's test data.
- **Prediction target: next-day log return**, not raw price — justified statistically via an Augmented Dickey-Fuller test confirming log returns are stationary while raw prices are not (see `results/eda_adf_stationarity.csv`).
- **Two naive baselines** (Random Walk, Persistence) included as a minimum bar every hybrid model must beat.
- **Diebold-Mariano test** used throughout to confirm whether differences in forecast accuracy are statistically significant rather than due to chance.
- **Fixed random seed** (`seed_control.py`) across NumPy, Python's `random` module, and TensorFlow, with deterministic GPU operations enabled, so all reported results are exactly reproducible from the same code and data.

---

## Key Finding

Across all four cryptocurrencies, **no hybrid model achieved a lower RMSE than the simple Random Walk baseline** — a result consistent with weak-form market efficiency in cryptocurrency markets (Urquhart, 2016; Nadarajah & Chu, 2017), not a shortcoming of the models themselves. Transformer-LSTM was nonetheless confirmed as the strongest hybrid architecture, with statistically significant outperformance over the Persistence baseline (Diebold-Mariano test, p < 0.05 across all four coins). Full results, per-model breakdowns, and the statistical tests behind these claims are in `results/all_results_summary.csv` and the accompanying dissertation.

---

## Technologies

Python 3.10 · TensorFlow 2.x (Keras API) · scikit-learn · yfinance · statsmodels · pandas · NumPy · Matplotlib · Seaborn

All model training was conducted on Google Colab Pro using T4 GPU acceleration.

---

## Acknowledgements

With thanks to my supervisor, Dr Michael, for guidance throughout this project — including the amendments that shaped its walk-forward validation design, statistical baselines, and reproducibility controls.
