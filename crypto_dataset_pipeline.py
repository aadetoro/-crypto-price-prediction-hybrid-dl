"""
============================================================
 Cryptocurrency Price Prediction — Data Pipeline v2
 MSc Data Science | University of Roehampton London
 Student: Abiodun Adeyinka Adetoro | A00065756
============================================================
 UPDATED: Now processes FOUR cryptocurrencies as requested
 by Dr Michael (extending beyond Bitcoin and Ethereum):
   1. Bitcoin      (BTC-USD)
   2. Ethereum     (ETH-USD)
   3. Binance Coin (BNB-USD)  ← NEW
   4. Litecoin     (LTC-USD)  ← NEW

 All four use the identical pipeline:
   Amendment 1 — Walk-forward validation splits
   Amendment 2 — Explicit prediction target (1-day log return)
   Amendment 3 — Naïve baselines (Random Walk + Persistence)
   Amendment 4 — Data leakage controls
============================================================
"""

# ── 0. INSTALL (uncomment in Google Colab) ────────────────────────────────────
# !pip install yfinance ta pandas numpy scikit-learn matplotlib requests -q

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import warnings
from ta import momentum, trend, volatility
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────

# ── UPDATED: Four coins ───────────────────────────────────────────────────────
COINS = {
    "Bitcoin":     "BTC-USD",   # Largest crypto by market cap
    "Ethereum":    "ETH-USD",   # Second largest, smart contract platform
    "BinanceCoin": "BNB-USD",   # Third largest, exchange-driven dynamics
    "Litecoin":    "LTC-USD",   # One of the oldest altcoins
}

# Coin display colours for charts
COIN_COLORS = {
    "Bitcoin":     "#F7931A",   # Bitcoin orange
    "Ethereum":    "#627EEA",   # Ethereum purple-blue
    "BinanceCoin": "#F0B90B",   # Binance yellow
    "Litecoin":    "#BFBBBB",   # Litecoin silver
}

START        = "2019-01-01"
END          = "2024-01-01"
SEQ_LEN      = 60     # 60-day look-back window
HORIZON      = 1      # predict 1-day-ahead log return
TARGET       = "Log_Return"
TRAIN_WINDOW = 730    # 2-year initial training window
TEST_WINDOW  = 90     # re-test every 90 days (1 quarter)

FEATURE_COLS = [
    "Close", "Volume", "Log_Return", "Daily_Range",
    "RSI_14", "RSI_7", "MACD", "MACD_Sig", "MACD_Hist",
    "EMA_12", "EMA_26", "BB_Width", "ATR_14", "Vol_20",
    "Vol_Ratio", "FearGreed_Score"
]


# ── 2. DOWNLOAD OHLCV DATA ────────────────────────────────────────────────────

def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance. Returns clean DataFrame."""
    print(f"  Downloading {ticker} ...")
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index   = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Check we got enough data
    if len(df) < TRAIN_WINDOW + TEST_WINDOW + SEQ_LEN:
        print(f"    WARNING: Only {len(df)} rows — may be insufficient for walk-forward splits")
    else:
        print(f"    ✓ {df.shape[0]} rows | {df.index[0].date()} → {df.index[-1].date()}")
    return df

print("=" * 65)
print("STEP 1: Downloading OHLCV data for all 4 cryptocurrencies")
print("=" * 65)
raw_data = {}
for name, ticker in COINS.items():
    raw_data[name] = download_ohlcv(ticker, START, END)


# ── 3. FEAR & GREED INDEX (shared across all coins) ───────────────────────────

def download_fear_greed(limit: int = 2000) -> pd.DataFrame:
    """
    Download Crypto Fear & Greed Index from alternative.me (free, no key).
    This index is market-wide — same value applies to all coins.
    """
    print("\nSTEP 2: Downloading Crypto Fear & Greed Index ...")
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()["data"]
        df   = pd.DataFrame(data)
        df["Date"]            = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["FearGreed_Score"] = df["value"].astype(float)
        df["FearGreed_Class"] = df["value_classification"]
        df = df[["Date","FearGreed_Score","FearGreed_Class"]].set_index("Date")
        df.sort_index(inplace=True)
        print(f"  ✓ {df.shape[0]} daily records | "
              f"{df.index[0].date()} → {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"  WARNING: Could not download Fear & Greed ({e}). Using neutral (50).")
        return pd.DataFrame()

fear_greed = download_fear_greed()


# ── 4. FEATURE ENGINEERING ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, fg: pd.DataFrame,
                      coin_name: str) -> pd.DataFrame:
    """
    Engineer all 16 features from raw OHLCV data.
    No future data is used at any point (Amendment 4 leakage control).
    """
    d = df.copy()

    # ── Log return (prediction target) ────────────────────────────────────
    d["Log_Return"]  = np.log(d["Close"] / d["Close"].shift(1))
    d["Daily_Range"] = (d["High"] - d["Low"]) / d["Close"]
    d["Price_Change"]= d["Close"].pct_change()

    # ── Momentum ──────────────────────────────────────────────────────────
    d["RSI_14"] = momentum.RSIIndicator(d["Close"], window=14).rsi()
    d["RSI_7"]  = momentum.RSIIndicator(d["Close"], window=7).rsi()
    stoch       = momentum.StochasticOscillator(d["High"], d["Low"], d["Close"])
    d["Stoch_K"]= stoch.stoch()
    d["Stoch_D"]= stoch.stoch_signal()

    # ── Trend ─────────────────────────────────────────────────────────────
    macd          = trend.MACD(d["Close"])
    d["MACD"]     = macd.macd()
    d["MACD_Sig"] = macd.macd_signal()
    d["MACD_Hist"]= macd.macd_diff()
    d["EMA_12"]   = trend.EMAIndicator(d["Close"], window=12).ema_indicator()
    d["EMA_26"]   = trend.EMAIndicator(d["Close"], window=26).ema_indicator()
    d["SMA_50"]   = trend.SMAIndicator(d["Close"], window=50).sma_indicator()
    d["SMA_200"]  = trend.SMAIndicator(d["Close"], window=200).sma_indicator()

    # ── Volatility ────────────────────────────────────────────────────────
    bb            = volatility.BollingerBands(d["Close"], window=20)
    d["BB_Upper"] = bb.bollinger_hband()
    d["BB_Lower"] = bb.bollinger_lband()
    d["BB_Mid"]   = bb.bollinger_mavg()
    d["BB_Width"] = (d["BB_Upper"] - d["BB_Lower"]) / d["BB_Mid"]
    d["ATR_14"]   = volatility.AverageTrueRange(
        d["High"], d["Low"], d["Close"], window=14).average_true_range()
    d["Vol_20"]   = d["Log_Return"].rolling(20).std()

    # ── Volume ratio ──────────────────────────────────────────────────────
    d["Vol_Ratio"] = d["Volume"] / d["Volume"].rolling(20).mean()

    # ── Sentiment (shared Fear & Greed Index) ────────────────────────────
    if not fg.empty:
        d = d.join(fg[["FearGreed_Score"]], how="left")
        d["FearGreed_Score"] = d["FearGreed_Score"].fillna(method="ffill")
    else:
        d["FearGreed_Score"] = 50.0  # neutral fallback

    # ── Target: next-day log return ───────────────────────────────────────
    d["Target"] = d["Log_Return"].shift(-HORIZON)

    # Drop NaN rows from rolling windows
    d.dropna(inplace=True)
    return d

print("\n" + "=" * 65)
print("STEP 3: Engineering features for all 4 coins")
print("=" * 65)
processed = {}
for name, df in raw_data.items():
    processed[name] = engineer_features(df, fear_greed, name)
    p = processed[name]
    print(f"  {name:<14} {p.shape[0]} rows | "
          f"{p.index[0].date()} → {p.index[-1].date()} | "
          f"{p.shape[1]} features")


# ── 5. NAÏVE BASELINES ────────────────────────────────────────────────────────

def naive_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Random Walk and Persistence baselines (Amendment 3)."""
    base = pd.DataFrame(index=df.index)
    base["Actual"]      = df["Target"]
    base["RandomWalk"]  = 0.0
    base["Persistence"] = df["Log_Return"]
    return base.dropna()

print("\n" + "=" * 65)
print("STEP 4: Computing naïve baselines")
print("=" * 65)
baselines = {}
for name, df in processed.items():
    baselines[name] = naive_baselines(df)
    b = baselines[name]
    rw = np.sqrt(((b["Actual"] - b["RandomWalk"])**2).mean())
    pe = np.sqrt(((b["Actual"] - b["Persistence"])**2).mean())
    print(f"  {name:<14} Random Walk RMSE: {rw:.6f} | Persistence RMSE: {pe:.6f}")


# ── 6. WALK-FORWARD SPLITS ────────────────────────────────────────────────────

def walk_forward_splits(df, train_days=TRAIN_WINDOW,
                        test_days=TEST_WINDOW) -> list:
    """Generate expanding-window walk-forward train/test splits."""
    splits, n, start = [], len(df), train_days
    while start + test_days <= n:
        splits.append((list(range(0, start)),
                       list(range(start, min(start + test_days, n)))))
        start += test_days
    return splits

print("\n" + "=" * 65)
print("STEP 5: Generating walk-forward splits")
print("=" * 65)
splits_info = {}
for name, df in processed.items():
    splits = walk_forward_splits(df)
    splits_info[name] = splits
    print(f"  {name:<14} {len(splits)} folds")
    for i, (tr, te) in enumerate(splits[:2]):
        print(f"    Fold {i+1}: Train → {df.index[tr[-1]].date()} | "
              f"Test {df.index[te[0]].date()} → {df.index[te[-1]].date()}")
    if len(splits) > 2:
        print(f"    ... and {len(splits)-2} more folds")


# ── 7. LEAKAGE-FREE SEQUENCE BUILDER ─────────────────────────────────────────

def build_sequences_for_fold(df, train_idx, test_idx, seq_len=SEQ_LEN):
    """
    Build (X_train, y_train, X_test, y_test) sequences.
    LEAKAGE CONTROL: Scaler fitted on train only, applied to test.
    """
    tr, te = df.iloc[train_idx], df.iloc[test_idx]

    scaler       = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(tr[FEATURE_COLS])
    test_scaled  = scaler.transform(te[FEATURE_COLS])

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    train_targets = target_scaler.fit_transform(tr[["Target"]])
    test_targets  = target_scaler.transform(te[["Target"]])

    def make_sequences(X_arr, y_arr, sl):
        Xs, ys = [], []
        for i in range(sl, len(X_arr)):
            Xs.append(X_arr[i-sl:i])
            ys.append(y_arr[i])
        return np.array(Xs), np.array(ys)

    X_tr, y_tr = make_sequences(train_scaled, train_targets, seq_len)
    X_te, y_te = make_sequences(test_scaled,  test_targets,  seq_len)
    return X_tr, y_tr, X_te, y_te, scaler, target_scaler

# Test with first coin first fold
first_coin = list(processed.keys())[0]
s0_tr, s0_te = splits_info[first_coin][0]
X_tr, y_tr, X_te, y_te, _, _ = build_sequences_for_fold(
    processed[first_coin], s0_tr, s0_te)
print(f"\n  Sequence shape check ({first_coin} Fold 1):")
print(f"  X_train: {X_tr.shape} | y_train: {y_tr.shape}")
print(f"  X_test:  {X_te.shape} | y_test:  {y_te.shape}")


# ── 8. SAVE ALL PROCESSED DATA ────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 6: Saving processed CSV files for all 4 coins")
print("=" * 65)
saved_files = []
for name, df in processed.items():
    fname = f"{name.lower()}_processed.csv"
    df.to_csv(fname)
    saved_files.append(fname)
    print(f"  ✓ Saved: {fname} ({df.shape[0]} rows × {df.shape[1]} cols)")

for name, b in baselines.items():
    fname = f"{name.lower()}_naive_baselines.csv"
    b.to_csv(fname)
    saved_files.append(fname)
    print(f"  ✓ Saved: {fname}")


# ── 9. VISUALISATION — ALL 4 COINS ────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 7: Generating overview charts for all 4 coins")
print("=" * 65)

fig, axes = plt.subplots(4, 3, figsize=(18, 16))
fig.suptitle(
    "All 4 Cryptocurrencies — Processed Daily Dataset Overview\n"
    "Abiodun Adeyinka Adetoro | A00065756 | MSc Data Science",
    fontsize=13, fontweight="bold"
)

for row, (name, df) in enumerate(processed.items()):
    color = COIN_COLORS[name]
    ticker = COINS[name]

    # Closing price
    axes[row, 0].plot(df["Close"], color=color, lw=1.2)
    axes[row, 0].set_title(f"{name} ({ticker}) — Closing Price", fontsize=10)
    axes[row, 0].set_ylabel("USD")
    axes[row, 0].grid(True, alpha=0.3)

    # Log return (prediction target)
    axes[row, 1].plot(df["Log_Return"], color=color, lw=0.7, alpha=0.8)
    axes[row, 1].axhline(0, color="black", lw=0.5, linestyle="--")
    axes[row, 1].set_title(f"{name} — Log Returns (Target)", fontsize=10)
    axes[row, 1].set_ylabel("Log Return")
    axes[row, 1].grid(True, alpha=0.3)

    # RSI
    axes[row, 2].plot(df["RSI_14"], color=color, lw=1.0)
    axes[row, 2].axhline(70, color="red",   lw=0.8, linestyle="--",
                          label="Overbought (70)")
    axes[row, 2].axhline(30, color="green", lw=0.8, linestyle="--",
                          label="Oversold (30)")
    axes[row, 2].set_title(f"{name} — RSI (14-day)", fontsize=10)
    axes[row, 2].set_ylabel("RSI")
    axes[row, 2].legend(fontsize=7)
    axes[row, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("all_coins_dataset_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: all_coins_dataset_overview.png")


# ── 10. CLOSING PRICE COMPARISON — ALL 4 COINS ────────────────────────────────

fig2, ax = plt.subplots(figsize=(16, 6))
for name, df in processed.items():
    # Normalise to 100 at start for fair comparison
    norm = (df["Close"] / df["Close"].iloc[0]) * 100
    ax.plot(norm, label=f"{name} ({COINS[name]})",
            color=COIN_COLORS[name], lw=1.5)

ax.set_title("All 4 Coins — Normalised Price Performance (Base=100 at Jan 2019)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Normalised Price (Base 100)")
ax.set_xlabel("Date")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(100, color="grey", lw=0.8, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("all_coins_price_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: all_coins_price_comparison.png")


# ── 11. FINAL SUMMARY ─────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("PIPELINE COMPLETE — SUMMARY")
print("=" * 65)
print(f"\n  Coins processed : {', '.join(COINS.keys())}")
print(f"  Date range      : {START} to {END}")
print(f"  Features        : {len(FEATURE_COLS)} per timestep")
print(f"  Sequence length : {SEQ_LEN} days look-back")
print(f"  Prediction target: {HORIZON}-day log return (Amendment 2)")
print(f"  Validation      : Walk-forward expanding window (Amendment 1)")
print(f"\n  Walk-forward folds per coin:")
for name in COINS:
    print(f"    {name:<14} {len(splits_info[name])} folds")

print(f"\n  Files saved:")
for f in saved_files:
    print(f"    {f}")
print("  all_coins_dataset_overview.png")
print("  all_coins_price_comparison.png")

print(f"""
  Next step — run the model script:
    exec(open("crypto_models_v2.py").read())

  The model script will automatically detect all 4 CSV files
  and train + evaluate all 6 models on each coin.
""")
print("=" * 65)
