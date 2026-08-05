# ============================================================================
# crypto_eda.py
# MSc Data Science Dissertation — Cryptocurrency Price Prediction
# Exploratory Data Analysis: run this AFTER crypto_dataset_pipeline_v2.py
# and BEFORE crypto_models_v3.py.
#
# Produces:
#   eda_descriptive_stats.csv        — mean/std/min/max/skew/kurtosis per coin
#   eda_missing_values.csv           — missing-value check per coin
#   eda_adf_stationarity.csv         — ADF test: raw Close vs Log_Return
#   eda_correlation_heatmaps.png     — 16-feature correlation matrix, 4 coins
#   eda_target_distribution.png      — histogram of Target (next-day log return)
#   eda_outliers.csv                 — days where |Log_Return| > 3 std devs
#
# FIX (this version): both chart functions now call plt.show() before
# plt.close(), so the correlation heatmap and target distribution charts
# actually display inline in Colab, instead of only being saved silently
# to disk. Previously plt.close() ran immediately after plt.savefig() with
# no plt.show() in between, so the figures never rendered in the notebook
# even though the PNG files were created correctly.
# ============================================================================

# ── SEED CONTROL — must run before any other imports ──────────────────────
import os
os.environ["PYTHONHASHSEED"] = "42"
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
print(f"[SEED] Global random seed set to {SEED}")
# ── END SEED CONTROL ────────────────────────────────────────────────────────

import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import adfuller

FEATURE_COLS = [
    "Close", "Volume", "Log_Return", "Daily_Range",
    "RSI_14", "RSI_7", "MACD", "MACD_Sig", "MACD_Hist",
    "EMA_12", "EMA_26", "BB_Width", "ATR_14", "Vol_20",
    "Vol_Ratio", "FearGreed_Score"
]

# ── Load all processed coin datasets ───────────────────────────────────────
def load_datasets():
    files = sorted(glob.glob("*_processed.csv"))
    if not files:
        raise FileNotFoundError("No *_processed.csv files found — run crypto_dataset_pipeline_v2.py first.")
    datasets = {}
    for f in files:
        coin_name = os.path.basename(f).replace("_processed.csv", "").capitalize()
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        datasets[coin_name] = df
        print(f"  Loaded {coin_name}: {df.shape[0]} rows, {df.shape[1]} columns")
    return datasets

# ── 1. Descriptive statistics ───────────────────────────────────────────────
def descriptive_stats(datasets):
    rows = []
    for coin, df in datasets.items():
        for col in ["Close", "Log_Return", "Target"]:
            s = df[col]
            rows.append({
                "Coin": coin, "Variable": col,
                "Mean": round(s.mean(), 6), "Std": round(s.std(), 6),
                "Min": round(s.min(), 6), "Max": round(s.max(), 6),
                "Skewness": round(scipy_stats.skew(s.dropna()), 4),
                "Kurtosis": round(scipy_stats.kurtosis(s.dropna()), 4),
            })
    out = pd.DataFrame(rows)
    out.to_csv("eda_descriptive_stats.csv", index=False)
    print("\n✓ Saved eda_descriptive_stats.csv")
    print(out.to_string(index=False))
    return out

# ── 2. Missing value check ──────────────────────────────────────────────────
def missing_value_check(datasets):
    rows = []
    for coin, df in datasets.items():
        total_missing = int(df[FEATURE_COLS + ["Target"]].isnull().sum().sum())
        rows.append({"Coin": coin, "Rows": len(df), "Total_Missing_Values": total_missing})
    out = pd.DataFrame(rows)
    out.to_csv("eda_missing_values.csv", index=False)
    print("\n✓ Saved eda_missing_values.csv")
    print(out.to_string(index=False))
    return out

# ── 3. ADF stationarity test: raw Close vs Log_Return ──────────────────────
def stationarity_test(datasets):
    rows = []
    for coin, df in datasets.items():
        for label, series in [("Close (raw price)", df["Close"]), ("Log_Return", df["Log_Return"])]:
            result = adfuller(series.dropna(), autolag="AIC")
            stat, p_value = result[0], result[1]
            rows.append({
                "Coin": coin, "Series": label,
                "ADF_Statistic": round(stat, 4), "p_value": round(p_value, 6),
                "Stationary_at_5pct": "Yes" if p_value < 0.05 else "No",
            })
    out = pd.DataFrame(rows)
    out.to_csv("eda_adf_stationarity.csv", index=False)
    print("\n✓ Saved eda_adf_stationarity.csv")
    print(out.to_string(index=False))
    return out

# ── 4. Correlation heatmaps (one subplot per coin) ──────────────────────────
def correlation_heatmaps(datasets):
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()
    for i, (coin, df) in enumerate(datasets.items()):
        corr = df[FEATURE_COLS].corr()
        sns.heatmap(corr, ax=axes[i], cmap="coolwarm", center=0,
                    annot=False, cbar=True, vmin=-1, vmax=1)
        axes[i].set_title(f"{coin} — Feature Correlation Matrix", fontweight="bold")
        axes[i].tick_params(axis="x", rotation=90, labelsize=7)
        axes[i].tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    plt.savefig("eda_correlation_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.show()   # FIX: display inline in Colab before closing
    plt.close()
    print("\n✓ Saved eda_correlation_heatmaps.png")

# ── 5. Target distribution (next-day log return) ────────────────────────────
def target_distribution(datasets):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, (coin, df) in enumerate(datasets.items()):
        target = df["Target"].dropna()
        axes[i].hist(target, bins=80, color="steelblue", alpha=0.8, density=True)
        mu, sigma = target.mean(), target.std()
        x = np.linspace(target.min(), target.max(), 200)
        axes[i].plot(x, scipy_stats.norm.pdf(x, mu, sigma), color="red", lw=1.5,
                     label=f"Normal fit (μ={mu:.4f}, σ={sigma:.4f})")
        skew = scipy_stats.skew(target)
        kurt = scipy_stats.kurtosis(target)
        axes[i].set_title(f"{coin} — Target Distribution\nSkew={skew:.3f}, Excess Kurtosis={kurt:.3f}",
                           fontweight="bold")
        axes[i].set_xlabel("Next-day Log Return")
        axes[i].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("eda_target_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()   # FIX: display inline in Colab before closing
    plt.close()
    print("\n✓ Saved eda_target_distribution.png")

# ── 6. Outlier check: days where |Log_Return| > 3 standard deviations ──────
def outlier_check(datasets):
    rows = []
    for coin, df in datasets.items():
        mu, sigma = df["Log_Return"].mean(), df["Log_Return"].std()
        threshold = 3 * sigma
        outliers = df[np.abs(df["Log_Return"] - mu) > threshold]
        for date, row in outliers.iterrows():
            rows.append({
                "Coin": coin, "Date": date.date(),
                "Log_Return": round(row["Log_Return"], 4),
                "Close": round(row["Close"], 2),
            })
    out = pd.DataFrame(rows).sort_values(["Coin", "Date"]) if rows else pd.DataFrame(
        columns=["Coin", "Date", "Log_Return", "Close"])
    out.to_csv("eda_outliers.csv", index=False)
    print(f"\n✓ Saved eda_outliers.csv — {len(out)} extreme-return days flagged (>3σ)")
    if len(out) > 0:
        print(out.head(15).to_string(index=False))
    return out

# ── Main ─────────────────────────────────────────────────────────────────────
def run_eda():
    print("=" * 70)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    datasets = load_datasets()

    descriptive_stats(datasets)
    missing_value_check(datasets)
    adf_results = stationarity_test(datasets)
    correlation_heatmaps(datasets)
    target_distribution(datasets)
    outlier_check(datasets)

    print("\n" + "=" * 70)
    print("  EDA SUMMARY")
    print("=" * 70)
    non_stationary_close = adf_results[(adf_results["Series"] == "Close (raw price)") &
                                        (adf_results["Stationary_at_5pct"] == "No")]
    stationary_returns = adf_results[(adf_results["Series"] == "Log_Return") &
                                      (adf_results["Stationary_at_5pct"] == "Yes")]
    print(f"  Raw Close price non-stationary for {len(non_stationary_close)}/4 coins (expected).")
    print(f"  Log_Return series stationary for {len(stationary_returns)}/4 coins (expected).")
    print("  This confirms log returns are the statistically appropriate model target.")
    print("\nEDA COMPLETE")

if __name__ == "__main__":
    run_eda()
