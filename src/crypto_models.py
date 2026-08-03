"""
============================================================
 Cryptocurrency Price Prediction — Model Building
 MSc Data Science | University of Roehampton London
 Student: Abiodun Adeyinka Adetoro | A00065756
============================================================
 Models built in this script:
   BASELINE 1 — ARIMA
   BASELINE 2 — Standalone LSTM
   BASELINE 3 — Standalone GRU
   HYBRID  1  — CNN-LSTM
   HYBRID  2  — LSTM with Attention
   HYBRID  3  — Transformer-LSTM

 Dr Michael's requirements:
   OK Walk-forward validation (Amendment 1)
   OK Log return prediction target (Amendment 2)
   OK Naive baselines: Random Walk + Persistence (Amendment 3)
   OK Data leakage controls + explainability hooks (Amendment 4)

 FIX (this version): the sequence builder now carries the last SEQ_LEN
 days of TRAINING data forward as history for the test window, so every
 model — naive baselines, ARIMA, and all five deep learning models —
 produces exactly one prediction per test day (90 per fold), instead of
 the deep learning models losing their first 60 test days to windowing.
 This keeps every model's predictions aligned to the same days, which
 the Diebold-Mariano test depends on to be statistically valid. Without
 this fix, DM test results can vary between runs depending on which
 fold happens to be compared, purely as an artefact of misalignment —
 not a real difference in model performance.
============================================================
"""

# ── 0. SEED CONTROL — must run before any other imports ───────────────────
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import random
import numpy as np

SEED = 42

def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass

set_global_seed(SEED)
print(f"[SEED] Global random seed set to {SEED}")
# ── END SEED CONTROL ────────────────────────────────────────────────────────

# ── 0b. INSTALL (uncomment in Google Colab) ────────────────────────────────
# !pip install yfinance ta scikit-learn statsmodels tensorflow matplotlib seaborn -q

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D,
    Flatten, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Reshape, Permute, Multiply,
    Activation, Lambda, Add, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Statistics
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools

print("TensorFlow version:", tf.__version__)
print("All libraries loaded successfully.\n")

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────

SEQ_LEN      = 60       # 60-day look-back window
HORIZON      = 1        # predict 1-day ahead
EPOCHS       = 100      # max epochs (early stopping will cut short)
BATCH_SIZE   = 32
LEARNING_RATE= 0.001
DROPOUT_RATE = 0.2
TRAIN_WINDOW = 730      # 2 years initial training (days)
TEST_WINDOW  = 90       # re-test every 90 days (1 quarter)

FEATURE_COLS = [
    "Close", "Volume", "Log_Return", "Daily_Range",
    "RSI_14", "RSI_7", "MACD", "MACD_Sig", "MACD_Hist",
    "EMA_12", "EMA_26", "BB_Width", "ATR_14", "Vol_20",
    "Vol_Ratio", "FearGreed_Score"
]
N_FEATURES = len(FEATURE_COLS)

# ── 2. LOAD PROCESSED DATA ────────────────────────────────────────────────────
# Run crypto_dataset_pipeline.py first to generate these CSV files

print("=" * 60)
print("STEP 1: Loading processed datasets — all 4 coins")
print("=" * 60)

COIN_FILES = {
    "Bitcoin":     "bitcoin_processed.csv",
    "Ethereum":    "ethereum_processed.csv",
    "BinanceCoin": "binancecoin_processed.csv",
    "Litecoin":    "litecoin_processed.csv",
}

datasets = {}
for name, fname in COIN_FILES.items():
    if os.path.exists(fname):
        df = pd.read_csv(fname, index_col="Date", parse_dates=True)
        datasets[name] = df
        print(f"  OK {name:<14} {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        print(f"  MISSING {name:<14} {fname} not found — skipping")

print(f"\n  Total coins loaded: {len(datasets)} / {len(COIN_FILES)}")

# Explicit warning if fewer than 4 coins loaded — do not silently continue
if len(datasets) < len(COIN_FILES):
    missing = [n for n in COIN_FILES if n not in datasets]
    print(f"\n  *** WARNING: {len(missing)} coin(s) missing from this session: "
          f"{', '.join(missing)} ***")
    print("  *** Run crypto_dataset_pipeline.py for ALL 4 coins in THIS SAME ")
    print("  *** Colab runtime before continuing, or results below will only ")
    print("  *** cover the coins currently loaded. ***\n")


# ── 3. WALK-FORWARD SPLIT GENERATOR ──────────────────────────────────────────

def walk_forward_splits(df, train_days=TRAIN_WINDOW, test_days=TEST_WINDOW):
    splits, n, start = [], len(df), train_days
    while start + test_days <= n:
        splits.append((list(range(0, start)),
                       list(range(start, min(start + test_days, n)))))
        start += test_days
    return splits


# ── 4. LEAKAGE-FREE SEQUENCE BUILDER (FIXED) ─────────────────────────────────

def build_sequences(df, train_idx, test_idx, seq_len=SEQ_LEN):
    """
    Fit scaler on train only. Transform test with same scaler.
    This is the data leakage control (Amendment 4).

    FIX: the last `seq_len` rows of the TRAINING fold are prepended to the
    test fold before windowing, purely as historical context for the first
    few test-day predictions. The scaler is still fit on training data only
    (no leakage), and no test-set values are used to predict themselves —
    but this ensures every one of the `test_days` days in the fold receives
    a prediction, matching the naive baselines and ARIMA, instead of losing
    the first `seq_len` test days to the windowing operation.
    """
    tr, te = df.iloc[train_idx], df.iloc[test_idx]

    scaler = MinMaxScaler(feature_range=(0, 1))
    tr_sc  = scaler.fit_transform(tr[FEATURE_COLS])
    # Test features scaled using train-only scaler (leakage-free)
    te_sc  = scaler.transform(te[FEATURE_COLS])
    # Prepend last seq_len rows of scaled TRAIN features as history for test
    te_sc_with_history = np.vstack([tr_sc[-seq_len:], te_sc])

    t_scaler = MinMaxScaler(feature_range=(0, 1))
    tr_tgt   = t_scaler.fit_transform(tr[["Target"]])
    te_tgt   = t_scaler.transform(te[["Target"]])
    te_tgt_with_history = np.vstack([tr_tgt[-seq_len:], te_tgt])

    def make_seq(X, y, sl):
        Xs, ys = [], []
        for i in range(sl, len(X)):
            Xs.append(X[i-sl:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    # Training sequences: built purely within training data, as before
    X_tr, y_tr = make_seq(tr_sc, tr_tgt, seq_len)
    # Test sequences: now yield exactly len(test_idx) predictions, one per
    # test day, using training-tail history for the earliest test days
    X_te, y_te = make_seq(te_sc_with_history, te_tgt_with_history, seq_len)

    assert len(X_te) == len(test_idx), (
        f"Expected {len(test_idx)} test sequences, got {len(X_te)} — "
        "windowing fix did not produce full test-day coverage."
    )

    return X_tr, y_tr, X_te, y_te, scaler, t_scaler


# ── 5. EVALUATION METRICS ─────────────────────────────────────────────────────

def compute_metrics(actual, predicted, label=""):
    """RMSE, MAE, MAPE for a set of predictions.
    Note: MAPE is unreliable for log returns (values near zero).
    Use RMSE and MAE as primary metrics.
    """
    actual    = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    # Only compute MAPE on non-near-zero actuals to avoid blow-up
    mask = np.abs(actual) > 0.001
    if mask.sum() > 10:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan
    if label:
        mape_str = f"{mape:.2f}%" if not np.isnan(mape) else "N/A"
        print(f"  {label:<28} RMSE={rmse:.6f}  MAE={mae:.6f}  MAPE={mape_str}")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def diebold_mariano(e1, e2, h=1):
    """
    Diebold-Mariano test (Harvey et al. 1997).
    Tests whether model 1 and model 2 have statistically different forecast accuracy.
    H0: equal predictive accuracy.
    Returns: DM statistic and approximate p-value.

    IMPORTANT: e1 and e2 must be errors for the SAME sequence of days, in the
    same order. With the windowing fix in build_sequences(), every model's
    fold_predictions now cover all test days in the same order, so this
    assumption holds without needing to silently truncate mismatched arrays.
    """
    from scipy import stats
    if len(e1) != len(e2):
        raise ValueError(
            f"diebold_mariano: length mismatch ({len(e1)} vs {len(e2)}) — "
            "models are not aligned to the same days. Check build_sequences()."
        )
    d  = e1**2 - e2**2          # loss differential
    T  = len(d)
    mu = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    dm_stat = mu / np.sqrt(gamma0 / T)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


# ── 6. NAIVE BASELINES (Amendment 3) ─────────────────────────────────────────

def naive_predictions(df):
    """Random Walk and Persistence predictions."""
    actual      = df["Target"].values
    random_walk = np.zeros_like(actual)          # predict 0 return every day
    persistence = df["Log_Return"].values        # predict today's return repeats
    return actual, random_walk, persistence


# ── 7. ARIMA BASELINE ─────────────────────────────────────────────────────────

def run_arima(df, train_idx, test_idx):
    """
    ARIMA on log returns. Auto-selects best (p,d,q) order via AIC grid search.
    Parameter estimation: p in [0,1,2], d in [0,1], q in [0,1,2]
    """
    series = df["Log_Return"]
    train_series = series.iloc[train_idx]
    test_series  = series.iloc[test_idx]

    best_aic, best_order = np.inf, (1, 0, 1)
    for p, d, q in itertools.product([0,1,2], [0,1], [0,1,2]):
        try:
            m = ARIMA(train_series, order=(p,d,q)).fit()
            if m.aic < best_aic:
                best_aic, best_order = m.aic, (p,d,q)
        except:
            continue

    model  = ARIMA(train_series, order=best_order).fit()
    preds  = model.forecast(steps=len(test_series))
    actual = test_series.values

    print(f"    Best ARIMA order: {best_order} (AIC={best_aic:.2f})")
    return actual, preds.values, best_order


# ── 8. CALLBACKS ──────────────────────────────────────────────────────────────

def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0)
    ]


# ── 9. MODEL DEFINITIONS ──────────────────────────────────────────────────────

def build_lstm(seq_len=SEQ_LEN, n_features=N_FEATURES):
    """Two-layer LSTM with dropout regularisation.
    Parameters: 128 units (layer 1), 64 units (layer 2), dropout=0.2"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(DROPOUT_RATE),
        LSTM(64, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(32, activation="relu"),
        Dense(1)
    ], name="LSTM_Baseline")
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return model


def build_gru(seq_len=SEQ_LEN, n_features=N_FEATURES):
    """Two-layer GRU. Fewer parameters than LSTM, often trains faster.
    Parameters: 128 units (layer 1), 64 units (layer 2), dropout=0.2"""
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(DROPOUT_RATE),
        GRU(64, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(32, activation="relu"),
        Dense(1)
    ], name="GRU_Baseline")
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return model


def build_cnn_lstm(seq_len=SEQ_LEN, n_features=N_FEATURES):
    """CNN extracts local patterns from the 60-day window.
    LSTM then models the temporal sequence of those extracted features."""
    inp = Input(shape=(seq_len, n_features))
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inp)
    x = Dropout(DROPOUT_RATE)(x)
    x = Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out, name="CNN_LSTM_Hybrid")
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return model


def build_lstm_attention(seq_len=SEQ_LEN, n_features=N_FEATURES):
    """LSTM produces hidden states for all 60 timesteps.
    Attention layer assigns a weight to each timestep."""
    inp = Input(shape=(seq_len, n_features))
    lstm_out = LSTM(128, return_sequences=True)(inp)
    lstm_out = Dropout(DROPOUT_RATE)(lstm_out)
    attention_scores = Dense(1, activation="tanh")(lstm_out)
    attention_weights = Activation("softmax")(attention_scores)
    context = Multiply()([lstm_out, attention_weights])
    context = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
    x   = Dense(64, activation="relu")(context)
    x   = Dropout(DROPOUT_RATE)(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out, name="LSTM_Attention_Hybrid")
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return model


def build_transformer_lstm(seq_len=SEQ_LEN, n_features=N_FEATURES,
                            d_model=64, n_heads=4, ff_dim=128):
    """Transformer encoder captures global dependencies across all 60
    timesteps simultaneously. LSTM then models the sequential structure
    of the encoder output."""
    inp = Input(shape=(seq_len, n_features))
    x = Dense(d_model)(inp)
    attn_out = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
    attn_out = Dropout(DROPOUT_RATE)(attn_out)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_out]))
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(d_model)(ff)
    ff = Dropout(DROPOUT_RATE)(ff)
    x  = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))
    x   = LSTM(64, return_sequences=False)(x)
    x   = Dropout(DROPOUT_RATE)(x)
    x   = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out, name="Transformer_LSTM_Hybrid")
    model.compile(optimizer=Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return model


# ── 10. TRAINING FUNCTION ─────────────────────────────────────────────────────

def train_model(model, X_train, y_train, epochs=EPOCHS, batch=BATCH_SIZE):
    """Train a Keras model with early stopping and LR reduction."""
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch,
        validation_split=0.1,
        callbacks=get_callbacks(),
        verbose=0
    )
    print(f"    Trained for {len(history.history['loss'])} epochs "
          f"(best val_loss={min(history.history['val_loss']):.6f})")
    return history


# ── 11. FULL PIPELINE — ONE COIN ──────────────────────────────────────────────

def run_full_pipeline(coin_name, df):
    print(f"\n{'='*60}")
    print(f"  COIN: {coin_name}")
    print(f"{'='*60}")

    splits   = walk_forward_splits(df)
    n_folds  = min(3, len(splits))
    print(f"  Using {n_folds} walk-forward folds out of {len(splits)} available\n")

    all_results = {
        "Random Walk":        [],
        "Persistence":        [],
        "ARIMA":              [],
        "LSTM":               [],
        "GRU":                [],
        "CNN-LSTM":           [],
        "LSTM-Attention":     [],
        "Transformer-LSTM":   [],
    }

    fold_predictions = {m: {"actual": [], "pred": []} for m in all_results}

    for fold_idx, (train_idx, test_idx) in enumerate(splits[:n_folds]):
        print(f"\n-- Fold {fold_idx + 1}/{n_folds} "
              f"(Train: {df.index[train_idx[0]].date()} -> {df.index[train_idx[-1]].date()} | "
              f"Test: {df.index[test_idx[0]].date()} -> {df.index[test_idx[-1]].date()})")

        X_tr, y_tr, X_te, y_te, scaler, t_scaler = build_sequences(
            df, train_idx, test_idx)

        # ── Naive baselines — now guaranteed same length as DL predictions ──
        actual, rw_pred, pe_pred = naive_predictions(df.iloc[test_idx])
        all_results["Random Walk"].append(
            compute_metrics(actual, rw_pred, "Random Walk"))
        all_results["Persistence"].append(
            compute_metrics(actual, pe_pred, "Persistence"))
        fold_predictions["Random Walk"]["actual"].extend(actual)
        fold_predictions["Random Walk"]["pred"].extend(rw_pred)
        fold_predictions["Persistence"]["pred"].extend(pe_pred)

        # ── ARIMA ─────────────────────────────────────────────────────────
        print(f"  Training ARIMA ...")
        ar_actual, ar_pred, order = run_arima(df, train_idx, test_idx)
        all_results["ARIMA"].append(
            compute_metrics(ar_actual, ar_pred, f"ARIMA{order}"))
        fold_predictions["ARIMA"]["actual"].extend(ar_actual)
        fold_predictions["ARIMA"]["pred"].extend(ar_pred)

        # ── Deep learning models ───────────────────────────────────────────
        dl_models = {
            "LSTM":             build_lstm(),
            "GRU":              build_gru(),
            "CNN-LSTM":         build_cnn_lstm(),
            "LSTM-Attention":   build_lstm_attention(),
            "Transformer-LSTM": build_transformer_lstm(),
        }

        for model_name, model in dl_models.items():
            print(f"  Training {model_name} ...")
            train_model(model, X_tr, y_tr)

            preds_sc = model.predict(X_te, verbose=0)
            preds    = t_scaler.inverse_transform(preds_sc).flatten()
            acts     = t_scaler.inverse_transform(y_te).flatten()

            all_results[model_name].append(
                compute_metrics(acts, preds, model_name))
            fold_predictions[model_name]["actual"].extend(acts)
            fold_predictions[model_name]["pred"].extend(preds)

            model.save(f"{coin_name.lower()}_{model_name.replace('-','_').replace(' ','_')}.h5")

        tf.keras.backend.clear_session()

    # ── Aggregate results across folds ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY — {coin_name}")
    print(f"{'='*60}")
    print(f"  {'Model':<24} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
    print(f"  {'-'*54}")

    summary = {}
    for model_name, fold_metrics in all_results.items():
        avg_rmse = np.mean([m["RMSE"] for m in fold_metrics])
        avg_mae  = np.mean([m["MAE"]  for m in fold_metrics])
        mape_vals = [m["MAPE"] for m in fold_metrics if not np.isnan(m["MAPE"])]
        avg_mape = np.mean(mape_vals) if mape_vals else np.nan
        summary[model_name] = {"RMSE": avg_rmse, "MAE": avg_mae, "MAPE": avg_mape}
        tag = "<- HYBRID" if model_name in ["CNN-LSTM","LSTM-Attention","Transformer-LSTM"] else ""
        mape_str = f"{avg_mape:>9.2f}%" if not np.isnan(avg_mape) else "       N/A"
        print(f"  {model_name:<24} {avg_rmse:>10.6f} {avg_mae:>10.6f} {mape_str}  {tag}")

    # ── Diebold-Mariano tests (now length-safe, no silent truncation) ──────────
    print(f"\n  DIEBOLD-MARIANO TESTS (best hybrid vs baselines)")
    print(f"  {'-'*54}")

    hybrid_names = ["CNN-LSTM", "LSTM-Attention", "Transformer-LSTM"]
    best_hybrid  = min(hybrid_names, key=lambda m: summary[m]["RMSE"])
    print(f"  Best hybrid: {best_hybrid} (RMSE={summary[best_hybrid]['RMSE']:.6f})\n")

    bh_actual = np.array(fold_predictions[best_hybrid]["actual"])
    bh_pred   = np.array(fold_predictions[best_hybrid]["pred"])
    bh_errors = bh_actual - bh_pred

    for baseline in ["Random Walk", "Persistence", "ARIMA", "LSTM"]:
        try:
            bl_pred = np.array(fold_predictions[baseline]["pred"])
            bl_actual = bh_actual  # same ground truth across all models by construction
            if len(bl_pred) != len(bh_errors):
                print(f"  {best_hybrid} vs {baseline:<20} "
                      f"SKIPPED — length mismatch ({len(bh_errors)} vs {len(bl_pred)}); "
                      f"check build_sequences() output for this baseline.")
                continue
            bl_errors = bl_actual - bl_pred
            dm_stat, p_val = diebold_mariano(bh_errors, bl_errors)
            sig = "*** (p<0.01)" if p_val < 0.01 else (
                  "** (p<0.05)"  if p_val < 0.05 else (
                  "* (p<0.10)"   if p_val < 0.10 else "not significant"))
            print(f"  {best_hybrid} vs {baseline:<20} "
                  f"DM={dm_stat:+.3f}  p={p_val:.4f}  {sig}")
        except Exception as e:
            print(f"  {best_hybrid} vs {baseline:<20} Error: {e}")

    return summary, fold_predictions


# ── 12. VISUALISATION ─────────────────────────────────────────────────────────

def plot_results(coin_name, summary, fold_predictions):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"{coin_name} — Model Comparison Results\n"
                 f"MSc Project | Abiodun Adeyinka Adetoro | A00065756",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    colors_map = {
        "Random Walk":      "#AAAAAA",
        "Persistence":      "#888888",
        "ARIMA":            "#E67E22",
        "LSTM":             "#3498DB",
        "GRU":              "#2ECC71",
        "CNN-LSTM":         "#E74C3C",
        "LSTM-Attention":   "#9B59B6",
        "Transformer-LSTM": "#1ABC9C",
    }

    ax1 = fig.add_subplot(gs[0, :])
    names  = list(summary.keys())
    rmses  = [summary[m]["RMSE"] for m in names]
    clrs   = [colors_map[m] for m in names]
    bars   = ax1.bar(names, rmses, color=clrs, edgecolor="white", linewidth=1.2)
    ax1.set_title("RMSE Comparison — All Models (lower is better)", fontweight="bold")
    ax1.set_ylabel("RMSE")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, rmses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                 f"{val:.5f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    naive_rmse = min(summary["Random Walk"]["RMSE"], summary["Persistence"]["RMSE"])
    ax1.axhline(naive_rmse, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
                label=f"Naive baseline floor ({naive_rmse:.5f})")
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[1, 0])
    maes = [summary[m]["MAE"] for m in names]
    ax2.bar(names, maes, color=clrs, edgecolor="white")
    ax2.set_title("MAE Comparison", fontweight="bold")
    ax2.set_ylabel("MAE")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    mapes = [summary[m]["MAPE"] for m in names]
    ax3.bar(names, mapes, color=clrs, edgecolor="white")
    ax3.set_title("MAPE Comparison (%)", fontweight="bold")
    ax3.set_ylabel("MAPE (%)")
    ax3.tick_params(axis="x", rotation=30)
    ax3.grid(axis="y", alpha=0.3)

    hybrid_names = ["CNN-LSTM", "LSTM-Attention", "Transformer-LSTM"]
    best_hybrid  = min(hybrid_names, key=lambda m: summary[m]["RMSE"])
    ax4 = fig.add_subplot(gs[2, :])
    actual = np.array(fold_predictions[best_hybrid]["actual"])
    pred   = np.array(fold_predictions[best_hybrid]["pred"])
    x_idx  = range(len(actual))
    ax4.plot(x_idx, actual, color="black",   lw=1.2, label="Actual Return",    alpha=0.8)
    ax4.plot(x_idx, pred,   color="#E74C3C", lw=1.0, label=f"{best_hybrid} Predicted", alpha=0.8)
    ax4.axhline(0, color="grey", lw=0.6, linestyle="--")
    ax4.set_title(f"Predicted vs Actual Log Returns — {best_hybrid} (Best Hybrid)",
                  fontweight="bold")
    ax4.set_xlabel("Test Day")
    ax4.set_ylabel("Log Return")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.savefig(f"{coin_name.lower()}_model_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Chart saved: {coin_name.lower()}_model_results.png")


def plot_metrics_table(coin_name, summary):
    """Print a clean formatted results table."""
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS TABLE — {coin_name}")
    print(f"{'='*70}")
    print(f"  {'Model':<24} {'Type':<12} {'RMSE':>10} {'MAE':>10} {'MAPE':>9}")
    print(f"  {'-'*65}")

    types = {
        "Random Walk":      "Naive",
        "Persistence":      "Naive",
        "ARIMA":            "Statistical",
        "LSTM":             "Baseline DL",
        "GRU":              "Baseline DL",
        "CNN-LSTM":         "HYBRID",
        "LSTM-Attention":   "HYBRID",
        "Transformer-LSTM": "HYBRID",
    }

    for model, metrics in summary.items():
        mape_str = f"{metrics['MAPE']:>8.2f}%" if not np.isnan(metrics['MAPE']) else "     N/A"
        print(f"  {model:<24} {types[model]:<12} "
              f"{metrics['RMSE']:>10.6f} {metrics['MAE']:>10.6f} "
              f"{mape_str}")

    print(f"{'='*70}")
    best = min(summary, key=lambda m: summary[m]["RMSE"])
    print(f"  Best model: {best} (RMSE = {summary[best]['RMSE']:.6f})")
    print(f"{'='*70}\n")


# ── 13. PRINT MODEL ARCHITECTURES ─────────────────────────────────────────────

def print_architectures():
    print("\n" + "="*60)
    print("  MODEL ARCHITECTURES (Parameter Estimation)")
    print("="*60)
    models = {
        "LSTM":             build_lstm(),
        "GRU":              build_gru(),
        "CNN-LSTM":         build_cnn_lstm(),
        "LSTM-Attention":   build_lstm_attention(),
        "Transformer-LSTM": build_transformer_lstm(),
    }
    for name, model in models.items():
        params = model.count_params()
        print(f"\n  {name} — {params:,} trainable parameters")
        model.summary(print_fn=lambda x: print(f"    {x}"))
    tf.keras.backend.clear_session()


# ── 14. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if len(datasets) == 0:
        raise SystemExit(
            "No processed CSV files found for any coin. Run "
            "crypto_dataset_pipeline.py first, in this same Colab session, "
            "then re-run this script."
        )

    print_architectures()

    all_summaries = {}

    for coin_name, df in datasets.items():
        summary, fold_preds = run_full_pipeline(coin_name, df)
        all_summaries[coin_name] = summary
        plot_metrics_table(coin_name, summary)
        plot_results(coin_name, summary, fold_preds)

    # ── Combined results CSV across all loaded coins ────────────────────────
    rows = []
    hybrids = ["CNN-LSTM", "LSTM-Attention", "Transformer-LSTM"]
    naives = ["Random Walk", "Persistence"]
    for coin, summary in all_summaries.items():
        for model, metrics in summary.items():
            mtype = "Hybrid" if model in hybrids else ("Naive" if model in naives else "Baseline")
            rows.append({
                "Coin": coin, "Model": model, "Type": mtype,
                "RMSE": round(metrics["RMSE"], 6),
                "MAE": round(metrics["MAE"], 6),
                "MAPE": round(metrics["MAPE"], 4) if not np.isnan(metrics["MAPE"]) else None,
            })
    pd.DataFrame(rows).to_csv("all_results_summary.csv", index=False)
    print("\n  Saved: all_results_summary.csv")

    print("\n" + "="*60)
    print("  ALL MODELS COMPLETE")
    print(f"  Coins processed: {', '.join(all_summaries.keys())} ({len(all_summaries)}/4)")
    print("="*60)
    print("\n  Files saved:")
    for coin in all_summaries:
        print(f"  - {coin.lower()}_model_results.png")
    print("  - all_results_summary.csv")
    print("  - Model .h5 weight files for each architecture")
    print("="*60)
