"""
============================================================
 Cryptocurrency Price Prediction
 Actual vs Predicted Charts — Training & Test Split
 MSc Data Science | University of Roehampton London
 Student: Abiodun Adeyinka Adetoro | A00065756
============================================================
 Generates charts exactly as Dr Michael requested:
   - Black line  = Actual price / log return
   - Red line    = Model prediction on TRAINING data
   - Blue dashed = Model prediction on TEST data
   - Vertical line marks where training ends and test begins

 Run AFTER crypto_dataset_pipeline.py and crypto_models_v2.py
============================================================
"""

# ── 0. INSTALL (uncomment in Colab) ───────────────────────────────────────────
# !pip install yfinance ta pandas numpy scikit-learn statsmodels tensorflow matplotlib -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
import itertools
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D,
    MultiHeadAttention, LayerNormalization, Multiply,
    Activation, Lambda, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Libraries loaded.\n")

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────
SEQ_LEN      = 60
HORIZON      = 1
EPOCHS       = 100
BATCH_SIZE   = 32
LEARNING_RATE= 0.001
DROPOUT_RATE = 0.2
TRAIN_WINDOW = 730
TEST_WINDOW  = 90

FEATURE_COLS = [
    "Close", "Volume", "Log_Return", "Daily_Range",
    "RSI_14", "RSI_7", "MACD", "MACD_Sig", "MACD_Hist",
    "EMA_12", "EMA_26", "BB_Width", "ATR_14", "Vol_20",
    "Vol_Ratio", "FearGreed_Score"
]
N_FEATURES = len(FEATURE_COLS)

# ── 2. LOAD DATA ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading processed datasets")
print("=" * 60)
btc = pd.read_csv("bitcoin_processed.csv",  index_col="Date", parse_dates=True)
eth = pd.read_csv("ethereum_processed.csv", index_col="Date", parse_dates=True)
datasets = {"Bitcoin": btc, "Ethereum": eth}
print(f"  BTC: {btc.shape[0]} rows | ETH: {eth.shape[0]} rows\n")


# ── 3. HELPERS ────────────────────────────────────────────────────────────────

def walk_forward_splits(df, train_days=TRAIN_WINDOW, test_days=TEST_WINDOW):
    splits, n, start = [], len(df), train_days
    while start + test_days <= n:
        splits.append((list(range(0, start)),
                       list(range(start, min(start + test_days, n)))))
        start += test_days
    return splits


def build_sequences(df, train_idx, test_idx, seq_len=SEQ_LEN):
    """Leakage-free: scaler fitted on train only."""
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    scaler   = MinMaxScaler(feature_range=(0,1))
    tr_sc    = scaler.fit_transform(tr[FEATURE_COLS])
    te_sc    = scaler.transform(te[FEATURE_COLS])
    t_scaler = MinMaxScaler(feature_range=(0,1))
    tr_tgt   = t_scaler.fit_transform(tr[["Target"]])
    te_tgt   = t_scaler.transform(te[["Target"]])
    def make_seq(X, y, sl):
        Xs, ys = [], []
        for i in range(sl, len(X)):
            Xs.append(X[i-sl:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)
    X_tr, y_tr = make_seq(tr_sc, tr_tgt, seq_len)
    X_te, y_te = make_seq(te_sc, te_tgt, seq_len)
    return X_tr, y_tr, X_te, y_te, scaler, t_scaler


def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0)
    ]


def compute_metrics(actual, predicted):
    actual, predicted = np.array(actual).flatten(), np.array(predicted).flatten()
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    return rmse, mae


# ── 4. MODEL BUILDERS ────────────────────────────────────────────────────────

def build_lstm():
    m = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, N_FEATURES)),
        Dropout(DROPOUT_RATE),
        LSTM(64), Dropout(DROPOUT_RATE),
        Dense(32, activation="relu"), Dense(1)
    ], name="LSTM")
    m.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
    return m

def build_gru():
    m = Sequential([
        GRU(128, return_sequences=True, input_shape=(SEQ_LEN, N_FEATURES)),
        Dropout(DROPOUT_RATE),
        GRU(64), Dropout(DROPOUT_RATE),
        Dense(32, activation="relu"), Dense(1)
    ], name="GRU")
    m.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
    return m

def build_cnn_lstm():
    inp = Input(shape=(SEQ_LEN, N_FEATURES))
    x = Conv1D(64, 3, activation="relu", padding="same")(inp)
    x = Dropout(DROPOUT_RATE)(x)
    x = Conv1D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = LSTM(64)(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m = Model(inp, out, name="CNN_LSTM")
    m.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
    return m

def build_lstm_attention():
    inp = Input(shape=(SEQ_LEN, N_FEATURES))
    lstm_out = LSTM(128, return_sequences=True)(inp)
    lstm_out = Dropout(DROPOUT_RATE)(lstm_out)
    scores  = Dense(1, activation="tanh")(lstm_out)
    weights = Activation("softmax")(scores)
    context = Multiply()([lstm_out, weights])
    context = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
    x = Dense(64, activation="relu")(context)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m = Model(inp, out, name="LSTM_Attention")
    m.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
    return m

def build_transformer_lstm():
    inp = Input(shape=(SEQ_LEN, N_FEATURES))
    x = Dense(64)(inp)
    attn = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    attn = Dropout(DROPOUT_RATE)(attn)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))
    ff = Dense(128, activation="relu")(x)
    ff = Dense(64)(ff)
    ff = Dropout(DROPOUT_RATE)(ff)
    x  = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))
    x  = LSTM(64)(x)
    x  = Dropout(DROPOUT_RATE)(x)
    x  = Dense(32, activation="relu")(x)
    out = Dense(1)(x)
    m = Model(inp, out, name="Transformer_LSTM")
    m.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
    return m


# ── 5. CORE: TRAIN MODEL AND GET BOTH TRAIN & TEST PREDICTIONS ────────────────

def get_train_test_predictions(model, X_tr, y_tr, X_te, y_te, t_scaler):
    """
    Train model, then predict on BOTH training AND test data.
    Returns: train_actual, train_pred, test_actual, test_pred (all inverse-transformed)
    """
    model.fit(X_tr, y_tr,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_split=0.1,
              callbacks=get_callbacks(),
              verbose=0)

    # Training predictions
    tr_preds_sc = model.predict(X_tr, verbose=0)
    tr_preds    = t_scaler.inverse_transform(tr_preds_sc).flatten()
    tr_actual   = t_scaler.inverse_transform(y_tr).flatten()

    # Test predictions
    te_preds_sc = model.predict(X_te, verbose=0)
    te_preds    = t_scaler.inverse_transform(te_preds_sc).flatten()
    te_actual   = t_scaler.inverse_transform(y_te).flatten()

    return tr_actual, tr_preds, te_actual, te_preds


def run_arima_train_test(df, train_idx, test_idx):
    """ARIMA: fit on train, predict train and test separately."""
    series = df["Log_Return"]
    tr_series = series.iloc[train_idx]
    te_series = series.iloc[test_idx]

    # Grid search
    best_aic, best_order = np.inf, (1, 0, 1)
    for p, d, q in itertools.product([0,1,2],[0,1],[0,1,2]):
        try:
            m = ARIMA(tr_series, order=(p,d,q)).fit()
            if m.aic < best_aic:
                best_aic, best_order = m.aic, (p,d,q)
        except:
            continue

    model = ARIMA(tr_series, order=best_order).fit()

    # In-sample (training) predictions
    tr_pred  = model.fittedvalues.values
    tr_actual = tr_series.values

    # Out-of-sample (test) forecast
    te_pred   = model.forecast(steps=len(te_series)).values
    te_actual = te_series.values

    # Align lengths (ARIMA fitted values may differ slightly)
    min_tr = min(len(tr_pred), len(tr_actual))
    tr_pred, tr_actual = tr_pred[-min_tr:], tr_actual[-min_tr:]

    print(f"    ARIMA{best_order} selected")
    return tr_actual, tr_pred, te_actual, te_pred


# ── 6. PLOTTING FUNCTION ──────────────────────────────────────────────────────

def plot_actual_vs_predicted(coin_name, model_results):
    """
    Dr Michael's requested plot:
    - Black solid line  = Actual log return
    - Red solid line    = Training predictions
    - Blue dashed line  = Test predictions
    - Vertical grey line marks the train/test boundary
    """
    n_models = len(model_results)
    n_cols   = 2
    n_rows   = (n_models + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    fig.suptitle(
        f"{coin_name} — Observed vs. Predicted Log Returns by Model\n"
        f"Black = Actual  |  Red = Training Predictions  |  Blue (dashed) = Test Predictions",
        fontsize=13, fontweight="bold", y=1.01
    )
    axes = axes.flatten()

    for idx, (model_name, data) in enumerate(model_results.items()):
        ax = axes[idx]

        tr_actual = data["tr_actual"]
        tr_pred   = data["tr_pred"]
        te_actual = data["te_actual"]
        te_pred   = data["te_pred"]
        rmse_tr   = data["rmse_tr"]
        rmse_te   = data["rmse_te"]

        n_tr = len(tr_actual)
        n_te = len(te_actual)
        total = n_tr + n_te

        x_tr = np.arange(n_tr)
        x_te = np.arange(n_tr, n_tr + n_te)

        # ── Plot actual (black) ────────────────────────────────────────────
        ax.plot(x_tr, tr_actual, color="black", lw=1.0,
                label="Actual", alpha=0.85, zorder=3)
        ax.plot(x_te, te_actual, color="black", lw=1.0,
                alpha=0.85, zorder=3)

        # ── Plot training predictions (red solid) ──────────────────────────
        ax.plot(x_tr, tr_pred, color="red", lw=1.0, linestyle="-",
                label=f"Training Pred (RMSE={rmse_tr:.4f})", alpha=0.8, zorder=4)

        # ── Plot test predictions (blue dashed) ───────────────────────────
        ax.plot(x_te, te_pred, color="blue", lw=1.2, linestyle="--",
                label=f"Test Pred (RMSE={rmse_te:.4f})", alpha=0.9, zorder=5)

        # ── Vertical dividing line ────────────────────────────────────────
        ax.axvline(x=n_tr, color="grey", lw=1.5, linestyle=":",
                   alpha=0.7, zorder=2)
        ax.text(n_tr + 1, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0
                else ax.get_ylim()[0] * 0.85,
                "TEST\nSTARTS", fontsize=7, color="grey", va="top")

        # ── Zero line ─────────────────────────────────────────────────────
        ax.axhline(0, color="grey", lw=0.5, linestyle="-", alpha=0.4)

        # ── Formatting ────────────────────────────────────────────────────
        is_hybrid = model_name in ["CNN-LSTM", "LSTM-Attention", "Transformer-LSTM"]
        title_prefix = "★ HYBRID: " if is_hybrid else "Baseline: "
        ax.set_title(f"{title_prefix}{model_name}", fontweight="bold",
                     fontsize=11, color="#1F3864" if is_hybrid else "#444444")
        ax.set_xlabel("Time Step (Days)", fontsize=9)
        ax.set_ylabel("Log Return", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # Shade test region
        ax.axvspan(n_tr, n_tr + n_te, alpha=0.06, color="blue")

    # Hide unused subplots
    for idx in range(len(model_results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fname = f"{coin_name.lower()}_actual_vs_predicted.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Chart saved: {fname}\n")
    return fname


# ── 7. FULL RUN ───────────────────────────────────────────────────────────────

def run_and_plot(coin_name, df):
    print(f"\n{'='*60}")
    print(f"  COIN: {coin_name}")
    print(f"{'='*60}")

    splits  = walk_forward_splits(df)
    # Use fold 1 (first fold) for the train/test split plot
    # This gives the clearest illustration of training vs test behaviour
    train_idx, test_idx = splits[0]

    print(f"  Using Fold 1:")
    print(f"    Train: {df.index[train_idx[0]].date()} → {df.index[train_idx[-1]].date()} ({len(train_idx)} days)")
    print(f"    Test:  {df.index[test_idx[0]].date()} → {df.index[test_idx[-1]].date()} ({len(test_idx)} days)\n")

    X_tr, y_tr, X_te, y_te, scaler, t_scaler = build_sequences(df, train_idx, test_idx)

    model_results = {}

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    print("  Training ARIMA ...")
    tr_a, tr_p, te_a, te_p = run_arima_train_test(df, train_idx, test_idx)
    rmse_tr, _ = compute_metrics(tr_a, tr_p)
    rmse_te, _ = compute_metrics(te_a, te_p)
    model_results["ARIMA"] = {
        "tr_actual": tr_a, "tr_pred": tr_p,
        "te_actual": te_a, "te_pred": te_p,
        "rmse_tr": rmse_tr, "rmse_te": rmse_te
    }
    print(f"    Train RMSE={rmse_tr:.5f} | Test RMSE={rmse_te:.5f}")

    # ── Deep learning models ──────────────────────────────────────────────────
    dl_models = {
        "LSTM":             build_lstm(),
        "GRU":              build_gru(),
        "CNN-LSTM":         build_cnn_lstm(),
        "LSTM-Attention":   build_lstm_attention(),
        "Transformer-LSTM": build_transformer_lstm(),
    }

    for model_name, model in dl_models.items():
        print(f"  Training {model_name} ...")
        tr_a, tr_p, te_a, te_p = get_train_test_predictions(
            model, X_tr, y_tr, X_te, y_te, t_scaler)
        rmse_tr, _ = compute_metrics(tr_a, tr_p)
        rmse_te, _ = compute_metrics(te_a, te_p)
        model_results[model_name] = {
            "tr_actual": tr_a, "tr_pred": tr_p,
            "te_actual": te_a, "te_pred": te_p,
            "rmse_tr": rmse_tr, "rmse_te": rmse_te
        }
        print(f"    Train RMSE={rmse_tr:.5f} | Test RMSE={rmse_te:.5f}")
        tf.keras.backend.clear_session()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n  {'Model':<24} {'Train RMSE':>12} {'Test RMSE':>12}")
    print(f"  {'-'*48}")
    for mname, data in model_results.items():
        tag = " ← HYBRID" if mname in ["CNN-LSTM","LSTM-Attention","Transformer-LSTM"] else ""
        print(f"  {mname:<24} {data['rmse_tr']:>12.5f} {data['rmse_te']:>12.5f}{tag}")

    # ── Generate the Dr Michael-style chart ───────────────────────────────────
    print(f"\n  Generating charts ...")
    fname = plot_actual_vs_predicted(coin_name, model_results)

    return model_results, fname


# ── 8. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chart_files = []

    for coin_name, df in datasets.items():
        _, fname = run_and_plot(coin_name, df)
        chart_files.append(fname)

    print("\n" + "="*60)
    print("  ALL CHARTS GENERATED")
    print("="*60)
    print("\n  Download your charts:")
    for f in chart_files:
        print(f"  - {f}")

    print("""
  To download in Colab, run:
    from google.colab import files
    files.download("bitcoin_actual_vs_predicted.png")
    files.download("ethereum_actual_vs_predicted.png")
""")
    print("="*60)
