# ============================================================
# SUPERVISED REGIME DETECTION PIPELINE
# Random Forest + LSTM with NBER/Pagan-Sossounov labels
# Train: 1990-2015 | Test: 2015-2026
# Earlier split than unsupervised/ensemble to include COVID
# bear episode in evaluation window.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_recall_fscore_support
)
from sklearn.inspection import permutation_importance
from sklearn.mixture import GaussianMixture
from sklearn.calibration import CalibratedClassifierCV

from hmmlearn.hmm import GaussianHMM

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "sp500_features.csv")

LOOKBACK_ST         = 20
LOOKBACK_LT         = 126
ANNUALIZATION       = 252
RISK_AVERSION_GAMMA = 4.0
K_REGIMES           = 3
VIX_WEIGHT          = 0.5
MACRO_WEIGHT        = 0.01
ALPHA_SHORT         = 0.5

SEQ_LEN     = 20
BATCH_SIZE  = 32
EPOCHS      = 20
HIDDEN_SIZE = 32

# ---------------------------------------------------------------
# 2015 split — earlier than unsupervised/ensemble (2021) to
# ensure the March 2020 COVID bear episode falls in the test set.
# Documented in Section~\ref{sec:supervised}.
# ---------------------------------------------------------------
TRAIN_SPLIT_DATE    = "2015-01-01"
UNSUP_SPLIT_DATE    = "2021-01-01"   # unsupervised models still fit on pre-2021

WARMUP              = LOOKBACK_LT * 2
MACRO_LAGS          = {"CPI": 21, "Unemployment": 7, "FedFunds": 1}
TC_BPS              = 5
BAND_MULT           = 1.5
MIN_WEIGHT          = 0.0
MAX_WEIGHT          = 1.0
BAND_ROLLING_WINDOW = 252

MIN_BULL_DAYS = 70
MIN_BULL_RET  = 0.15

REGIME_OVERLAY = {0: -0.2, 1: 0.0, 2: 0.2}   # bear, neutral, bull

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# HELPERS
# ============================================================

def causal_ewm_mean(x, span):
    alpha   = 2.0 / (span + 1)
    result  = np.full(len(x), np.nan)
    ewm_val = np.nan
    for i, val in enumerate(x):
        if np.isnan(val):
            result[i] = np.nan
            continue
        ewm_val  = val if np.isnan(ewm_val) else (1 - alpha) * ewm_val + alpha * val
        result[i] = ewm_val
    return pd.Series(result, index=x.index)

def causal_ewm_std(x, span):
    alpha    = 2.0 / (span + 1)
    result   = np.full(len(x), np.nan)
    ewm_mean = np.nan
    ewm_var  = np.nan
    for i, val in enumerate(x):
        if np.isnan(val):
            result[i] = np.nan
            continue
        if np.isnan(ewm_mean):
            ewm_mean  = val
            ewm_var   = 0.0
            result[i] = np.nan
        else:
            prev_mean = ewm_mean
            ewm_mean  = (1 - alpha) * ewm_mean + alpha * val
            ewm_var   = (1 - alpha) * (ewm_var + alpha * (val - prev_mean) ** 2)
            result[i] = np.sqrt(ewm_var)
    return pd.Series(result, index=x.index)

def forward_filter(hmm_model, X):
    n_samples    = X.shape[0]
    n_states     = hmm_model.n_components
    log_emission = hmm_model._compute_log_likelihood(X)
    log_transmat = np.log(hmm_model.transmat_ + 1e-300)
    log_alpha    = np.full((n_samples, n_states), -np.inf)
    log_alpha[0] = np.log(hmm_model.startprob_ + 1e-300) + log_emission[0]
    for t in range(1, n_samples):
        for j in range(n_states):
            log_alpha[t, j] = (
                np.logaddexp.reduce(log_alpha[t - 1] + log_transmat[:, j])
                + log_emission[t, j]
            )
    return np.argmax(log_alpha, axis=1)

def forward_filter_proba(hmm_model, X):
    n_samples    = X.shape[0]
    n_states     = hmm_model.n_components
    log_emission = hmm_model._compute_log_likelihood(X)
    log_transmat = np.log(hmm_model.transmat_ + 1e-300)
    log_alpha    = np.full((n_samples, n_states), -np.inf)
    log_alpha[0] = np.log(hmm_model.startprob_ + 1e-300) + log_emission[0]
    for t in range(1, n_samples):
        for j in range(n_states):
            log_alpha[t, j] = (
                np.logaddexp.reduce(log_alpha[t - 1] + log_transmat[:, j])
                + log_emission[t, j]
            )
    alpha = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
    return alpha / alpha.sum(axis=1, keepdims=True)

def sharpe_ratio(rets):
    mu = rets.mean() * ANNUALIZATION
    sd = rets.std() * np.sqrt(ANNUALIZATION)
    return np.nan if sd == 0 else mu / sd

def max_drawdown(eq):
    return (eq / eq.cummax() - 1).min()

def segment_cagr(equity):
    if len(equity) < 2:
        return np.nan
    years = len(equity) / ANNUALIZATION
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

def pagan_sossounov_bull(px_series, min_days=70, min_ret=0.15):
    """
    Pagan-Sossounov (2003) peak-trough bull identification.
    Labels constructed offline over full price series — legitimate
    as training targets since no label information enters the
    causal prediction feature set.
    """
    prices  = px_series.values
    n       = len(prices)
    is_bull = np.zeros(n, dtype=bool)
    window  = min_days // 2

    troughs, peaks = [], []
    for i in range(window, n - window):
        if prices[i] == prices[max(0, i - window):i + window + 1].min():
            troughs.append(i)
        if prices[i] == prices[max(0, i - window):i + window + 1].max():
            peaks.append(i)

    for t_idx in troughs:
        next_peaks = [p for p in peaks if p > t_idx]
        if not next_peaks:
            continue
        p_idx      = next_peaks[0]
        duration   = p_idx - t_idx
        cum_return = (prices[p_idx] / prices[t_idx]) - 1
        if duration >= min_days and cum_return >= min_ret:
            is_bull[t_idx:p_idx + 1] = True

    return pd.Series(is_bull, index=px_series.index)

def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return acc, p, r, f

def run_backtest(regime_series, u_star_series, band_series,
                 ret_series, r_daily_series, overlay_map):
    dates      = regime_series.index
    w          = 0.0
    equity_val = 1.0
    eq_curve   = []
    wts        = []

    for date in dates:
        regime  = regime_series.loc[date]
        overlay = overlay_map.get(
            int(regime) if not np.isnan(regime) else 1, 0.0
        )
        tgt = float(np.clip(
            u_star_series.loc[date] + overlay,
            MIN_WEIGHT, MAX_WEIGHT
        ))
        b    = band_series.loc[date]
        diff = tgt - w

        if abs(diff) > b:
            direction  = np.sign(diff)
            new_w      = float(np.clip(
                w + direction * b, MIN_WEIGHT, MAX_WEIGHT
            ))
            traded     = abs(new_w - w)
            equity_val *= (1 - (TC_BPS / 1e4) * traded)
            w          = new_w

        port_ret   = w * ret_series.loc[date] + (1 - w) * r_daily_series.loc[date]
        equity_val *= (1 + port_ret)
        eq_curve.append(equity_val)
        wts.append(w)

    return (
        pd.Series(eq_curve, index=dates),
        pd.Series(wts,      index=dates)
    )

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0).sort_index()
df.columns = [c.strip() for c in df.columns]

price_col = [c for c in df.columns if "close" in c.lower()][0]
px        = df[price_col].astype(float)
logret    = np.log(px).diff()
ret       = px.pct_change()

# ============================================================
# MACRO PUBLICATION LAG
# ============================================================

macro_cols = [c for c in ["CPI", "Unemployment", "FedFunds"] if c in df.columns]
for col in macro_cols:
    df[col] = df[col].shift(MACRO_LAGS.get(col, 1))
df[macro_cols] = df[macro_cols].ffill()
print(f"Macro publication lags applied: { {c: MACRO_LAGS[c] for c in macro_cols} }")

if "FedFunds" in df.columns:
    r_daily = df["FedFunds"].astype(float)
    r_daily = r_daily / (100.0 if r_daily.max() > 1 else 1.0) / ANNUALIZATION
    r_daily = r_daily.reindex(df.index).ffill().fillna(0)
else:
    r_daily = pd.Series(0.0, index=df.index)

# ============================================================
# CAUSAL FEATURE ENGINEERING
# ============================================================

mu_st    = causal_ewm_mean(logret, LOOKBACK_ST) * ANNUALIZATION
mu_lt    = causal_ewm_mean(logret, LOOKBACK_LT) * ANNUALIZATION
sigma_lt = causal_ewm_std(logret,  LOOKBACK_LT) * np.sqrt(ANNUALIZATION)

macro_signal = pd.Series(0.0, index=df.index)
if macro_cols:
    z = df[macro_cols].rolling(LOOKBACK_LT).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8), raw=False
    ).clip(-3, 3)
    w_macro      = {"CPI": -0.4, "Unemployment": -0.3, "FedFunds": -0.3}
    macro_signal = sum(z[c] * w_macro[c] for c in z.columns if c in w_macro)

mu_ann = (
    ALPHA_SHORT         * mu_st
    + (1 - ALPHA_SHORT) * mu_lt
    + MACRO_WEIGHT      * macro_signal
)

if "VIX" in df.columns:
    vix_ann   = df["VIX"].astype(float) / 100.0
    sigma_ann = (
        (1 - VIX_WEIGHT) * sigma_lt
        + VIX_WEIGHT      * vix_ann.reindex(sigma_lt.index).ffill()
    ).clip(lower=1e-6)
    print("sigma_ann: 50% realized vol (EWM-126) + 50% VIX implied vol")
else:
    sigma_ann = sigma_lt.clip(lower=1e-6)
    print("VIX not found — sigma_ann uses realized vol only")

# ============================================================
# HJB OPTIMAL WEIGHT + NO-TRADE BAND
# ============================================================

excess_mu = mu_ann - r_daily * ANNUALIZATION
u_star    = (excess_mu / (RISK_AVERSION_GAMMA * sigma_ann ** 2)).clip(
    MIN_WEIGHT, MAX_WEIGHT
)
u_star.iloc[:WARMUP] = 0.0
print(f"Warmup: first {WARMUP} days zeroed (until {df.index[WARMUP].date()})")
u_star = u_star.shift(1).fillna(0.0)

sigma_rolling_median   = sigma_ann.rolling(
    window=BAND_ROLLING_WINDOW, min_periods=LOOKBACK_LT
).median()
sigma_expanding_median = sigma_ann.expanding(min_periods=1).median()
sigma_median           = sigma_rolling_median.fillna(sigma_expanding_median)

tc   = TC_BPS / 1e4
band = (
    BAND_MULT
    * np.sqrt(tc)
    * (sigma_ann / sigma_median).clip(0.5, 2.0)
    * 0.02
)
band = band.fillna(band.expanding(min_periods=1).median())
band.iloc[:WARMUP] = 999
band = band.shift(1).fillna(band.expanding(min_periods=1).median())

# ============================================================
# REGIME LABELS — PAGAN-SOSSOUNOV + NBER
# ============================================================

try:
    import pandas_datareader.data as web
    print("\nPulling USREC from FRED...")
    usrec_raw = web.DataReader(
        "USREC", "fred", start=df.index[0], end=df.index[-1]
    )
    usrec_raw.columns = ["recession"]
    print("  USREC loaded from FRED.")
except Exception as e:
    usrec_csv = os.path.join(BASE_DIR, "USREC.csv")
    if os.path.exists(usrec_csv):
        print("  Loading from local USREC.csv...")
        usrec_raw = pd.read_csv(usrec_csv, parse_dates=[0], index_col=0)
        usrec_raw.columns = ["recession"]
    else:
        print(f"  USREC unavailable ({e}) — using Pagan-Sossounov only.")
        usrec_raw = None

out = pd.DataFrame(index=df.index)
out["price"] = px

if usrec_raw is not None:
    usrec_daily  = usrec_raw["recession"].reindex(
        df.index, method="ffill"
    ).fillna(0)
    expansion_px = px.copy()
    expansion_px[usrec_daily == 1] = np.nan
    expansion_px = expansion_px.ffill()
    ps_bull      = pagan_sossounov_bull(
        expansion_px, MIN_BULL_DAYS, MIN_BULL_RET
    )
    out["regime_true"] = "neutral"
    out.loc[usrec_daily == 1,                       "regime_true"] = "bear"
    out.loc[(usrec_daily == 0) & (ps_bull == True), "regime_true"] = "bull"
else:
    ps_bull = pagan_sossounov_bull(px, MIN_BULL_DAYS, MIN_BULL_RET)
    ps_bear = pagan_sossounov_bull(-px, MIN_BULL_DAYS, MIN_BULL_RET)
    out["regime_true"] = "neutral"
    out.loc[ps_bull == True, "regime_true"] = "bull"
    out.loc[ps_bear == True, "regime_true"] = "bear"

print(f"\nFull-sample label distribution:")
print(out["regime_true"].value_counts())

label_map     = {"bear": 0, "neutral": 1, "bull": 2}
int_to_regime = {0: "bear", 1: "neutral", 2: "bull"}
y_all         = out["regime_true"].map(label_map)

# ============================================================
# UNSUPERVISED REGIMES — GMM + HMM
# Fit on pre-2021 data consistent with unsupervised chapter.
# Soft probabilities used as supervised features.
# ============================================================

feature_dict = {
    "ret":   logret,
    "vol":   sigma_ann,
    "mu":    mu_ann,
    "macro": macro_signal,
}
if "VIX" in df.columns:
    feature_dict["vix"] = df["VIX"].astype(float)

X_unsup_df = pd.DataFrame(feature_dict).dropna()

# Unsupervised models fit on pre-2021 — consistent with Chapter 5.2
X_unsup_pretrain_raw = X_unsup_df.loc[:UNSUP_SPLIT_DATE]
X_unsup_all_raw      = X_unsup_df   # full index for prediction

scaler_unsup     = StandardScaler()
X_unsup_pre_sc   = scaler_unsup.fit_transform(X_unsup_pretrain_raw)
X_unsup_all_sc   = scaler_unsup.transform(X_unsup_all_raw)

# GMM
gmm = GaussianMixture(n_components=K_REGIMES, random_state=42)
gmm.fit(X_unsup_pre_sc)

gmm_labels_pre = gmm.predict(X_unsup_pre_sc)
mean_ret_gmm   = {
    s: X_unsup_pretrain_raw["ret"].values[gmm_labels_pre == s].mean()
    for s in range(K_REGIMES)
}
gmm_col_order = sorted(mean_ret_gmm, key=mean_ret_gmm.get)

gmm_proba_all = gmm.predict_proba(X_unsup_all_sc)[:, gmm_col_order]

# HMM
hmm_model = GaussianHMM(
    n_components=K_REGIMES, covariance_type="full",
    n_iter=500, random_state=42
)
hmm_model.fit(X_unsup_pre_sc)

hmm_labels_pre = hmm_model.predict(X_unsup_pre_sc)
mean_ret_hmm   = {
    s: X_unsup_pretrain_raw["ret"].values[hmm_labels_pre == s].mean()
    for s in range(K_REGIMES)
}
hmm_col_order = sorted(mean_ret_hmm, key=mean_ret_hmm.get)

hmm_proba_all = forward_filter_proba(
    hmm_model, X_unsup_all_sc
)[:, hmm_col_order]

# ============================================================
# SUPERVISED FEATURE MATRIX
# Base features + GMM/HMM soft probabilities
# Supervised train/test split at 2015
# ============================================================

base_features = list(feature_dict.keys())
n_base        = len(base_features)

X_full = np.hstack([
    X_unsup_all_raw.values,
    gmm_proba_all,
    hmm_proba_all
])
feat_names = (
    base_features
    + ["gmm_bear", "gmm_neutral", "gmm_bull"]
    + ["hmm_bear", "hmm_neutral", "hmm_bull"]
)

y_aligned = y_all.reindex(X_unsup_df.index).dropna()
valid_mask = ~y_aligned.isna()
X_full     = X_full[valid_mask.values]
full_index = X_unsup_df.index[valid_mask.values]
y_full     = y_aligned[valid_mask].values.astype(int)

# Supervised split at 2015
train_mask = full_index < pd.Timestamp(TRAIN_SPLIT_DATE)
test_mask  = full_index >= pd.Timestamp(TRAIN_SPLIT_DATE)

X_sup_train_raw = X_full[train_mask]
X_sup_test_raw  = X_full[test_mask]
y_train         = y_full[train_mask]
y_test          = y_full[test_mask]
train_index     = full_index[train_mask]
test_index      = full_index[test_mask]

# Scale base features only — proba columns already in [0,1]
scaler_sup = StandardScaler()
X_sup_train = X_sup_train_raw.copy()
X_sup_test  = X_sup_test_raw.copy()
X_sup_train[:, :n_base] = scaler_sup.fit_transform(
    X_sup_train_raw[:, :n_base]
)
X_sup_test[:, :n_base]  = scaler_sup.transform(
    X_sup_test_raw[:, :n_base]
)

print(f"\nSupervised train: {train_index[0].date()} -> "
      f"{train_index[-1].date()} ({len(train_index)} obs)")
print(f"Supervised test:  {test_index[0].date()} -> "
      f"{test_index[-1].date()} ({len(test_index)} obs)")
print(f"Feature matrix — train: {X_sup_train.shape}, "
      f"test: {X_sup_test.shape}")

print(f"\nTest label distribution:")
test_labels = pd.Series(y_test).map(int_to_regime)
print(test_labels.value_counts())

# ============================================================
# RANDOM FOREST (CALIBRATED)
# ============================================================

print("\nFitting Random Forest...")
rf_base = RandomForestClassifier(
    n_estimators=300, max_depth=6,
    class_weight="balanced", random_state=42
)
rf_cal = CalibratedClassifierCV(rf_base, method="sigmoid", cv=5)
rf_cal.fit(X_sup_train, y_train)

rf_preds      = rf_cal.predict(X_sup_test)
rf_proba_test = rf_cal.predict_proba(X_sup_test)

out.loc[test_index, "regime_rf"] = rf_preds

print("\nRANDOM FOREST CLASSIFICATION REPORT")
print(classification_report(
    y_test, rf_preds,
    target_names=["Bear", "Neutral", "Bull"],
    zero_division=0
))

# Feature importance from uncalibrated base
rf_imp = RandomForestClassifier(
    n_estimators=300, max_depth=6,
    class_weight="balanced", random_state=42
)
rf_imp.fit(X_sup_train, y_train)
print("RF FEATURE IMPORTANCE (Gini):")
for name, imp in sorted(
    zip(feat_names, rf_imp.feature_importances_),
    key=lambda x: x[1], reverse=True
):
    print(f"  {name:<20}: {imp:.3f}")

# ============================================================
# LSTM
# ============================================================

class RegimeDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X       = X.astype(np.float32)
        self.y       = y.astype(np.int64)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.seq_len],
            self.y[idx + self.seq_len]
        )

class LSTMRegime(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(self.dropout(h[-1]))

train_ds     = RegimeDataset(X_sup_train, y_train, SEQ_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

lstm_model = LSTMRegime(
    X_sup_train.shape[1], HIDDEN_SIZE, K_REGIMES
).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    lstm_model.parameters(), lr=1e-3, weight_decay=1e-5
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5
)

print(f"\nFitting LSTM (input_size={X_sup_train.shape[1]}, "
      f"seq_len={SEQ_LEN}, epochs={EPOCHS})...")
lstm_model.train()
for epoch in range(EPOCHS):
    loss_sum = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(lstm_model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item()
    scheduler.step()
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {loss_sum/len(train_loader):.4f}")

# LSTM inference — sliding window, first SEQ_LEN rows get uniform prior
lstm_model.eval()
lstm_logits = []
with torch.no_grad():
    for i in range(len(X_sup_test) - SEQ_LEN):
        seq = torch.tensor(
            X_sup_test[i:i + SEQ_LEN], dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)
        lstm_logits.append(lstm_model(seq).cpu().numpy()[0])

lstm_proba_valid = torch.softmax(
    torch.tensor(np.array(lstm_logits)), dim=-1
).numpy()
lstm_proba_test  = np.vstack([
    np.full((SEQ_LEN, K_REGIMES), 1.0 / K_REGIMES),
    lstm_proba_valid
])
lstm_preds = lstm_proba_test.argmax(axis=1)

out.loc[test_index, "regime_lstm"] = lstm_preds

print("\nLSTM CLASSIFICATION REPORT")
print(classification_report(
    y_test, lstm_preds,
    target_names=["Bear", "Neutral", "Bull"],
    zero_division=0
))

# ============================================================
# TABLE 5.2 — EVALUATION METRICS
# ============================================================

table_52 = pd.DataFrame(
    columns=["Accuracy", "Precision", "Recall", "Macro-F1"]
)
table_52.loc["Random Forest"] = metrics(y_test, rf_preds)
table_52.loc["LSTM"]          = metrics(y_test, lstm_preds)

print("\nTABLE 5.2 — REGIME CLASSIFICATION PERFORMANCE")
print(table_52.round(3))

print("\nRF Regime Day Counts (test set):")
print(pd.Series(rf_preds).map(int_to_regime).value_counts())

print("\nLSTM Regime Day Counts (test set):")
print(pd.Series(lstm_preds).map(int_to_regime).value_counts())

# ============================================================
# BACKTEST — test period only, trade-to-boundary
# ============================================================

# Baseline — no regime overlay
w_b      = 0.0
eq_b     = 1.0
eq_base  = []
wts_base = []

for date in test_index:
    tgt  = float(u_star.loc[date])
    b    = band.loc[date]
    diff = tgt - w_b
    if abs(diff) > b:
        direction = np.sign(diff)
        new_w     = float(np.clip(
            w_b + direction * b, MIN_WEIGHT, MAX_WEIGHT
        ))
        eq_b *= (1 - tc * abs(new_w - w_b))
        w_b   = new_w
    port_ret = w_b * ret.loc[date] + (1 - w_b) * r_daily.loc[date]
    eq_b    *= (1 + port_ret)
    eq_base.append(eq_b)
    wts_base.append(w_b)

eq_baseline = pd.Series(eq_base, index=test_index)

# RF-guided backtest
eq_rf, wt_rf = run_backtest(
    pd.Series(rf_preds,   index=test_index),
    u_star.reindex(test_index),
    band.reindex(test_index),
    ret.reindex(test_index),
    r_daily.reindex(test_index),
    REGIME_OVERLAY
)

# LSTM-guided backtest
eq_lstm, wt_lstm = run_backtest(
    pd.Series(lstm_preds, index=test_index),
    u_star.reindex(test_index),
    band.reindex(test_index),
    ret.reindex(test_index),
    r_daily.reindex(test_index),
    REGIME_OVERLAY
)

# ============================================================
# PERFORMANCE TABLE
# ============================================================

print("\n" + "=" * 55)
print("PORTFOLIO PERFORMANCE (test set 2015-2026)")
print("=" * 55)
print(f"  {'Strategy':<22}  {'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>8}")
print(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*8}")

for label, eq in [
    ("Baseline (HJB)",   eq_baseline),
    ("RF + HJB",         eq_rf),
    ("LSTM + HJB",       eq_lstm),
]:
    rets = eq.pct_change().fillna(0)
    print(f"  {label:<22}  "
          f"{segment_cagr(eq):>7.2%}  "
          f"{sharpe_ratio(rets):>7.2f}  "
          f"{max_drawdown(eq):>8.2%}")

# ============================================================
# VISUALISATION — ACADEMIC STYLE
# ============================================================

import matplotlib as mpl
from matplotlib.patches import Patch

mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "text.usetex":       False,
})

colors_map = {0: "red", 1: "grey", 2: "green"}

# ── Figure 1 — Regime comparison (3-panel: RF, LSTM, True)
fig, axes = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
fig.subplots_adjust(hspace=0.55)

for ax, (preds, title) in zip(axes, [
    (rf_preds,  "Random Forest Regime Classification"),
    (lstm_preds, "LSTM Regime Classification"),
    (y_test,    "True Regime (Pagan-Sossounov and NBER)"),
]):
    for regime, color in colors_map.items():
        mask = np.array(preds) == regime
        ax.fill_between(
            test_index, 0, 1,
            where=mask,
            color=color, alpha=0.4,
            transform=ax.get_xaxis_transform()
        )
    ax.set_yticks([])
    ax.set_title(title, pad=2)
    ax.set_ylabel("Regime")

axes[-1].set_xlabel("Date")

legend_els = [
    Patch(facecolor="green", alpha=0.5, label="Bull"),
    Patch(facecolor="grey",  alpha=0.5, label="Neutral"),
    Patch(facecolor="red",   alpha=0.5, label="Bear"),
]
axes[0].legend(handles=legend_els, loc="upper right", framealpha=0.7)
fig.suptitle("Supervised Regime Detection - RF vs LSTM (Test: 2015-2026)",
             fontsize=9, fontstyle="italic")
plt.savefig("fig_supervised_regimes.png", dpi=300, bbox_inches="tight")
plt.show()

# ── Figure 2 — Feature importance
rf_importances = sorted(
    zip(feat_names, rf_imp.feature_importances_),
    key=lambda x: x[1], reverse=True
)
feat_labels = [f for f, _ in rf_importances]
feat_vals   = [v for _, v in rf_importances]

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.barh(feat_labels[::-1], feat_vals[::-1],
               color="steelblue", alpha=0.8, height=0.6)
ax.set_xlabel("Mean Gini Impurity Reduction")
ax.set_title("Random Forest Feature Importance")
ax.set_xlim(0, max(feat_vals) * 1.15)
for bar, val in zip(bars, feat_vals[::-1]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=6.5)
plt.tight_layout()
plt.savefig("fig_rf_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

# ── Figure 3 — Equity curves
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(test_index, eq_baseline.values, lw=1.2, ls="--",
        color="tab:grey",   label="Baseline (HJB only)")
ax.plot(test_index, eq_rf.values,       lw=1.2,
        color="tab:blue",   label="RF + HJB")
ax.plot(test_index, eq_lstm.values,     lw=1.2,
        color="tab:orange", label="LSTM + HJB")
ax.set_title("Equity Curves - Supervised Models vs Baseline (Test: 2015-2026)")
ax.set_xlabel("Date")
ax.set_ylabel("Normalised Wealth")
ax.legend(framealpha=0.7)
plt.tight_layout()
plt.savefig("fig_supervised_equity.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nDone. Saved: fig_supervised_regimes.png, fig_rf_feature_importance.png,")
print("             fig_supervised_equity.png")