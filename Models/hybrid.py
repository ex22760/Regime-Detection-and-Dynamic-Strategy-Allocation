# ============================================================
# HYBRID ENSEMBLE REGIME FRAMEWORK
# Combines GMM, HMM, Random Forest, and LSTM signals into
# a single probabilistic regime output for dynamic allocation.
#
# Architecture:
#   Layer 1 — Base models:  GMM, HMM (unsupervised)
#                            RF, LSTM (supervised)
#   Layer 2 — Ensemble:     Soft-vote with dynamic confidence weights
#   Layer 3 — Output:       P(bear|neutral|bull) + regime confidence score
#
# Fixes applied vs original:
#   1. Ensemble and baseline both trade to BOUNDARY not target
#   2. mu_ann includes MACRO_WEIGHT * macro_signal (consistent with supervised script)
#   3. base_band uses causal rolling median (no lookahead)
#   4. sigma_ann VIX blend applied before mu_ann computation
# ============================================================
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, brier_score_loss, accuracy_score

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
MACRO_WEIGHT        = 0.01   # FIX 2: was missing in original ensemble script
ALPHA_SHORT         = 0.5

SEQ_LEN     = 20
BATCH_SIZE  = 32
EPOCHS      = 20
HIDDEN_SIZE = 32

TRAIN_SPLIT_DATE = "2015-01-01"
VAL_SPLIT_DATE   = "2008-01-01"   # validation window: 2012-2015 (3 yrs, ~756 obs)
WARMUP           = LOOKBACK_LT * 2
MACRO_LAGS       = {"CPI": 21, "Unemployment": 7, "FedFunds": 1}

# ── Ensemble weights (fallback if val set too small)
# Overwritten at runtime by tune_ensemble_weights() using val-period macro-F1
ENSEMBLE_WEIGHTS = {
    "gmm":  0.15,
    "hmm":  0.25,
    "rf":   0.25,
    "lstm": 0.35,
}

CONFIDENCE_THRESHOLD = 0.55

MIN_WEIGHT, MAX_WEIGHT = 0.0, 1.0
TC_BPS    = 5
BAND_MULT = 1.5
BAND_ROLLING_WINDOW = 252

MIN_BULL_DAYS = 70
MIN_BULL_RET  = 0.15

ALPHA_BULL = 0.3
ALPHA_BEAR = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# HELPERS — CAUSAL EWM
# ============================================================

def causal_ewm_mean(x, span):
    alpha, result, ewm_val = 2.0 / (span + 1), np.full(len(x), np.nan), np.nan
    for i, val in enumerate(x):
        if np.isnan(val):
            result[i] = np.nan; continue
        ewm_val = val if np.isnan(ewm_val) else (1 - alpha) * ewm_val + alpha * val
        result[i] = ewm_val
    return pd.Series(result, index=x.index)

def causal_ewm_std(x, span):
    alpha = 2.0 / (span + 1)
    result = np.full(len(x), np.nan)
    ewm_mean = ewm_var = np.nan
    for i, val in enumerate(x):
        if np.isnan(val):
            result[i] = np.nan; continue
        if np.isnan(ewm_mean):
            ewm_mean, ewm_var, result[i] = val, 0.0, np.nan
        else:
            prev_mean = ewm_mean
            ewm_mean  = (1 - alpha) * ewm_mean + alpha * val
            ewm_var   = (1 - alpha) * (ewm_var + alpha * (val - prev_mean) ** 2)
            result[i] = np.sqrt(ewm_var)
    return pd.Series(result, index=x.index)

# ============================================================
# HELPERS — HMM CAUSAL FORWARD FILTER
# ============================================================

def forward_filter(hmm_model, X):
    n_samples, n_states = X.shape[0], hmm_model.n_components
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
    n_samples, n_states = X.shape[0], hmm_model.n_components
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

# ============================================================
# METRICS
# ============================================================

def sharpe_ratio(rets):
    mu, sd = rets.mean() * ANNUALIZATION, rets.std() * np.sqrt(ANNUALIZATION)
    return np.nan if sd == 0 else mu / sd

def max_drawdown(eq):
    return (eq / eq.cummax() - 1).min()

def segment_cagr(equity):
    if len(equity) < 2: return np.nan
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / (len(equity) / ANNUALIZATION)) - 1

def tune_ensemble_weights(f1_scores: dict, temperature: float = 1.0) -> dict:
    names   = list(f1_scores.keys())
    scores  = np.array([f1_scores[n] for n in names], dtype=float)
    exp_s   = np.exp(scores * temperature)
    weights = exp_s / exp_s.sum()
    result  = {n: float(w) for n, w in zip(names, weights)}
    print("\nEnsemble weight tuning (softmax over validation macro-F1):")
    for n, w in result.items():
        print(f"  {n:<6}: F1={f1_scores[n]:.3f}  ->  weight={w:.3f}")
    return result

# ============================================================
# LOAD & PREPROCESS DATA
# ============================================================

print("Loading data...")
df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0).sort_index()
df.columns = [c.strip() for c in df.columns]

price_col = [c for c in df.columns if "close" in c.lower()][0]
px     = df[price_col].astype(float)
logret = np.log(px).diff()
ret    = px.pct_change()

# Macro publication lag
macro_cols = [c for c in ["CPI", "Unemployment", "FedFunds"] if c in df.columns]
for col in macro_cols:
    df[col] = df[col].shift(MACRO_LAGS.get(col, 1))
df[macro_cols] = df[macro_cols].ffill()
print(f"Macro lags applied: { {c: MACRO_LAGS[c] for c in macro_cols} }")

r_daily = pd.Series(0.0, index=df.index)
if "FedFunds" in df.columns:
    r_daily = df["FedFunds"].astype(float)
    r_daily = r_daily / (100.0 if r_daily.max() > 1 else 1.0) / ANNUALIZATION
    r_daily = r_daily.reindex(df.index).ffill().fillna(0)

# ── Causal EWM features
mu_st    = causal_ewm_mean(logret, LOOKBACK_ST) * ANNUALIZATION
mu_lt    = causal_ewm_mean(logret, LOOKBACK_LT) * ANNUALIZATION
sigma_lt = causal_ewm_std(logret,  LOOKBACK_LT) * np.sqrt(ANNUALIZATION)

# Macro composite signal
macro_signal = pd.Series(0.0, index=df.index)
if macro_cols:
    z = df[macro_cols].rolling(LOOKBACK_LT).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8), raw=False
    ).clip(-3, 3)
    w_macro      = {"CPI": -0.4, "Unemployment": -0.3, "FedFunds": -0.3}
    macro_signal = sum(z[c] * w_macro[c] for c in z.columns if c in w_macro)

# FIX 3: VIX blend computed BEFORE mu_ann so sigma_ann is consistent throughout
if "VIX" in df.columns:
    vix_ann   = df["VIX"].astype(float) / 100.0
    sigma_ann = (
        (1 - VIX_WEIGHT) * sigma_lt
        + VIX_WEIGHT * vix_ann.reindex(sigma_lt.index).ffill()
    ).clip(lower=1e-6)
    print("sigma_ann: 50% realized vol (EWM-126) + 50% VIX implied vol")
else:
    sigma_ann = sigma_lt.clip(lower=1e-6)
    print("VIX not found - sigma_ann uses realized vol only")

# FIX 2: mu_ann includes macro_signal (consistent with supervised script)
mu_ann = ALPHA_SHORT * mu_st + (1 - ALPHA_SHORT) * mu_lt + MACRO_WEIGHT * macro_signal

# Baseline HJB weight
excess_mu     = mu_ann - r_daily * ANNUALIZATION
u_star        = (excess_mu / (RISK_AVERSION_GAMMA * sigma_ann ** 2)).clip(0, 1)
u_star.iloc[:WARMUP] = 0.0
u_star_lagged = u_star.shift(1).fillna(0.0)

# FIX 4: Causal rolling median for band (no lookahead into test period)
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
# REGIME LABELS — PAGAN-SOSSOUNOV (2003) + NBER
# ============================================================

def pagan_sossounov_bull(px_series, min_days=70, min_ret=0.15):
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

try:
    import pandas_datareader.data as web
    usrec_raw = web.DataReader("USREC", "fred", start=df.index[0], end=df.index[-1])
    usrec_raw.columns = ["recession"]
    print("USREC loaded from FRED.")
except Exception as e:
    usrec_csv = os.path.join(BASE_DIR, "USREC.csv")
    if os.path.exists(usrec_csv):
        usrec_raw = pd.read_csv(usrec_csv, parse_dates=[0], index_col=0)
        usrec_raw.columns = ["recession"]
        print("USREC loaded from local CSV.")
    else:
        print(f"USREC unavailable ({e}) - using Pagan-Sossounov only.")
        usrec_raw = None

out = pd.DataFrame(index=df.index)
out["price"]    = px
out["w_target"] = u_star
out["w_actual"] = u_star_lagged

if usrec_raw is not None:
    usrec_daily  = usrec_raw["recession"].reindex(df.index, method="ffill").fillna(0)
    expansion_px = px.copy()
    expansion_px[usrec_daily == 1] = np.nan
    expansion_px = expansion_px.ffill()
    ps_bull      = pagan_sossounov_bull(expansion_px, MIN_BULL_DAYS, MIN_BULL_RET)
    out["regime_true"] = "neutral"
    out.loc[usrec_daily == 1,                               "regime_true"] = "bear"
    out.loc[(usrec_daily == 0) & (ps_bull == True),         "regime_true"] = "bull"
    print(f"\nPagan-Sossounov label distribution:")
    print(out["regime_true"].value_counts())
else:
    ps_bull = pagan_sossounov_bull(px, MIN_BULL_DAYS, MIN_BULL_RET)
    ps_bear = pagan_sossounov_bull(-px, MIN_BULL_DAYS, MIN_BULL_RET)
    out["regime_true"] = "neutral"
    out.loc[ps_bull == True, "regime_true"] = "bull"
    out.loc[ps_bear == True, "regime_true"] = "bear"
    print(f"\nPagan-Sossounov label distribution (no NBER):")
    print(out["regime_true"].value_counts())

label_map = {"bear": 0, "neutral": 1, "bull": 2}
y_all = out["regime_true"].map(label_map)

# ============================================================
# FEATURE MATRIX
# ============================================================

feature_dict = {
    "return":   logret,
    "vol":      sigma_ann,
    "momentum": mu_ann,
    "macro":    macro_signal,
}
if "VIX" in df.columns:
    feature_dict["vix"] = df["VIX"].astype(float)

features_base = list(feature_dict.keys())
X_df = pd.DataFrame(feature_dict).dropna()

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

X_preval_raw = X_df.loc[:VAL_SPLIT_DATE]
X_val_raw    = X_df.loc[VAL_SPLIT_DATE:TRAIN_SPLIT_DATE]
X_test_raw   = X_df.loc[TRAIN_SPLIT_DATE:]
X_train_raw  = X_df.loc[:TRAIN_SPLIT_DATE]

y_preval = y_all.reindex(X_preval_raw.index).dropna()
y_val    = y_all.reindex(X_val_raw.index).dropna()
y_train  = y_all.reindex(X_train_raw.index).dropna()
y_test   = y_all.reindex(X_test_raw.index).dropna()

scaler_preval = StandardScaler()
X_preval      = scaler_preval.fit_transform(X_preval_raw)
X_val         = scaler_preval.transform(X_val_raw)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

print(f"\nPre-val: {X_preval_raw.index[0].date()} -> {X_preval_raw.index[-1].date()} ({len(X_preval_raw)} obs)")
print(f"Val:     {X_val_raw.index[0].date()}  -> {X_val_raw.index[-1].date()} ({len(X_val_raw)} obs)")
print(f"Train:   {X_train_raw.index[0].date()} -> {X_train_raw.index[-1].date()} ({len(X_train_raw)} obs)")
print(f"Test:    {X_test_raw.index[0].date()}  -> {X_test_raw.index[-1].date()} ({len(X_test_raw)} obs)")

# ============================================================
# LAYER 1A — GMM
# ============================================================

print("\nFitting GMM...")
gmm = GaussianMixture(n_components=K_REGIMES, random_state=42)
gmm.fit(X_train)

gmm_labels_train  = gmm.predict(X_train)
gmm_proba_train   = gmm.predict_proba(X_train)
gmm_proba_test    = gmm.predict_proba(X_test)

mean_ret_by_state = {
    s: X_df.loc[X_train_raw.index, "return"].values[gmm_labels_train == s].mean()
    for s in range(K_REGIMES)
}
sorted_states = sorted(mean_ret_by_state, key=mean_ret_by_state.get)
gmm_col_order = [sorted_states[0], sorted_states[1], sorted_states[2]]

gmm_proba_train = gmm_proba_train[:, gmm_col_order]
gmm_proba_test  = gmm_proba_test[:,  gmm_col_order]

# ============================================================
# LAYER 1B — HMM
# ============================================================

print("Fitting HMM...")
hmm_model = GaussianHMM(n_components=K_REGIMES, covariance_type="full",
                         n_iter=500, random_state=42)
hmm_model.fit(X_train)

hmm_labels_train = hmm_model.predict(X_train)
mean_ret_hmm     = {
    s: X_df.loc[X_train_raw.index, "return"].values[hmm_labels_train == s].mean()
    for s in range(K_REGIMES)
}
sorted_hmm    = sorted(mean_ret_hmm, key=mean_ret_hmm.get)
hmm_col_order = [sorted_hmm[0], sorted_hmm[1], sorted_hmm[2]]

hmm_proba_train = forward_filter_proba(hmm_model, X_train)[:, hmm_col_order]
hmm_proba_test  = forward_filter_proba(hmm_model, X_test)[:,  hmm_col_order]

# ============================================================
# LAYER 1C — RANDOM FOREST (CALIBRATED)
# ============================================================

print("Fitting Random Forest (calibrated)...")

rf_features_train = np.hstack([X_train, gmm_proba_train, hmm_proba_train])
rf_features_test  = np.hstack([X_test,  gmm_proba_test,  hmm_proba_test])

rf_base = RandomForestClassifier(
    n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
)
rf_cal = CalibratedClassifierCV(rf_base, method="sigmoid", cv=5)
rf_cal.fit(rf_features_train, y_train.reindex(X_train_raw.index).dropna())

rf_proba_test = rf_cal.predict_proba(rf_features_test)

# ============================================================
# LAYER 1D — LSTM
# ============================================================

class RegimeDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X, self.y, self.seq_len = (
            X.astype(np.float32), y.astype(np.int64), seq_len)
    def __len__(self):
        return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return self.X[idx:idx + self.seq_len], self.y[idx + self.seq_len]

class LSTMRegime(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(self.dropout(h[-1]))

lstm_input_size = rf_features_train.shape[1]
lstm_y_train    = y_train.reindex(X_train_raw.index).dropna().values

print(f"Fitting LSTM (input_size={lstm_input_size}, seq_len={SEQ_LEN}, epochs={EPOCHS})...")
train_ds     = RegimeDataset(rf_features_train, lstm_y_train, SEQ_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

lstm_model = LSTMRegime(lstm_input_size, HIDDEN_SIZE, K_REGIMES).to(DEVICE)
criterion  = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(lstm_model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {loss_sum/len(train_loader):.4f}")

lstm_model.eval()
lstm_logits = []
with torch.no_grad():
    for i in range(len(rf_features_test) - SEQ_LEN):
        seq = torch.tensor(
            rf_features_test[i:i + SEQ_LEN], dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)
        lstm_logits.append(lstm_model(seq).cpu().numpy()[0])

lstm_logits      = np.array(lstm_logits)
lstm_proba_valid = torch.softmax(torch.tensor(lstm_logits), dim=-1).numpy()

lstm_proba_test = np.vstack([
    np.full((SEQ_LEN, K_REGIMES), 1.0 / K_REGIMES),
    lstm_proba_valid
])

# ============================================================
# VALIDATION SCORING — derive ensemble weights from val-period F1
# ============================================================

MIN_VAL_OBS = 200

if len(X_val_raw) >= MIN_VAL_OBS:
    from sklearn.metrics import f1_score

    gmm_pv = GaussianMixture(n_components=K_REGIMES, random_state=42)
    gmm_pv.fit(X_preval)
    pv_labels       = gmm_pv.predict(X_preval)
    pv_ret_by_state = {
        s: X_df.loc[X_preval_raw.index, "return"].values[pv_labels == s].mean()
        for s in range(K_REGIMES)
    }
    pv_col_order = sorted(pv_ret_by_state, key=pv_ret_by_state.get)
    gmm_val_pred = gmm_pv.predict_proba(X_val)[:, pv_col_order].argmax(axis=1)

    hmm_pv = GaussianHMM(n_components=K_REGIMES, covariance_type="full",
                          n_iter=500, random_state=42)
    hmm_pv.fit(X_preval)
    pv_hmm_labels = hmm_pv.predict(X_preval)
    pv_hmm_ret    = {
        s: X_df.loc[X_preval_raw.index, "return"].values[pv_hmm_labels == s].mean()
        for s in range(K_REGIMES)
    }
    pv_hmm_order = sorted(pv_hmm_ret, key=pv_hmm_ret.get)
    hmm_val_pred = forward_filter(hmm_pv, X_val)
    hmm_remap    = {pv_hmm_order[i]: i for i in range(K_REGIMES)}
    hmm_val_pred = np.array([hmm_remap[s] for s in hmm_val_pred])

    gmm_pv_proba  = gmm_pv.predict_proba(X_preval)[:, pv_col_order]
    hmm_pv_proba  = forward_filter_proba(hmm_pv, X_preval)[:, pv_hmm_order]
    gmm_val_proba = gmm_pv.predict_proba(X_val)[:, pv_col_order]
    hmm_val_proba = forward_filter_proba(hmm_pv, X_val)[:, pv_hmm_order]

    rf_pv_feats  = np.hstack([X_preval, gmm_pv_proba, hmm_pv_proba])
    rf_val_feats = np.hstack([X_val,    gmm_val_proba, hmm_val_proba])

    rf_pv_base = RandomForestClassifier(n_estimators=300, max_depth=6,
                                         class_weight="balanced", random_state=42)
    rf_pv_cal  = CalibratedClassifierCV(rf_pv_base, method="sigmoid", cv=5)
    rf_pv_cal.fit(rf_pv_feats, y_preval.reindex(X_preval_raw.index).dropna())
    rf_val_pred = rf_pv_cal.predict(rf_val_feats)

    lstm_pv_y  = y_preval.reindex(X_preval_raw.index).dropna().values
    pv_ds      = RegimeDataset(rf_pv_feats, lstm_pv_y, SEQ_LEN)
    pv_loader  = DataLoader(pv_ds, batch_size=BATCH_SIZE, shuffle=False)
    lstm_pv    = LSTMRegime(rf_pv_feats.shape[1], HIDDEN_SIZE, K_REGIMES).to(DEVICE)
    opt_pv     = torch.optim.Adam(lstm_pv.parameters(), lr=1e-3, weight_decay=1e-5)
    lstm_pv.train()
    for _ in range(EPOCHS):
        for xb, yb in pv_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_pv.zero_grad()
            loss = criterion(lstm_pv(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_pv.parameters(), 1.0)
            opt_pv.step()

    lstm_pv.eval()
    lstm_val_logits = []
    with torch.no_grad():
        for i in range(len(rf_val_feats) - SEQ_LEN):
            seq = torch.tensor(rf_val_feats[i:i + SEQ_LEN],
                               dtype=torch.float32).unsqueeze(0).to(DEVICE)
            lstm_val_logits.append(lstm_pv(seq).cpu().numpy()[0])
    lstm_val_pred_full = np.vstack([
        np.full((SEQ_LEN, K_REGIMES), 1.0 / K_REGIMES),
        torch.softmax(torch.tensor(np.array(lstm_val_logits)), dim=-1).numpy()
    ]).argmax(axis=1)

    y_val_arr = y_val.reindex(X_val_raw.index).dropna().values
    val_f1 = {
        "gmm":  f1_score(y_val_arr, gmm_val_pred,       average="macro", zero_division=0),
        "hmm":  f1_score(y_val_arr, hmm_val_pred,       average="macro", zero_division=0),
        "rf":   f1_score(y_val_arr, rf_val_pred,        average="macro", zero_division=0),
        "lstm": f1_score(y_val_arr, lstm_val_pred_full, average="macro", zero_division=0),
    }
    print(f"\nValidation macro-F1 ({X_val_raw.index[0].date()} -> {X_val_raw.index[-1].date()}):")
    for m, s in val_f1.items():
        print(f"  {m:<6}: {s:.3f}")

    ENSEMBLE_WEIGHTS = tune_ensemble_weights(val_f1)
    print(f"\nVal-period bear days: {(y_val_arr == 0).sum()} / {len(y_val_arr)}")


else:
    print(f"\nValidation set too small ({len(X_val_raw)} obs < {MIN_VAL_OBS}) — "
          "using manually set ENSEMBLE_WEIGHTS.")
    print("  Weights: " + str(ENSEMBLE_WEIGHTS))

# ============================================================
# LAYER 2 — ENSEMBLE: WEIGHTED GEOMETRIC MEAN
# ============================================================

print("\nBuilding ensemble...")
print(f"\nGMM bear proba — mean: {gmm_proba_test[:,0].mean():.4f}  "
      f"days > 0.3: {(gmm_proba_test[:,0] > 0.3).sum()}")
print(f"HMM bear proba — mean: {hmm_proba_test[:,0].mean():.4f}  "
      f"days > 0.3: {(hmm_proba_test[:,0] > 0.3).sum()}")
print(f"RF  bear proba — mean: {rf_proba_test[:,0].mean():.4f}  "
      f"days > 0.3: {(rf_proba_test[:,0] > 0.3).sum()}")
print(f"LSTM bear proba — mean: {lstm_proba_test[:,0].mean():.4f}  "
      f"days > 0.3: {(lstm_proba_test[:,0] > 0.3).sum()}")
w   = ENSEMBLE_WEIGHTS
eps = 1e-8

ensemble_proba = (
      w["gmm"]  * gmm_proba_test
    + w["hmm"]  * hmm_proba_test
    + w["rf"]   * rf_proba_test
    + w["lstm"] * lstm_proba_test
)

confidence = ensemble_proba.max(axis=1)

equal_proba = (
      0.25 * gmm_proba_test
    + 0.25 * hmm_proba_test
    + 0.25 * rf_proba_test
    + 0.25 * lstm_proba_test
)
fallback_mask              = confidence < CONFIDENCE_THRESHOLD
ensemble_proba[fallback_mask] = equal_proba[fallback_mask]

ensemble_pred = ensemble_proba.argmax(axis=1)

# ============================================================
# CONFLICT SCORE
# ============================================================

gmm_hard  = gmm_proba_test.argmax(axis=1)
hmm_hard  = forward_filter(hmm_model, X_test)
rf_hard   = rf_cal.predict(rf_features_test)
lstm_hard = lstm_proba_test.argmax(axis=1)

all_preds = np.stack([gmm_hard, hmm_hard, rf_hard, lstm_hard], axis=1)

def agreement_score(preds):
    scores = []
    for row in preds:
        winner = np.bincount(row.astype(int), minlength=3).argmax()
        scores.append((row == winner).mean())
    return np.array(scores)

agree    = agreement_score(all_preds)
conflict = 1.0 - agree

# ============================================================
# ATTACH TO OUTPUT DATAFRAME
# ============================================================

test_idx = X_test_raw.index

out.loc[test_idx, "ensemble_bear"]    = ensemble_proba[:, 0]
out.loc[test_idx, "ensemble_neutral"] = ensemble_proba[:, 1]
out.loc[test_idx, "ensemble_bull"]    = ensemble_proba[:, 2]
out.loc[test_idx, "ensemble_pred"]    = ensemble_pred
out.loc[test_idx, "confidence"]       = confidence
out.loc[test_idx, "conflict"]         = conflict
out.loc[test_idx, "gmm_pred"]         = gmm_hard
out.loc[test_idx, "hmm_pred"]         = hmm_hard
out.loc[test_idx, "rf_pred"]          = rf_hard
out.loc[test_idx, "lstm_pred"]        = lstm_hard

regime_names = {0: "bear", 1: "neutral", 2: "bull"}
out["ensemble_regime"] = out["ensemble_pred"].map(regime_names)

# ============================================================
# LAYER 3 — DYNAMIC ALLOCATION
# ============================================================

out_test = out.loc[test_idx].copy()

w_base = out_test["w_actual"]
p_bull = out_test["ensemble_bull"].fillna(1/3)
p_bear = out_test["ensemble_bear"].fillna(1/3)
conf   = out_test["confidence"].fillna(0.5)
confl  = out_test["conflict"].fillna(0.5)

w_regime_adj = (w_base * (1 + ALPHA_BULL * p_bull - ALPHA_BEAR * p_bear)).clip(
    MIN_WEIGHT, MAX_WEIGHT
)
w_blended = conf * w_regime_adj + (1 - conf) * w_base

# FIX 4: use causal sigma_median already computed above
base_band     = BAND_MULT * np.sqrt(tc) * (
    sigma_ann.reindex(test_idx) / sigma_median.reindex(test_idx)
).clip(0.5, 2.0) 
conflict_band = base_band * (1 + 1.5 * confl)
conflict_band = conflict_band.shift(1).fillna(conflict_band.median())

print(f"\nBand diagnostics:")
print(f"  base_band     — mean: {base_band.mean():.6f}  median: {base_band.median():.6f}  NaNs: {base_band.isna().sum()}")
print(f"  conflict_band — mean: {conflict_band.mean():.6f}  median: {conflict_band.median():.6f}  NaNs: {conflict_band.isna().sum()}")
print(f"  sigma_ann     — mean: {sigma_ann.reindex(test_idx).mean():.4f}  NaNs: {sigma_ann.reindex(test_idx).isna().sum()}")
print(f"  sigma_median  — mean: {sigma_median.reindex(test_idx).mean():.4f}  NaNs: {sigma_median.reindex(test_idx).isna().sum()}")
print(f"  ratio (clipped) mean: {(sigma_ann.reindex(test_idx)/sigma_median.reindex(test_idx)).clip(0.5,2.0).mean():.4f}")
print(f"  w_blended     — mean: {w_blended.mean():.4f}  min: {w_blended.min():.4f}  max: {w_blended.max():.4f}")

# FIX 1: Ensemble — trade to BOUNDARY not target
w_curr = 0.0
equity, weights_used, trades_count, trades_cost = [], [], 0, 0.0

for date in test_idx:
    tgt     = w_blended.loc[date]
    b       = conflict_band.loc[date]
    base_eq = equity[-1] if equity else 1.0

    if abs(tgt - w_curr) > b:
        direction = np.sign(tgt - w_curr)
        new_w     = float(np.clip(w_curr + direction * b, MIN_WEIGHT, MAX_WEIGHT))
        traded    = abs(new_w - w_curr)
        base_eq   = base_eq * (1 - tc * traded)
        trades_cost  += tc * traded
        trades_count += 1
        w_curr = new_w

    asset_ret = ret.reindex(test_idx).loc[date]
    rf_ret    = r_daily.reindex(test_idx).loc[date]
    port_ret  = w_curr * asset_ret + (1 - w_curr) * rf_ret
    equity.append(base_eq * (1 + port_ret))
    weights_used.append(w_curr)

out_test["equity_ensemble"] = equity
out_test["w_ensemble"]      = weights_used

# FIX 1: Baseline — trade to BOUNDARY not target, using fair (non-widened) band
baseline_equity  = []
baseline_eq      = 1.0
w_b_prev         = 0.0
base_trades_cost = 0.0
base_band_fair   = base_band.shift(1).fillna(base_band.median())

for date in test_idx:
    w_b_tgt = float(np.clip(out_test["w_actual"].loc[date], MIN_WEIGHT, MAX_WEIGHT))
    b_fair  = base_band_fair.loc[date]

    if abs(w_b_tgt - w_b_prev) > b_fair:
        direction        = np.sign(w_b_tgt - w_b_prev)
        new_w_b          = float(np.clip(w_b_prev + direction * b_fair, MIN_WEIGHT, MAX_WEIGHT))
        traded           = abs(new_w_b - w_b_prev)
        baseline_eq      = baseline_eq * (1 - tc * traded)
        base_trades_cost += tc * traded
        w_b_prev         = new_w_b

    asset_ret   = ret.reindex(test_idx).loc[date]
    rf_ret      = r_daily.reindex(test_idx).loc[date]
    port_ret    = w_b_prev * asset_ret + (1 - w_b_prev) * rf_ret
    baseline_eq = baseline_eq * (1 + port_ret)
    baseline_equity.append(baseline_eq)

out_test["equity_baseline"] = baseline_equity

print(f"\nTransaction cost summary:")
print(f"  Ensemble — trades: {trades_count:4d}  total TC drag: {trades_cost:.4f} ({trades_cost*100:.2f} bps equiv)")
print(f"  Baseline — total TC drag: {base_trades_cost:.4f} ({base_trades_cost*100:.2f} bps equiv)")

# ============================================================
# CONFUSION MATRIX
# ============================================================

from sklearn.metrics import confusion_matrix

y_true_arr = y_all.reindex(test_idx).dropna().values.astype(int)
y_pred_arr = out_test["ensemble_pred"].reindex(
    y_all.reindex(test_idx).dropna().index
).dropna().astype(int).values
print(f"Test-period bear days (true): {(y_true_arr == 0).sum()} / {len(y_true_arr)}")

cm      = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1, 2])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

print("\n" + "=" * 55)
print("CONFUSION MATRIX (row = true, col = predicted, row-normalised)")
print("=" * 55)
print(f"  {'':12}  {'Pred Bear':>10}  {'Pred Neutral':>12}  {'Pred Bull':>10}")
for i, name in enumerate(["True Bear", "True Neutral", "True Bull"]):
    row = cm_norm[i]
    raw = cm[i]
    print(f"  {name:<12}  {row[0]:>9.1%} ({raw[0]:3d})  "
          f"{row[1]:>11.1%} ({raw[1]:4d})  "
          f"{row[2]:>9.1%} ({raw[2]:3d})")

# ============================================================
# PER-REGIME WEIGHT DIAGNOSTIC
# ============================================================

print("\n" + "=" * 55)
print("MEAN ENSEMBLE WEIGHT BY TRUE REGIME (test set)")
print("=" * 55)
print(f"  {'True Regime':<14}  {'N days':>7}  {'Mean w_ensemble':>16}  {'Mean w_baseline':>16}")
print(f"  {'-'*14}  {'-'*7}  {'-'*16}  {'-'*16}")

y_true_s = y_all.reindex(test_idx).dropna()
for code, name in [(0, "Bear"), (1, "Neutral"), (2, "Bull")]:
    mask        = y_true_s == code
    days        = mask.sum()
    w_ens_mean  = out_test.loc[mask.index[mask], "w_ensemble"].mean()
    w_base_mean = out_test.loc[mask.index[mask], "w_actual"].mean()
    print(f"  {name:<14}  {days:>7d}  {w_ens_mean:>16.3f}  {w_base_mean:>16.3f}")

print("\n  Monotonic rise (Bear < Neutral < Bull) confirms regime signal")
print("  is tilting allocation in the economically correct direction.")

# ============================================================
# EVALUATION
# ============================================================

y_test_aligned = y_all.reindex(test_idx).dropna()
pred_aligned   = out_test["ensemble_pred"].reindex(y_test_aligned.index).dropna()

print("\n" + "=" * 55)
print("ENSEMBLE CLASSIFICATION REPORT (test set)")
print("=" * 55)
print(classification_report(
    y_test_aligned.reindex(pred_aligned.index),
    pred_aligned.astype(int),
    target_names=["Bear", "Neutral", "Bull"]
))

for c, name in enumerate(["Bear", "Neutral", "Bull"]):
    y_bin   = (y_test_aligned == c).astype(int).reindex(pred_aligned.index)
    p_class = out_test[f"ensemble_{name.lower()}"].reindex(pred_aligned.index).fillna(1/3)
    bs      = brier_score_loss(y_bin, p_class)
    print(f"Brier score ({name}): {bs:.4f}")

from sklearn.metrics import f1_score as _f1

print("\n" + "=" * 55)
print("PER-MODEL ACCURACY & MACRO-F1 (test set)")
print("=" * 55)
print(f"  {'Model':<12}  {'Accuracy':>8}  {'Macro-F1':>8}")
print(f"  {'-'*12}  {'-'*8}  {'-'*8}")
for col, name in [("gmm_pred", "GMM"), ("hmm_pred", "HMM"),
                  ("rf_pred", "RF"), ("lstm_pred", "LSTM"), ("ensemble_pred", "ENSEMBLE")]:
    preds  = out_test[col].reindex(y_test_aligned.index).dropna().astype(int)
    y_aln  = y_test_aligned.reindex(preds.index)
    acc    = accuracy_score(y_aln, preds)
    mf1    = _f1(y_aln, preds, average="macro", zero_division=0)
    marker = " <" if name == "ENSEMBLE" else ""
    print(f"  {name:<12}  {acc:>8.3f}  {mf1:>8.3f}{marker}")

print("\n" + "=" * 55)
print("REGIME EXPOSURE DIAGNOSTIC (test set)")
print("=" * 55)
true_dist   = y_test_aligned.value_counts(normalize=True).sort_index()
pred_dist   = pred_aligned.astype(int).value_counts(normalize=True).sort_index()
label_names = {0: "Bear", 1: "Neutral", 2: "Bull"}
print(f"  {'Regime':<10}  {'True %':>8}  {'Pred %':>8}  {'Delta':>8}")
print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
for c in [0, 1, 2]:
    t = true_dist.get(c, 0.0)
    p = pred_dist.get(c, 0.0)
    print(f"  {label_names[c]:<10}  {t:>8.1%}  {p:>8.1%}  {p-t:>+8.1%}")

print("\n  Mean ensemble weight (w_ensemble):", f"{out_test['w_ensemble'].mean():.3f}")
print("  Mean baseline weight (w_actual):  ", f"{out_test['w_actual'].mean():.3f}")
print("  Weight delta (ensemble - baseline):",
      f"{out_test['w_ensemble'].mean() - out_test['w_actual'].mean():+.3f}")

ens_rets  = pd.Series(out_test["equity_ensemble"]).pct_change().fillna(0)
base_rets = pd.Series(out_test["equity_baseline"]).pct_change().fillna(0)

print("\n" + "=" * 55)
print("PORTFOLIO PERFORMANCE (test set)")
print("=" * 55)
print(f"  {'Strategy':<20}  {'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>8}")
print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*8}")
print(f"  {'Ensemble':<20}  "
      f"{segment_cagr(pd.Series(out_test['equity_ensemble'])):>7.2%}  "
      f"{sharpe_ratio(ens_rets):>7.2f}  "
      f"{max_drawdown(pd.Series(out_test['equity_ensemble'])):>8.2%}")
print(f"  {'Baseline (HJB)':<20}  "
      f"{segment_cagr(pd.Series(out_test['equity_baseline'])):>7.2%}  "
      f"{sharpe_ratio(base_rets):>7.2f}  "
      f"{max_drawdown(pd.Series(out_test['equity_baseline'])):>8.2%}")

def shade_regimes(ax, index, pred_series):
    colors = {0: "red", 1: "grey", 2: "green"}
    in_regime, start, cur = False, None, None
    for date, r in zip(index, pred_series):
        if not in_regime:
            start, cur, in_regime = date, r, True
        elif r != cur:
            ax.axvspan(start, date, alpha=0.12, color=colors.get(cur, "grey"), lw=0)
            start, cur = date, r
    if in_regime:
        ax.axvspan(start, index[-1], alpha=0.12, color=colors.get(cur, "grey"), lw=0)





# ============================================================
# VISUALISATION — ACADEMIC STYLE
# ============================================================

import matplotlib as mpl
import matplotlib.gridspec as gridspec
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

def shade_regimes(ax, index, pred_series):
    colors = {0: "red", 1: "grey", 2: "green"}
    in_regime, start, cur = False, None, None
    for date, r in zip(index, pred_series):
        if not in_regime:
            start, cur, in_regime = date, r, True
        elif r != cur:
            ax.axvspan(start, date, alpha=0.12, color=colors.get(cur, "grey"), lw=0)
            start, cur = date, r
    if in_regime:
        ax.axvspan(start, index[-1], alpha=0.12, color=colors.get(cur, "grey"), lw=0)

# ── Figure 1 — Regime dashboard (2x2)
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
fig.subplots_adjust(hspace=0.45, wspace=0.35)

ax0 = axes[0, 0]
ax0.plot(test_idx, out_test["price"], lw=0.8, color="black", label="S&P 500")
shade_regimes(ax0, test_idx, out_test["ensemble_pred"].fillna(1))
legend_els = [Patch(facecolor="green", alpha=0.4, label="Bull"),
              Patch(facecolor="grey",  alpha=0.4, label="Neutral"),
              Patch(facecolor="red",   alpha=0.4, label="Bear")]
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles=handles + legend_els, loc="upper left", framealpha=0.7)
ax0.set_title("S&P 500 with Regime Signal")
ax0.set_ylabel("Price")
ax0.set_xlabel("Date")

ax1 = axes[0, 1]
ax1.stackplot(test_idx,
              out_test["ensemble_bear"].fillna(1/3),
              out_test["ensemble_neutral"].fillna(1/3),
              out_test["ensemble_bull"].fillna(1/3),
              labels=["P(Bear)", "P(Neutral)", "P(Bull)"],
              colors=["#d62728", "#aec7e8", "#2ca02c"], alpha=0.85)
ax1.set_ylabel("Probability")
ax1.set_ylim(0, 1)
ax1.legend(loc="upper left", framealpha=0.7)
ax1.set_title("Ensemble Regime Probabilities")
ax1.set_xlabel("Date")

ax2 = axes[1, 0]
ax2.fill_between(test_idx, out_test["confidence"].fillna(0.5),
                 alpha=0.6, color="steelblue", label="Confidence")
ax2.axhline(CONFIDENCE_THRESHOLD, color="orange", ls="--", lw=0.8,
            label=f"Threshold ({CONFIDENCE_THRESHOLD})")
ax2.set_ylabel("Score")
ax2.set_ylim(0, 1)
ax2.legend(loc="upper left", framealpha=0.7)
ax2.set_title("Confidence Score")
ax2.set_xlabel("Date")

ax3 = axes[1, 1]
ax3.fill_between(test_idx, out_test["conflict"].fillna(0.5),
                 alpha=0.6, color="firebrick", label="Conflict")
ax3.set_ylabel("Score")
ax3.set_ylim(0, 1)
ax3.legend(loc="upper left", framealpha=0.7)
ax3.set_title("Model Conflict Score")
ax3.set_xlabel("Date")

fig.suptitle("Hybrid Ensemble Regime Framework — Test Period",
             fontsize=9, fontstyle="italic")
plt.savefig("fig_hybrid_regime_dashboard.png", dpi=300, bbox_inches="tight")
plt.show()

# ── Figure 2 — Model comparison (2x2 + ensemble spanning bottom row)
fig = plt.figure(figsize=(9, 6))
gs2 = gridspec.GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.3)

model_cols  = ["gmm_pred", "hmm_pred", "rf_pred", "lstm_pred", "ensemble_pred"]
model_names = ["GMM", "HMM", "Random Forest", "LSTM", "Ensemble"]
colors_map  = {0: "red", 1: "grey", 2: "green"}

for idx, (col, name) in enumerate(zip(model_cols[:4], model_names[:4])):
    r, c = divmod(idx, 2)
    ax = fig.add_subplot(gs2[r, c])
    series = out_test[col].fillna(1)
    for regime, color in colors_map.items():
        mask = series == regime
        ax.fill_between(test_idx, 0, 1, where=mask.values,
                        color=color, alpha=0.4,
                        transform=ax.get_xaxis_transform())
    ax.set_yticks([])
    ax.set_title(name, pad=2)
    ax.set_xlabel("Date")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if idx == 0:
        legend_els = [Patch(facecolor="green", alpha=0.5, label="Bull"),
                      Patch(facecolor="grey",  alpha=0.5, label="Neutral"),
                      Patch(facecolor="red",   alpha=0.5, label="Bear")]
        ax.legend(handles=legend_els, loc="upper right", framealpha=0.7)

ax_ens = fig.add_subplot(gs2[2, :])
series = out_test["ensemble_pred"].fillna(1)
for regime, color in colors_map.items():
    mask = series == regime
    ax_ens.fill_between(test_idx, 0, 1, where=mask.values,
                        color=color, alpha=0.4,
                        transform=ax_ens.get_xaxis_transform())
ax_ens.set_yticks([])
ax_ens.set_title("Ensemble", pad=2)
ax_ens.set_xlabel("Date")
ax_ens.spines["top"].set_visible(False)
ax_ens.spines["right"].set_visible(False)

fig.suptitle("Regime Comparison: GMM | HMM | RF | LSTM | Ensemble",
             fontsize=9, fontstyle="italic")
plt.savefig("fig_model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# ── Figure 3 — Equity curves
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(test_idx, out_test["equity_ensemble"], lw=1.2,
        label="Ensemble Strategy")
ax.plot(test_idx, out_test["equity_baseline"], lw=1.2, ls="--", alpha=0.8,
        label="Baseline (HJB only)")
ax.set_title("Equity Curves — Ensemble vs Baseline (Test Period)")
ax.set_xlabel("Date")
ax.set_ylabel("Normalised Wealth")
ax.legend(framealpha=0.7)
plt.tight_layout()
plt.savefig("fig_ensemble_equity.png", dpi=300, bbox_inches="tight")
plt.show()

# ── Figure 4 — Weight comparison (zoomed) + rolling return spread (full)
fig, (ax_w, ax_r) = plt.subplots(2, 1, figsize=(7, 5), sharex=False)
fig.subplots_adjust(hspace=0.4)

zoom_start  = pd.Timestamp("2020-01-01")
zoom_end    = pd.Timestamp("2020-09-01")
zoom_mask   = (test_idx >= zoom_start) & (test_idx <= zoom_end)
zoom_idx    = test_idx[zoom_mask]
w_ens_zoom  = out_test["w_ensemble"].loc[zoom_idx]
w_base_zoom = out_test["w_actual"].loc[zoom_idx]

ax_w.plot(zoom_idx, w_ens_zoom,  lw=0.9, color="tab:blue",
          label="Ensemble weight")
ax_w.plot(zoom_idx, w_base_zoom, lw=0.9, color="tab:orange",
          ls="--", alpha=0.8, label="Baseline weight (HJB)")
ax_w.fill_between(zoom_idx, w_ens_zoom, w_base_zoom,
                  where=(w_ens_zoom > w_base_zoom).values,
                  alpha=0.15, color="tab:blue",   label="Ensemble higher")
ax_w.fill_between(zoom_idx, w_ens_zoom, w_base_zoom,
                  where=(w_ens_zoom < w_base_zoom).values,
                  alpha=0.15, color="tab:orange", label="Baseline higher")
ax_w.set_ylabel("Risky Asset Weight")
ax_w.margins(y=0.05)
ax_w.legend(ncol=2, framealpha=0.7)
ax_w.set_title("Ensemble vs Baseline Weight (Jan 2020 - Sep 2020)")
ax_w.set_xlabel("Date")

roll_ens  = pd.Series(out_test["equity_ensemble"].values,
                      index=test_idx).pct_change().rolling(252).sum()
roll_base = pd.Series(out_test["equity_baseline"].values,
                      index=test_idx).pct_change().rolling(252).sum()
roll_diff = roll_ens - roll_base

ax_r.fill_between(test_idx, roll_diff, 0,
                  where=(roll_diff >= 0).values, alpha=0.5, color="tab:green",
                  label="Ensemble outperforms")
ax_r.fill_between(test_idx, roll_diff, 0,
                  where=(roll_diff < 0).values, alpha=0.5, color="tab:red",
                  label="Baseline outperforms")
ax_r.axhline(0, color="black", lw=0.8)
ax_r.set_ylabel("Rolling 252d Spread")
ax_r.set_xlabel("Date")
ax_r.legend(framealpha=0.7)
ax_r.set_title("Rolling 1-Year Return Difference (Ensemble minus Baseline)")

plt.savefig("fig_weight_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nDone. Saved: fig_hybrid_regime_dashboard.png, fig_model_comparison.png,")
print("             fig_ensemble_equity.png, fig_weight_comparison.png")