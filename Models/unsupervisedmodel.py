import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "sp500_features.csv")

TICKER_COL_CANDIDATES = [
    "Close", "Adj Close",
    "('Close','^GSPC')", "('Adj Close','^GSPC')",
    "Close_^GSPC", "Adj Close_^GSPC"
]

RISK_AVERSION_GAMMA  = 4.0
ANNUALIZATION        = 252
TC_BPS               = 5
BAND_MULT            = 1.5
MIN_WEIGHT           = 0.0
MAX_WEIGHT           = 1.0
ALPHA_SHORT          = 0.5
LOOKBACK_ST          = 20
LOOKBACK_LT          = 126
BAND_ROLLING_WINDOW  = 252
VIX_WEIGHT           = 0.5
MACRO_WEIGHT         = 0.01
TRAIN_SPLIT_DATE     = "2015-01-01"
WARMUP               = LOOKBACK_LT * 2
MACRO_LAGS           = {"CPI": 21, "Unemployment": 7, "FedFunds": 1}


def pick_price_column(df):
    for c in TICKER_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if "close" in str(c).lower():
            return c
    raise KeyError("No price column found")

def to_daily_rate(fed_funds):
    r_ann = fed_funds.astype(float) / (100.0 if fed_funds.max() > 1 else 1.0)
    return r_ann / ANNUALIZATION

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

def plot_regimes(ax, index, regime_series, regime_label, color):
    in_regime = False
    start     = None
    for date, reg in zip(index, regime_series):
        if reg == regime_label and not in_regime:
            start     = date
            in_regime = True
        elif reg != regime_label and in_regime:
            ax.axvspan(start, date, alpha=0.15, color=color, lw=0)
            in_regime = False
    if in_regime:
        ax.axvspan(start, index[-1], alpha=0.15, color=color, lw=0)


# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(CSV_PATH, parse_dates=[0], index_col=0).sort_index()
df.columns = [c.strip() for c in df.columns]

price_col = pick_price_column(df)
px        = df[price_col].astype(float)
logret    = np.log(px).diff()
ret       = px.pct_change()

# ===============================
# MACRO PUBLICATION LAG
# ===============================

macro_cols = [c for c in ["CPI", "Unemployment", "FedFunds"] if c in df.columns]
for col in macro_cols:
    df[col] = df[col].shift(MACRO_LAGS.get(col, 1))
df[macro_cols] = df[macro_cols].ffill()
print(f"Macro publication lags applied: { {c: MACRO_LAGS[c] for c in macro_cols} }")

if "FedFunds" in df.columns:
    r_daily = to_daily_rate(df["FedFunds"]).reindex(df.index).ffill().fillna(0)
else:
    r_daily = pd.Series(0.0, index=df.index)

# ===============================
# CAUSAL FEATURE ENGINEERING
# ===============================

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

# ===============================
# HJB OPTIMAL WEIGHT
# ===============================

excess_mu = mu_ann - r_daily * ANNUALIZATION
u_star    = (excess_mu / (RISK_AVERSION_GAMMA * sigma_ann ** 2)).clip(MIN_WEIGHT, MAX_WEIGHT)
u_star.iloc[:WARMUP] = 0.0
u_star = u_star.shift(1).fillna(0.0)

# ===============================
# NO-TRADE BAND
# ===============================

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

# ===============================
# BACKTEST
# ===============================

dates      = df.index
w          = 0.0
equity_val = 1.0
weights    = []
targets    = []
eq_curve   = []

for t in range(1, len(dates)):
    date = dates[t]
    tgt  = u_star.loc[date]
    b    = band.loc[date]
    diff = tgt - w

    if abs(diff) > b:
        direction  = np.sign(diff)
        new_w      = np.clip(w + direction * b, MIN_WEIGHT, MAX_WEIGHT)
        traded     = abs(new_w - w)
        equity_val *= (1 - tc * traded)
        w          = new_w

    port_ret   = w * ret.loc[date] + (1 - w) * r_daily.loc[date]
    equity_val *= (1 + port_ret)

    weights.append(w)
    targets.append(tgt)
    eq_curve.append(equity_val)

equity         = pd.Series(eq_curve,  index=dates[1:])
weights_series = pd.Series(weights,   index=equity.index)
targets_series = pd.Series(targets,   index=equity.index)

out = pd.DataFrame({
    "price":    px.reindex(equity.index),
    "equity":   equity,
    "w_actual": weights_series,
    "w_target": targets_series,
    "ret":      ret.reindex(equity.index),
    "r_daily":  r_daily.reindex(equity.index),
})

# ===============================
# FEATURE MATRIX
# ===============================

feature_dict = {
    "ret":   logret,
    "vol":   sigma_ann,
    "mu":    mu_ann,
    "macro": macro_signal,
}
if "VIX" in df.columns:
    feature_dict["vix"] = df["VIX"].astype(float)

regime_df = pd.DataFrame(feature_dict).dropna()

# ===============================
# TRAIN / TEST SPLIT
# ===============================

regime_train = regime_df.loc[:TRAIN_SPLIT_DATE].copy()
regime_test  = regime_df.loc[TRAIN_SPLIT_DATE:].copy()

print(f"\nUnsupervised train: {regime_train.index[0].date()} -> "
      f"{regime_train.index[-1].date()} ({len(regime_train)} obs)")
print(f"Unsupervised test:  {regime_test.index[0].date()} -> "
      f"{regime_test.index[-1].date()} ({len(regime_test)} obs)")

# ===============================
# SCALER
# ===============================

scaler  = StandardScaler()
X_train = scaler.fit_transform(regime_train)
X_test  = scaler.transform(regime_test)

# ===============================
# GMM
# ===============================

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_train)

gmm_labels_train = gmm.predict(X_train)
gmm_proba_train  = gmm.predict_proba(X_train)
gmm_proba_test   = gmm.predict_proba(X_test)

mean_ret_gmm  = {
    s: regime_train["ret"].values[gmm_labels_train == s].mean()
    for s in range(3)
}
gmm_col_order = sorted(mean_ret_gmm, key=mean_ret_gmm.get)
gmm_state_map = {gmm_col_order[i]: i for i in range(3)}

regime_train["gmm_state"] = np.array([gmm_state_map[s] for s in gmm_labels_train])
regime_test["gmm_state"]  = np.array(
    [gmm_state_map[s] for s in gmm.predict(X_test)]
)

# ===============================
# HMM
# ===============================

hmm_model = GaussianHMM(
    n_components=3, covariance_type="full", n_iter=500, random_state=42
)
hmm_model.fit(X_train)

hmm_labels_train = hmm_model.predict(X_train)
hmm_labels_test  = forward_filter(hmm_model, X_test)

mean_ret_hmm  = {
    s: regime_train["ret"].values[hmm_labels_train == s].mean()
    for s in range(3)
}
hmm_col_order = sorted(mean_ret_hmm, key=mean_ret_hmm.get)
hmm_state_map = {hmm_col_order[i]: i for i in range(3)}

regime_train["hmm_state"] = np.array([hmm_state_map[s] for s in hmm_labels_train])
regime_test["hmm_state"]  = np.array([hmm_state_map[s] for s in hmm_labels_test])

int_to_regime = {0: "bear", 1: "neutral", 2: "bull"}
regime_train["hmm_regime"] = regime_train["hmm_state"].map(int_to_regime)
regime_test["hmm_regime"]  = regime_test["hmm_state"].map(int_to_regime)

# ===============================
# RECOMBINE
# ===============================

regime_df = pd.concat([regime_train, regime_test])
out       = out.join(
    regime_df[["gmm_state", "hmm_state", "hmm_regime"]], how="left"
)

# ===============================
# REGIME DAY COUNTS + CONDITIONAL CAGR
# — must be after out.join so hmm_regime exists
# ===============================

print("\nHMM Regime Day Counts (test set):")
print(out.loc[TRAIN_SPLIT_DATE:]["hmm_regime"].value_counts())

print("\nCAGR by HMM Regime (test only):")
for regime in ["bear", "neutral", "bull"]:
    subset = out.loc[TRAIN_SPLIT_DATE:]
    subset = subset[subset["hmm_regime"] == regime]
    cagr   = segment_cagr(subset["equity"])
    n      = len(subset)
    print(f"  {regime:<10} N={n:>5d}  CAGR={cagr:.2%}")

# ===============================
# HMM TRANSITION MATRIX
# ===============================

reordered_transmat = hmm_model.transmat_[np.ix_(hmm_col_order, hmm_col_order)]

transition_matrix = pd.DataFrame(
    reordered_transmat,
    columns=["Bear", "Neutral", "Bull"],
    index=["Bear", "Neutral", "Bull"]
)
print("\nHMM Transition Matrix (aligned to bear/neutral/bull):")
print(transition_matrix.round(3))

# ===============================
# DIAGNOSTICS
# ===============================

print(f"\nCorrelation of u_star with same-day ret:  {u_star.corr(ret):.4f}")
print(f"Correlation of u_star with next-day ret:  {u_star.corr(ret.shift(-1)):.4f}")

rets_port = equity.pct_change().fillna(0)
print(f"\nAnnualised port return: {rets_port.mean() * ANNUALIZATION:.4f}")
print(f"Annualised S&P return:  {ret.reindex(equity.index).mean() * ANNUALIZATION:.4f}")

# ===============================
# PERFORMANCE METRICS
# ===============================

rets_out = out["equity"].pct_change().fillna(0)

print("\nOverall Performance:")
print(f"  CAGR:   {segment_cagr(out['equity']):.2%}")
print(f"  Sharpe: {sharpe_ratio(rets_out):.2f}")
print(f"  MaxDD:  {max_drawdown(out['equity']):.2%}")

for label, subset in [
    ("Train (in-sample)",     out.loc[:TRAIN_SPLIT_DATE]),
    ("Test  (out-of-sample)", out.loc[TRAIN_SPLIT_DATE:])
]:
    r = subset["equity"].pct_change().fillna(0)
    print(f"\n{label}:")
    print(f"  CAGR:   {segment_cagr(subset['equity']):.2%}")
    print(f"  Sharpe: {sharpe_ratio(r):.2f}")
    print(f"  MaxDD:  {max_drawdown(subset['equity']):.2%}")

# ===============================
# VISUALISATION — ACADEMIC STYLE
# ===============================

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

# Figure 1 — S&P 500 with HMM regimes
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(out.index, out["price"], lw=0.8, color="black", label="S\\&P 500")

plot_regimes(ax, out.index, out["hmm_regime"], "bull",    color="green")
plot_regimes(ax, out.index, out["hmm_regime"], "bear",    color="red")
plot_regimes(ax, out.index, out["hmm_regime"], "neutral", color="grey")

ax.axvline(pd.Timestamp(TRAIN_SPLIT_DATE), color="black", ls="--",
           lw=0.8, label="Train/Test Split")

legend_elements = [
    Patch(facecolor="green", alpha=0.3, label="Bull"),
    Patch(facecolor="red",   alpha=0.3, label="Bear"),
    Patch(facecolor="grey",  alpha=0.3, label="Neutral"),
]
handles, labels_leg = ax.get_legend_handles_labels()
ax.legend(handles=handles + legend_elements, framealpha=0.7, ncol=3)
ax.set_title("S\\&P 500 with HMM-Inferred Regimes")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
plt.tight_layout()
plt.savefig("fig_hmm_regimes.png", dpi=300, bbox_inches="tight")
plt.show()

# Figure 2 — Transition matrix heatmap
fig, ax = plt.subplots(figsize=(4, 3.5))
sns.heatmap(
    reordered_transmat, annot=True, fmt=".3f", cmap="Blues",
    xticklabels=["Bear", "Neutral", "Bull"],
    yticklabels=["Bear", "Neutral", "Bull"],
    ax=ax, cbar_kws={"shrink": 0.8},
    linewidths=0.5, linecolor="white",
    annot_kws={"size": 8}
)
ax.set_title("HMM Transition Matrix")
ax.set_ylabel("Current State")
ax.set_xlabel("Next State")
plt.tight_layout()
plt.savefig("fig_hmm_transition.png", dpi=300, bbox_inches="tight")
plt.show()