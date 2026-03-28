import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SP500_CSV = os.path.join(BASE_DIR, "sp500_features.csv")

TICKER_COL_CANDIDATES = [
    "Close", "Adj Close",
    "('Close','^GSPC')", "('Adj Close','^GSPC')",
    "Close_^GSPC", "Adj Close_^GSPC",
]

RISK_AVERSION_GAMMA = 4.0
ANNUALIZATION       = 252
TC_BPS              = 5
BAND_MULT           = 1.5
MIN_WEIGHT          = 0.0
MAX_WEIGHT          = 1.0
ALPHA_SHORT         = 0.5
LOOKBACK_ST         = 20
LOOKBACK_LT         = 126
BAND_ROLLING_WINDOW = 252
WARMUP              = LOOKBACK_LT * 2
TRAIN_SPLIT_DATE    = "2021-01-01"
MACRO_LAGS          = {"CPI": 21, "Unemployment": 7, "FedFunds": 1}

# ============================================================
# ACADEMIC STYLE
# ============================================================

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

# ============================================================
# HELPERS
# ============================================================

def pick_price_column(df):
    for c in TICKER_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if "close" in str(c).lower():
            return c
    raise KeyError("No price column found. Columns: " + str(df.columns.tolist()))

def to_daily_rate(fed_funds):
    r_ann = fed_funds.astype(float) / (100.0 if fed_funds.max() > 1 else 1.0)
    return r_ann / ANNUALIZATION

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

# ============================================================
# LOAD & BACKTEST
# ============================================================

print("Loading S&P 500...")
df = pd.read_csv(SP500_CSV, parse_dates=[0], index_col=0).sort_index()
df.columns = [c.strip() for c in df.columns]

macro_cols = [c for c in ["CPI", "Unemployment", "FedFunds"] if c in df.columns]
for col in macro_cols:
    df[col] = df[col].shift(MACRO_LAGS.get(col, 1))
df[macro_cols] = df[macro_cols].ffill()
print(f"Macro publication lags applied: { {c: MACRO_LAGS[c] for c in macro_cols} }")

price_col = pick_price_column(df)
print(f"Using price column: {price_col}")

px     = df[price_col].astype(float)
logret = np.log(px).diff()
ret    = px.pct_change()

if "FedFunds" in df.columns:
    r_daily = to_daily_rate(df["FedFunds"]).reindex(df.index).ffill().fillna(0)
else:
    r_daily = pd.Series(0.0, index=df.index)

mu_st    = causal_ewm_mean(logret, LOOKBACK_ST) * ANNUALIZATION
mu_lt    = causal_ewm_mean(logret, LOOKBACK_LT) * ANNUALIZATION
sigma_lt = causal_ewm_std(logret,  LOOKBACK_LT) * np.sqrt(ANNUALIZATION)

mu_ann    = ALPHA_SHORT * mu_st + (1 - ALPHA_SHORT) * mu_lt
sigma_ann = sigma_lt.clip(lower=1e-6)

excess_mu = mu_ann - r_daily * ANNUALIZATION
u_star    = (excess_mu / (RISK_AVERSION_GAMMA * sigma_ann ** 2)).clip(
    MIN_WEIGHT, MAX_WEIGHT
)
u_star.iloc[:WARMUP] = 0.0
print(f"Warmup: first {WARMUP} days zeroed (until {df.index[WARMUP].date()})")

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

u_star = u_star.shift(1).fillna(0.0)
band   = band.shift(1).fillna(band.expanding(min_periods=1).median())

dates    = df.index
w        = 0.0
equity   = 1.0
weights  = []
targets  = []
eq_curve = []

for t in range(1, len(dates)):
    date = dates[t]
    tgt  = u_star.loc[date]
    b    = band.loc[date]
    diff = tgt - w

    if abs(diff) > b:
        direction = np.sign(diff)
        new_w     = np.clip(w + direction * b, MIN_WEIGHT, MAX_WEIGHT)
        traded    = abs(new_w - w)
        equity   *= (1 - tc * traded)
        w         = new_w

    port_ret = w * ret.loc[date] + (1 - w) * r_daily.loc[date]
    equity  *= (1 + port_ret)

    weights.append(w)
    targets.append(tgt)
    eq_curve.append(equity)

equity_s  = pd.Series(eq_curve, index=dates[1:])
weights_s = pd.Series(weights,  index=equity_s.index)
targets_s = pd.Series(targets,  index=equity_s.index)

out = pd.DataFrame({
    "price":    px.reindex(equity_s.index),
    "equity":   equity_s,
    "w_actual": weights_s,
    "w_target": targets_s,
    "ret":      ret.reindex(equity_s.index),
    "r_daily":  r_daily.reindex(equity_s.index),
})

# ============================================================
# METRICS
# ============================================================

train_eq   = out["equity"].loc[:TRAIN_SPLIT_DATE]
test_eq    = out["equity"].loc[TRAIN_SPLIT_DATE:]
train_rets = train_eq.pct_change().fillna(0)
test_rets  = test_eq.pct_change().fillna(0)
all_rets   = out["equity"].pct_change().fillna(0)

print(f"\nOverall:  CAGR={segment_cagr(out['equity']):.2%}  "
      f"Sharpe={sharpe_ratio(all_rets):.2f}  "
      f"MaxDD={max_drawdown(out['equity']):.2%}")
print(f"Train:    CAGR={segment_cagr(train_eq):.2%}  "
      f"Sharpe={sharpe_ratio(train_rets):.2f}  "
      f"MaxDD={max_drawdown(train_eq):.2%}")
print(f"Test:     CAGR={segment_cagr(test_eq):.2%}  "
      f"Sharpe={sharpe_ratio(test_rets):.2f}  "
      f"MaxDD={max_drawdown(test_eq):.2%}")

# ============================================================
# WEALTH COMPARISON
# ============================================================

interest  = 0.03625
rf_rate   = interest / ANNUALIZATION

rf_wealth = [1.0]
for _ in range(len(out.index)):
    rf_wealth.append(rf_wealth[-1] * (1 + rf_rate))
rf_wealth = pd.Series(rf_wealth[1:], index=out.index)

risky_sp = [1.0]
for t in range(1, len(out.index)):
    risky_sp.append(risky_sp[-1] * (1 + out["ret"].iloc[t]))
risky_sp = pd.Series(risky_sp, index=out.index)

static_60 = [1.0]
for t in range(1, len(out.index)):
    r = 0.6 * out["ret"].iloc[t] + 0.4 * out["r_daily"].iloc[t]
    static_60.append(static_60[-1] * (1 + r))
static_60 = pd.Series(static_60, index=out.index)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(out.index,    rf_wealth,       lw=1.0, ls=":",  color="grey",
        label="Risk-Free (3.625%)")
ax.plot(out.index,    risky_sp,        lw=1.0, ls="--", color="tab:green",
        alpha=0.8, label="Buy-and-Hold S\\&P 500")
ax.plot(out.index,    static_60,       lw=1.0, ls="-.", color="tab:orange",
        alpha=0.8, label="60\\% Static Allocation")
ax.plot(out.index,    out["equity"],   lw=1.4,           color="tab:blue",
        label="HJB Strategy")
ax.axvline(pd.Timestamp(TRAIN_SPLIT_DATE), color="black", ls="--",
           lw=0.8, label="Train/Test Split")
ax.set_title("Wealth Comparison — HJB Strategy vs Benchmarks")
ax.set_xlabel("Date")
ax.set_ylabel("Normalised Wealth")
ax.legend(framealpha=0.7, ncol=2)
plt.tight_layout()
plt.savefig("fig_wealth_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# WEIGHT VS TARGET — 2023 TO PRESENT
# ============================================================

sub  = out.loc["2023-01-01":]
diff = sub["w_target"] - sub["w_actual"]

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(sub.index, sub["w_actual"], lw=1.0, color="tab:blue",
        label="Actual $w(t)$")
ax.plot(sub.index, sub["w_target"], lw=0.9, ls="--", alpha=0.9,
        color="tab:orange", label="Target $u^*(t)$")
ax.fill_between(sub.index, sub["w_actual"], sub["w_target"],
                where=(diff > 0).values, alpha=0.18, color="tab:green",
                label="Buying pressure")
ax.fill_between(sub.index, sub["w_actual"], sub["w_target"],
                where=(diff < 0).values, alpha=0.18, color="tab:red",
                label="Selling pressure")
ax.set_title("Actual Weight vs HJB Target — 2023 to Present")
ax.set_xlabel("Date")
ax.set_ylabel("Risky Asset Weight")
ax.legend(framealpha=0.7, ncol=2)
plt.tight_layout()
plt.savefig("fig_weight_vs_target_2023.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nDone. Saved: fig_wealth_comparison.png, fig_weight_vs_target_full.png,")
print("             fig_weight_vs_target_2023.png")