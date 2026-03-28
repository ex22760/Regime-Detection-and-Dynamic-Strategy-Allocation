import os
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")  # Your FRED API key


START_DATE = "1990-01-02"  # VIX data starts here; all FRED series predate this
END_DATE = "2025-10-01"


print("Downloading S&P 500 data...")
ticker = "^GSPC"
sp500 = yf.download(ticker, start=START_DATE, end=END_DATE, interval="1d")
sp500.index = pd.to_datetime(sp500.index)
print("S&P 500 data loaded:", sp500.shape)


fred = Fred(api_key=FRED_API_KEY)
indicators = {
    "CPI": "CPIAUCSL",
    "Unemployment": "UNRATE",
    "FedFunds": "FEDFUNDS"
}

print("Downloading FRED macroeconomic data...")
macro_data = pd.DataFrame()
for name, code in indicators.items():
    series = fred.get_series(code, observation_start=START_DATE)
    series = series.rename(name)
    macro_data = pd.concat([macro_data, series], axis=1)

macro_data.index = pd.to_datetime(macro_data.index)
macro_data = macro_data.resample('D').ffill()  # Forward-fill missing days
print("Macro data loaded:", macro_data.shape)


print("Downloading VIX data...")
vix = yf.download("^VIX", start=START_DATE, end=END_DATE, interval="1d")
vix.index = pd.to_datetime(vix.index)

# Flatten multi-index if present and keep only Close
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in vix.columns]
vix_close_col = [c for c in vix.columns if "Close" in c][0]
vix = vix[[vix_close_col]].rename(columns={vix_close_col: "VIX"})
vix = vix.resample('D').ffill()
print("VIX data loaded:", vix.shape)


print("Merging datasets...")
full_data = sp500.join([macro_data, vix], how="outer")
full_data = full_data.loc[START_DATE:]  # Keep only data from start date onward
full_data = full_data.ffill()  # Fill any remaining NaNs
print("Merged dataset shape:", full_data.shape)


print("Computing engineered features...")

# Flatten multi-index columns from yfinance
if isinstance(full_data.columns, pd.MultiIndex):
    full_data.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in full_data.columns]

# Identify the S&P 500 Close column dynamically
close_col_candidates = [c for c in full_data.columns if "Close" in c and "VIX" not in c]
if not close_col_candidates:
    raise KeyError("Cannot find the S&P 500 Close column in your dataset.")
close_col = close_col_candidates[0]
print("Using Close column:", close_col)

# Compute features
full_data['LogReturn'] = np.log(full_data[close_col] / full_data[close_col].shift(1))
full_data['Volatility_10d'] = full_data['LogReturn'].rolling(window=10).std()
full_data['Momentum_10d'] = full_data[close_col] - full_data[close_col].shift(10)
full_data['SMA_10'] = full_data[close_col].rolling(window=10).mean()
full_data['SMA_50'] = full_data[close_col].rolling(window=50).mean()
full_data['EMA_10'] = full_data[close_col].ewm(span=10, adjust=False).mean()
full_data['EMA_50'] = full_data[close_col].ewm(span=50, adjust=False).mean()
rolling_max = full_data[close_col].cummax()
full_data['Drawdown'] = (full_data[close_col] - rolling_max) / rolling_max

# Drop initial NaNs from rolling calculations
full_data = full_data.dropna()


full_data.to_csv(r"c:\Users\sujin_2qsl8p2\Desktop\dissertation\sp500_features.csv")
print(full_data.head())

# ============================================================
# REGIME LABELS (NBER + PAGAN-SOSSOUNOV)
# ============================================================

print("Computing NBER + Pagan-Sossounov regimes...")

# Ensure NBER recession exists
nber = fred.get_series("USREC", observation_start=START_DATE)
nber = nber.rename("Recession")
nber.index = pd.to_datetime(nber.index)
nber = nber.resample('D').ffill()

full_data = full_data.join(nber, how="left").ffill()

# Pagan-Sossounov trend (bull vs neutral)
def pagan_sossounov_trend(price, window=200, threshold=0.15):
    trend = price.rolling(window).mean()
    signal = pd.Series(0, index=price.index)
    signal[(price > trend) & ((price / trend - 1) > threshold)] = 1
    return signal

ps_signal = pagan_sossounov_trend(full_data[close_col])

# Final regime:
# -2 = Bear (NBER recession)
#  1 = Bull (PS uptrend in expansion)
#  0 = Neutral (remaining expansion)
full_data["Regime_PS"] = 0
full_data.loc[full_data["Recession"] == 1, "Regime_PS"] = -2
full_data.loc[(full_data["Recession"] == 0) & (ps_signal == 1), "Regime_PS"] = 1

# ---------- FIGURE 4.1: S&P 500 PRICE ----------
plt.figure()
plt.plot(full_data.index, full_data[close_col])
plt.title("S&P 500 Daily Close Price")
plt.xlabel("Date")
plt.ylabel("Index Level")
plt.tight_layout()
plt.savefig("fig04_sp500_price.png", dpi=300)
plt.show()

# ---------- FIGURE 4.2: RETURNS & VOLATILITY ----------
fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(full_data.index, full_data["LogReturn"])
ax[0].set_title("Daily Log Returns")
ax[0].set_ylabel("Log Return")

ax[1].plot(full_data.index, full_data["Volatility_10d"])
ax[1].set_title("Rolling 10-Day Volatility")
ax[1].set_ylabel("Volatility")

plt.xlabel("Date")
plt.tight_layout()
plt.savefig("fig04_returns_volatility.png", dpi=300)
plt.show()

# ---------- FIGURE 4.3: TREND & MOMENTUM ----------
plt.figure()
plt.plot(full_data.index, full_data[close_col], label="Price", alpha=0.7)
plt.plot(full_data.index, full_data["SMA_10"], label="SMA 10")
plt.plot(full_data.index, full_data["SMA_50"], label="SMA 50")
plt.plot(full_data.index, full_data["EMA_10"], label="EMA 10", linestyle="--")
plt.plot(full_data.index, full_data["EMA_50"], label="EMA 50", linestyle="--")
plt.title("Trend and Momentum Indicators")
plt.xlabel("Date")
plt.ylabel("Index Level")
plt.legend()
plt.tight_layout()
plt.savefig("fig04_trend_indicators.png", dpi=300)
plt.show()

# ---------- FIGURE 4.4: DRAWDOWNS ----------
plt.figure()
plt.plot(full_data.index, full_data["Drawdown"])
plt.fill_between(full_data.index, full_data["Drawdown"], 0, alpha=0.3)
plt.title("S&P 500 Drawdowns")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.tight_layout()
plt.savefig("fig04_drawdowns.png", dpi=300)
plt.show()

# ---------- FIGURE 4.4: NBER + PAGAN-SOSSOUNOV REGIMES ----------
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(full_data.index, full_data[close_col], color="black", lw=1.2)

# Masks
bull = full_data["Regime_PS"] == 1
neutral = full_data["Regime_PS"] == 0
bear = full_data["Regime_PS"] == -2

def shade(ax, dates, mask, color, alpha):
    in_block = False
    start = None
    for i in range(len(mask)):
        if mask.iloc[i] and not in_block:
            start = dates[i]
            in_block = True
        elif not mask.iloc[i] and in_block:
            ax.axvspan(start, dates[i], color=color, alpha=alpha)
            in_block = False
    if in_block:
        ax.axvspan(start, dates[-1], color=color, alpha=alpha)

shade(ax, full_data.index, bull, "green", 0.15)
shade(ax, full_data.index, neutral, "grey", 0.10)
shade(ax, full_data.index, bear, "red", 0.25)

ax.set_title("Figure 4.4: S&P 500 with NBER and Pagan-Sossounov Regime Labels")
ax.set_xlabel("Date")
ax.set_ylabel("Index Level")

import matplotlib.patches as mpatches
legend = [
    mpatches.Patch(color="green", alpha=0.2, label="Bull (PS Uptrend)"),
    mpatches.Patch(color="grey", alpha=0.2, label="Neutral (Expansion)"),
    mpatches.Patch(color="red", alpha=0.3, label="Bear (NBER Recession)")
]
ax.legend(handles=legend)

ax.grid(alpha=0.3)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig("fig04_regimes_ps.png", dpi=300)
plt.show()

# ---------- FIGURE 4.5: MACRO Z-SCORES ----------
macro_cols = ["CPI", "Unemployment", "FedFunds"]
macro_z = (
    full_data[macro_cols]
    .rolling(252)
    .apply(lambda x: (x.iloc[-1] - x.mean()) / x.std())
)

plt.figure()
for col in macro_cols:
    plt.plot(macro_z.index, macro_z[col], label=col)

plt.title("Standardised Macroeconomic Indicators (Rolling Z-Scores)")
plt.xlabel("Date")
plt.ylabel("Z-Score")
plt.legend()
plt.tight_layout()
plt.savefig("fig04_macro_zscores.png", dpi=300)
plt.show()

# ---------- FIGURE 4.6: VIX (MARKET VOLATILITY / FEAR INDEX) ----------
plt.figure()
plt.plot(full_data.index, full_data["VIX"], color="crimson", label="VIX")
plt.axhline(y=20, color="orange", linestyle="--", linewidth=0.8, label="Elevated volatility (20)")
plt.axhline(y=30, color="red", linestyle="--", linewidth=0.8, label="High volatility (30)")
plt.fill_between(full_data.index, full_data["VIX"], 0, alpha=0.15, color="crimson")
plt.title("CBOE Volatility Index (VIX) — Market Fear Gauge")
plt.xlabel("Date")
plt.ylabel("VIX Level")
plt.legend()
plt.tight_layout()
plt.savefig("fig04_vix.png", dpi=300)
plt.show()

print("All Chapter 4 figures generated and saved.")


# ============================================================
# FIGURE 4.7 — MACRO NORMALITY DIAGNOSTICS
# For each macro series: raw time series, histogram + KDE,
# Q-Q plot, and rolling z-score as used in the model.
# Tests: Shapiro-Wilk and Jarque-Bera with printed results.
# ============================================================

LOOKBACK_LT = 126

macro_labels = {
    "CPI":          "CPI (Consumer Price Index)",
    "Unemployment": "Unemployment Rate (%)",
    "FedFunds":     "Federal Funds Rate (%)"
}

print("\n--- MACRO NORMALITY TESTS ---")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Figure 4.7 — Macro Indicator Normality Diagnostics",
             fontsize=14, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

for row, col in enumerate(macro_cols):
    series = full_data[col].dropna()
    label  = macro_labels.get(col, col)

    # Shapiro-Wilk (on up to 5000 samples)
    sample = series.sample(min(5000, len(series)), random_state=42)
    sw_stat, sw_p = stats.shapiro(sample)

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(series)

    skew = series.skew()
    kurt = series.kurtosis()  # excess kurtosis (normal = 0)

    print(f"\n{label}:")
    print(f"  Shapiro-Wilk  stat={sw_stat:.4f}, p={sw_p:.2e}  "
          f"{'→ NON-NORMAL' if sw_p < 0.05 else '→ normal'}")
    print(f"  Jarque-Bera   stat={jb_stat:.2f}, p={jb_p:.2e}  "
          f"{'→ NON-NORMAL' if jb_p < 0.05 else '→ normal'}")
    print(f"  Skewness: {skew:.3f}  |  Excess kurtosis: {kurt:.3f}")

    # Panel 1: Time series
    ax1 = fig.add_subplot(gs[row, 0])
    ax1.plot(series.index, series.values, lw=0.8, color="steelblue")
    ax1.set_title(f"{label}\nTime Series", fontsize=9)
    ax1.set_ylabel("Value")
    ax1.tick_params(labelsize=7)

    # Panel 2: Histogram + normal fit + KDE
    ax2 = fig.add_subplot(gs[row, 1])
    ax2.hist(series.values, bins=60, density=True,
             color="steelblue", alpha=0.6, edgecolor="white", linewidth=0.3)
    x_range = np.linspace(series.min(), series.max(), 300)
    ax2.plot(x_range, stats.norm.pdf(x_range, series.mean(), series.std()),
             color="red", lw=2, label="Normal fit")
    ax2.plot(x_range, stats.gaussian_kde(series.values)(x_range),
             color="orange", lw=2, ls="--", label="KDE")
    ax2.set_title(f"Distribution\nSkew={skew:.2f}, Kurt={kurt:.2f}", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.tick_params(labelsize=7)

    # Panel 3: Q-Q plot
    ax3 = fig.add_subplot(gs[row, 2])
    (osm, osr), (slope, intercept, r) = stats.probplot(series.values, dist="norm")
    ax3.scatter(osm, osr, s=2, alpha=0.4, color="steelblue")
    ax3.plot(osm, slope * np.array(osm) + intercept, color="red", lw=1.5)
    ax3.set_title(f"Q-Q Plot\nR²={r**2:.3f}", fontsize=9)
    ax3.set_xlabel("Theoretical quantiles", fontsize=7)
    ax3.set_ylabel("Sample quantiles", fontsize=7)
    ax3.tick_params(labelsize=7)

    # Panel 4: Rolling z-score as used in the model
    ax4 = fig.add_subplot(gs[row, 3])
    z_score = full_data[col].rolling(LOOKBACK_LT).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8), raw=False
    ).clip(-3, 3)
    ax4.plot(z_score.index, z_score.values, lw=0.8, color="darkorange")
    ax4.axhline(0,  color="black", lw=0.8, ls="--")
    ax4.axhline(2,  color="red",   lw=0.8, ls=":", alpha=0.7)
    ax4.axhline(-2, color="red",   lw=0.8, ls=":", alpha=0.7)
    ax4.set_title(
        f"Rolling Z-Score (126d)\nSW p={sw_p:.1e}  JB p={jb_p:.1e}",
        fontsize=9
    )
    ax4.set_ylabel("Z-score")
    ax4.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig("fig04_macro_normality.png", dpi=200, bbox_inches="tight")
print("\nSaved: fig04_macro_normality.png")
plt.show()

print(
    "\nNOTE: Non-normality of the raw macro series does not invalidate the z-score "
    "approach. The rolling z-score is used purely for standardisation (comparable "
    "scale across series), not to derive probabilities from a normal assumption."
)


# ============================================================
# FIGURE 4.8 — VIX vs REALISED VOLATILITY DIAGNOSTICS
# Time series overlay, scatter + regression, rolling correlation,
# and VIX risk premium distribution.
# ============================================================

print("\n--- VIX vs REALISED VOLATILITY ---")

# Realised vol: causal EWM std annualised (matches hybrid.py)
def causal_ewm_std(x, span):
    alpha    = 2.0 / (span + 1)
    result   = np.full(len(x), np.nan)
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

sigma_realised = causal_ewm_std(
    full_data["LogReturn"], LOOKBACK_LT
).clip(lower=1e-6) * np.sqrt(252)

vix_dec = full_data["VIX"] / 100.0   # convert to decimal annualised

common = pd.DataFrame({
    "realised": sigma_realised,
    "vix":      vix_dec
}).dropna()

corr_pearson,  p_pearson  = stats.pearsonr(common["realised"], common["vix"])
corr_spearman, p_spearman = stats.spearmanr(common["realised"], common["vix"])
vix_premium   = common["vix"] - common["realised"]
rolling_corr  = common["realised"].rolling(252).corr(common["vix"])

print(f"  Pearson  r={corr_pearson:.3f}, p={p_pearson:.2e}")
print(f"  Spearman r={corr_spearman:.3f}, p={p_spearman:.2e}")
print(f"  VIX risk premium — mean: {vix_premium.mean()*100:.2f}%  "
      f"std: {vix_premium.std()*100:.2f}%")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Figure 4.8 — VIX vs Realised Volatility Diagnostics",
             fontsize=13, fontweight="bold")

# Panel 1: Time series overlay
ax = axes[0, 0]
ax.plot(common.index, common["realised"] * 100, lw=0.9,
        color="steelblue", label="Realised Vol (EWM-126, ann. %)", alpha=0.9)
ax.plot(common.index, common["vix"] * 100, lw=0.9,
        color="darkorange", label="VIX (ann. %)", alpha=0.8)
ax.set_title("Time Series: VIX vs Realised Volatility")
ax.set_ylabel("Annualised Volatility (%)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 2: Scatter + OLS + 45-degree line
ax = axes[0, 1]
ax.scatter(common["realised"] * 100, common["vix"] * 100,
           s=2, alpha=0.25, color="steelblue")
slope, intercept, r_val, _, _ = stats.linregress(common["realised"], common["vix"])
x_fit = np.linspace(common["realised"].min(), common["realised"].max(), 200)
ax.plot(x_fit * 100, (slope * x_fit + intercept) * 100,
        color="red", lw=2, label=f"OLS (R²={r_val**2:.3f})")
ax.plot([0, 80], [0, 80], color="grey", lw=1.2, ls="--",
        alpha=0.7, label="45° line (VIX = Realised)")
ax.set_xlabel("Realised Vol (%)")
ax.set_ylabel("VIX (%)")
ax.set_title(f"Scatter — Pearson r={corr_pearson:.3f}, Spearman r={corr_spearman:.3f}")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 3: Rolling 252-day correlation
ax = axes[1, 0]
ax.plot(rolling_corr.index, rolling_corr.values, lw=1.2,
        color="purple", label="Rolling 252d Pearson r")
ax.axhline(corr_pearson, color="red", lw=1.2, ls="--",
           label=f"Full-sample r={corr_pearson:.3f}")
ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6, label="r=0.5 reference")
ax.set_title("Rolling 252-Day Correlation: VIX vs Realised Vol")
ax.set_ylabel("Pearson r")
ax.set_ylim(-0.1, 1.1)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 4: VIX risk premium distribution
ax = axes[1, 1]
ax.hist(vix_premium.values * 100, bins=80, density=True,
        color="darkorange", alpha=0.6, edgecolor="white", linewidth=0.3)
x_range = np.linspace(vix_premium.min() * 100, vix_premium.max() * 100, 300)
ax.plot(x_range, stats.gaussian_kde(vix_premium.dropna().values * 100)(x_range),
        color="red", lw=2, label="KDE")
ax.axvline(0, color="black", lw=1.5, ls="--", label="Zero premium")
ax.axvline(vix_premium.mean() * 100, color="blue", lw=1.5, ls="--",
           label=f"Mean={vix_premium.mean()*100:.1f}%")
ax.set_xlabel("VIX Premium = VIX − Realised Vol (%)")
ax.set_ylabel("Density")
ax.set_title(f"VIX Risk Premium — Mean={vix_premium.mean()*100:.2f}%, "
             f"Std={vix_premium.std()*100:.2f}%")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig04_vix_diagnostics.png", dpi=200, bbox_inches="tight")
print("Saved: fig04_vix_diagnostics.png")
plt.show()

print("\nAll figures generated and saved.")

# ============================================================
# FIGURE 4.9 — BRY-BOSCHAN VS NBER (CONFUSION MATRIX)
# ============================================================

print("\n--- BRY-BOSCHAN vs NBER COMPARISON ---")

# Simplified Bry–Boschan
def bry_boschan(price, window=5):
    log_p = np.log(price)

    peaks = (log_p.shift(window) < log_p) & (log_p.shift(-window) < log_p)
    troughs = (log_p.shift(window) > log_p) & (log_p.shift(-window) > log_p)

    regime = pd.Series(0, index=price.index)

    last_type = None
    for i in range(len(price)):
        if peaks.iloc[i]:
            last_type = "peak"
        elif troughs.iloc[i]:
            last_type = "trough"

        if last_type == "trough":
            regime.iloc[i] = 1
        elif last_type == "peak":
            regime.iloc[i] = -1

    return regime

bb_regime = bry_boschan(full_data[close_col])

# Convert to binary recession-style (match NBER)
bb_binary = (bb_regime == -1).astype(int)   # 1 = contraction
nber_binary = (full_data["Recession"] == 1).astype(int)

# Align
common = pd.DataFrame({
    "BB": bb_binary,
    "NBER": nber_binary
}).dropna()

# Confusion matrix
tp = np.sum((common["BB"] == 1) & (common["NBER"] == 1))
tn = np.sum((common["BB"] == 0) & (common["NBER"] == 0))
fp = np.sum((common["BB"] == 1) & (common["NBER"] == 0))
fn = np.sum((common["BB"] == 0) & (common["NBER"] == 1))

cm = np.array([[tn, fp],
               [fn, tp]])

print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(cm)

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(["Expansion", "Recession"])
ax.set_yticklabels(["Expansion", "Recession"])

ax.set_xlabel("NBER")
ax.set_ylabel("Bry-Boschan")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

ax.set_title("Figure 4.9: Bry-Boschan vs NBER Regime Classification")

plt.tight_layout()
plt.savefig("fig04_bb_vs_nber_confusion.png", dpi=300)
plt.show()

# Accuracy stats
accuracy = (tp + tn) / cm.sum()
print(f"Accuracy: {accuracy:.3f}")