# ============================================================
# REGIME DASHBOARD — Real-Time Implementation
# Chapter 9: Real-Time Implementation and Visualisation
#
# Architecture:
#   1. Fetch live S&P 500 + VIX from yfinance
#   2. Fetch CPI, Unemployment, FedFunds from FRED
#   3. Compute causal features (same pipeline as training)
#   4. Run frozen ensemble inference (GMM, HMM, RF, LSTM)
#   5. Blend three strategies via regime probabilities
#   6. Display live dashboard with auto-refresh
#
# Usage:
#   pip install streamlit yfinance pandas-datareader plotly
#   python -m streamlit run dashboard.py
#
# The models (gmm, hmm, rf_cal, lstm_model, scaler) must be
# saved from dynamic_allocation.py before running this dashboard.
# Run: python dynamic_allocation.py
# This saves regime_models.pkl, lstm_model.pt, lstm_input_size.txt
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import os
import pickle
import torch
import torch.nn as nn
from datetime import datetime, timedelta

# ── Page config — must be first Streamlit call
st.set_page_config(
    page_title="Regime Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3 !important;
}

p, span, div, label, li, a, td, th { color: #e6edf3 !important; }

[data-testid="stDataFrame"] * { color: #e6edf3 !important; }

[data-testid="stTabs"] button { color: #8b949e !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e6edf3 !important;
    border-bottom-color: #58a6ff !important;
}

[data-testid="stWidgetLabel"] { color: #8b949e !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }

.stApp { background-color: #0d1117; }

[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
}

[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.02em; }
h1 { font-size: 1.4rem; font-weight: 500; color: #e6edf3; }
h2 { font-size: 1.0rem; font-weight: 500; color: #8b949e;
     text-transform: uppercase; letter-spacing: 0.08em; }

hr { border-color: #21262d; }

.regime-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.bear    { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.neutral { background: rgba(234,179,8,0.15);  color: #eab308; border: 1px solid rgba(234,179,8,0.3); }
.bull    { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }

.info-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    margin: 8px 0;
}

.stButton button {
    background: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}
.stButton button:hover {
    border-color: #58a6ff;
    color: #58a6ff;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS — must match training configuration exactly
# ============================================================
LOOKBACK_ST          = 20
LOOKBACK_LT          = 126
ANNUALIZATION        = 252
RISK_AVERSION_GAMMA  = 4.0
K_REGIMES            = 3
SEQ_LEN              = 20
HIDDEN_SIZE          = 32
CONFIDENCE_THRESHOLD = 0.55
MIN_WEIGHT           = 0.0
MAX_WEIGHT           = 1.0
TC_BPS               = 5
BAND_MULT            = 1.5
BEAR_FLOOR           = 0.05
VOL_NEUTRAL_CAP      = 2.0
MACRO_LAGS           = {"CPI": 21, "Unemployment": 7, "FedFunds": 1}
DEVICE               = torch.device("cpu")

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# LSTM ARCHITECTURE — must match training exactly
# ============================================================
class LSTMRegime(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(self.dropout(h[-1]))

# ============================================================
# CAUSAL EWM HELPERS
# ============================================================
def causal_ewm_mean(x, span):
    alpha = 2.0 / (span + 1)
    result, ewm_val = np.full(len(x), np.nan), np.nan
    for i, val in enumerate(x):
        if np.isnan(val):
            result[i] = np.nan; continue
        ewm_val  = val if np.isnan(ewm_val) else (1 - alpha) * ewm_val + alpha * val
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
# PLOTLY THEME
# ============================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono", color="#e6edf3", size=11),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=True,
               tickfont=dict(color="#e6edf3"), title_font=dict(color="#e6edf3")),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", showgrid=True,
               tickfont=dict(color="#e6edf3"), title_font=dict(color="#e6edf3")),
    margin=dict(l=48, r=24, t=36, b=36),
    legend=dict(
        bgcolor="rgba(22,27,34,0.8)",
        bordercolor="#30363d",
        borderwidth=1,
        font=dict(size=10, color="#e6edf3")
    ),
    hovermode="x unified",
)

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_models():
    pkl_path  = os.path.join(MODEL_DIR, "regime_models.pkl")
    lstm_path = os.path.join(MODEL_DIR, "lstm_model.pt")
    size_path = os.path.join(MODEL_DIR, "lstm_input_size.txt")

    if not os.path.exists(pkl_path):
        return None, "regime_models.pkl not found. Run dynamic_allocation.py first."

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    lstm_input_size = int(open(size_path).read().strip()) if os.path.exists(size_path) else 11
    lstm = LSTMRegime(lstm_input_size, HIDDEN_SIZE, K_REGIMES).to(DEVICE)
    if os.path.exists(lstm_path):
        lstm.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
    lstm.eval()
    bundle["lstm"] = lstm
    return bundle, None

# ============================================================
# DATA FETCHING
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(ticker, vix_ticker, lookback_years):
    end   = datetime.today()
    start = end - timedelta(days=lookback_years * 365)
    try:
        raw = yf.download(ticker, start=start, end=end,
                          interval="1d", auto_adjust=True, progress=False)
        px_series = raw["Close"].squeeze()
        if isinstance(px_series, pd.DataFrame):
            px_series = px_series.iloc[:, 0]
        px_series = px_series.dropna()
    except Exception as e:
        return None, None, str(e)

    vix_series = pd.Series(dtype=float)
    if vix_ticker:
        try:
            raw_vix = yf.download(vix_ticker, start=start, end=end,
                                  interval="1d", auto_adjust=True, progress=False)
            vix_series = raw_vix["Close"].squeeze()
            if isinstance(vix_series, pd.DataFrame):
                vix_series = vix_series.iloc[:, 0]
            vix_series = vix_series.dropna()
        except Exception:
            vix_series = pd.Series(dtype=float)

    return px_series, vix_series, None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_macro_data(start_year):
    try:
        import pandas_datareader.data as web
        start = f"{start_year}-01-01"
        ff  = web.DataReader("FEDFUNDS", "fred", start=start)
        cpi = web.DataReader("CPIAUCSL", "fred", start=start)
        une = web.DataReader("UNRATE",   "fred", start=start)
        ff.columns  = ["FedFunds"]
        cpi.columns = ["CPI"]
        une.columns = ["Unemployment"]
        return pd.concat([ff, cpi, une], axis=1), None
    except Exception as e:
        return None, str(e)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def build_features(px, vix, macro, gamma=4.0, st_weight=0.5, lt_weight=0.5):
    common_idx = px.index
    logret = np.log(px).diff()

    r_daily = pd.Series(0.0, index=common_idx)
    if macro is not None and "FedFunds" in macro.columns:
        ff = macro["FedFunds"].reindex(common_idx, method="ffill").fillna(0)
        ff = ff / (100.0 if ff.max() > 1 else 1.0) / ANNUALIZATION
        r_daily = ff

    macro_reindexed = None
    if macro is not None:
        macro_reindexed = macro.reindex(common_idx, method="ffill")
        for col, lag in MACRO_LAGS.items():
            if col in macro_reindexed.columns:
                macro_reindexed[col] = macro_reindexed[col].shift(lag)
        macro_reindexed = macro_reindexed.ffill()

    mu_st     = causal_ewm_mean(logret, LOOKBACK_ST) * ANNUALIZATION
    mu_lt     = causal_ewm_mean(logret, LOOKBACK_LT) * ANNUALIZATION
    sigma_ann = causal_ewm_std(logret, LOOKBACK_LT).clip(lower=1e-6) * np.sqrt(ANNUALIZATION)
    total     = st_weight + lt_weight
    mu_ann    = (st_weight / total) * mu_st + (lt_weight / total) * mu_lt

    macro_signal = pd.Series(0.0, index=common_idx)
    if macro_reindexed is not None:
        avail = [c for c in ["CPI", "Unemployment", "FedFunds"] if c in macro_reindexed.columns]
        if avail:
            z = macro_reindexed[avail].rolling(LOOKBACK_LT).apply(
                lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8), raw=False
            ).clip(-3, 3)
            w_macro      = {"CPI": -0.4, "Unemployment": -0.3, "FedFunds": -0.3}
            macro_signal = sum(z[c] * w_macro[c] for c in z.columns if c in w_macro)

    if vix is not None and len(vix) > 0:
        vix_aligned = vix.reindex(common_idx, method="ffill")
        if not vix_aligned.isna().all():
            vix_ann   = vix_aligned.astype(float) / 100.0
            sigma_ann = (0.5 * sigma_ann + 0.5 * vix_ann).clip(lower=1e-6)

    excess_mu     = mu_ann - r_daily * ANNUALIZATION
    u_star        = (excess_mu / (gamma * sigma_ann ** 2)).clip(0, 1)
    u_star_lagged = u_star.shift(1).fillna(0.0)

    feat_dict = {
        "return":   logret,
        "vol":      sigma_ann,
        "momentum": mu_ann,
        "macro":    macro_signal,
    }
    if vix is not None and len(vix) > 0:
        vix_aligned = vix.reindex(common_idx, method="ffill")
        feat_dict["vix"] = vix_aligned.astype(float)

    X_df = pd.DataFrame(feat_dict).dropna()
    return X_df, sigma_ann, mu_ann, r_daily, u_star_lagged, px.pct_change()

# ============================================================
# ENSEMBLE INFERENCE
# ============================================================
def run_ensemble(X_df, models):
    gmm           = models["gmm"]
    hmm_model     = models["hmm"]
    rf_cal        = models["rf_cal"]
    lstm_model    = models["lstm"]
    scaler        = models["scaler"]
    ens_weights   = models["ensemble_weights"]
    gmm_col_order = models["gmm_col_order"]
    hmm_col_order = models["hmm_col_order"]

    X = scaler.transform(X_df)

    gmm_proba = gmm.predict_proba(X)[:, gmm_col_order]
    hmm_proba = forward_filter_proba(hmm_model, X)[:, hmm_col_order]

    rf_feats = np.hstack([X, gmm_proba, hmm_proba])
    rf_proba = rf_cal.predict_proba(rf_feats)

    lstm_model.eval()
    lstm_logits = []
    with torch.no_grad():
        for i in range(len(rf_feats) - SEQ_LEN):
            seq = torch.tensor(
                rf_feats[i:i + SEQ_LEN], dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)
            lstm_logits.append(lstm_model(seq).cpu().numpy()[0])

    lstm_proba_valid = torch.softmax(
        torch.tensor(np.array(lstm_logits)), dim=-1
    ).numpy()
    lstm_proba = np.vstack([
        np.full((SEQ_LEN, K_REGIMES), 1.0 / K_REGIMES),
        lstm_proba_valid
    ])

    # Weighted arithmetic mean
    w = ens_weights
    ensemble_proba = (
          w["gmm"]  * gmm_proba
        + w["hmm"]  * hmm_proba
        + w["rf"]   * rf_proba
        + w["lstm"] * lstm_proba
    )

    confidence = ensemble_proba.max(axis=1)

    equal_proba = 0.25 * (gmm_proba + hmm_proba + rf_proba + lstm_proba)
    fallback    = confidence < CONFIDENCE_THRESHOLD
    ensemble_proba[fallback] = equal_proba[fallback]

    # Conflict score
    all_hard = np.stack([
        gmm_proba.argmax(axis=1),
        hmm_proba.argmax(axis=1),
        rf_proba.argmax(axis=1),
        lstm_proba.argmax(axis=1),
    ], axis=1)
    conflict = np.array([
        1.0 - (np.bincount(row.astype(int), minlength=3).max() / 4.0)
        for row in all_hard
    ])

    pred = ensemble_proba.argmax(axis=1)

    return pd.DataFrame({
        "p_bear":     ensemble_proba[:, 0],
        "p_neutral":  ensemble_proba[:, 1],
        "p_bull":     ensemble_proba[:, 2],
        "confidence": confidence,
        "conflict":   conflict,
        "pred":       pred,
        "pred_name":  [["bear", "neutral", "bull"][p] for p in pred],
    }, index=X_df.index)

# ============================================================
# STRATEGY ALLOCATION
# ============================================================
def compute_allocation(result, sigma_ann, mu_ann, w_hjb, models, gamma=4.0):
    sigma_med = models["sigma_ann_median"]

    p_bull    = result["p_bull"].values
    p_neut    = result["p_neutral"].values
    p_bear    = result["p_bear"].values
    conf      = result["confidence"].values
    w_hjb_arr = w_hjb.reindex(result.index).fillna(0).values

    trend_signal = np.clip(
        mu_ann.reindex(result.index).values /
        (sigma_ann.reindex(result.index).values + 1e-8), -1, 1
    )
    w_momentum  = np.clip(w_hjb_arr * (1 + 0.2 * np.maximum(trend_signal, 0)), 0, MAX_WEIGHT)
    vol_ratio   = np.clip(sigma_ann.reindex(result.index).values / (sigma_med + 1e-8), 0.5, VOL_NEUTRAL_CAP)
    w_neutral   = np.clip(w_hjb_arr / vol_ratio, MIN_WEIGHT, MAX_WEIGHT)
    w_defensive = np.full(len(result), BEAR_FLOOR)

    w_raw     = p_bull * w_momentum + p_neut * w_neutral + p_bear * w_defensive
    w_blended = np.clip(conf * w_raw + (1 - conf) * w_hjb_arr, MIN_WEIGHT, MAX_WEIGHT)

    return pd.Series(w_blended, index=result.index), pd.Series(w_hjb_arr, index=result.index)

# ============================================================
# PERFORMANCE METRICS
# ============================================================
def compute_performance(prices, weights, r_daily):
    ret    = prices.pct_change().reindex(weights.index).fillna(0)
    rf     = r_daily.reindex(weights.index).fillna(0)
    tc     = TC_BPS / 1e4
    equity = [1.0]
    w_prev = 0.0

    for date in weights.index[1:]:
        w_tgt   = float(weights.loc[date])
        base_eq = equity[-1]
        if abs(w_tgt - w_prev) > 0.01:
            base_eq = base_eq * (1 - tc * abs(w_tgt - w_prev))
            w_prev  = w_tgt
        daily_ret = w_prev * ret.loc[date] + (1 - w_prev) * rf.loc[date]
        equity.append(base_eq * (1 + daily_ret))

    eq_series = pd.Series(equity, index=weights.index)
    rets      = eq_series.pct_change().fillna(0)
    n_years   = len(eq_series) / ANNUALIZATION
    cagr      = (eq_series.iloc[-1] / eq_series.iloc[0]) ** (1 / max(n_years, 0.1)) - 1
    sharpe    = (rets.mean() * ANNUALIZATION) / (rets.std() * np.sqrt(ANNUALIZATION) + 1e-8)
    maxdd     = (eq_series / eq_series.cummax() - 1).min()

    return eq_series, {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": maxdd}

# ============================================================
# PLOTTING
# ============================================================
def plot_price_with_regimes(px_series, result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=px_series.index, y=px_series.values,
        line=dict(color="#e6edf3", width=1.5),
        name="Price", hovertemplate="%{y:,.0f}"
    ))
    pred = result["pred_name"].reindex(px_series.index).ffill()
    regime_colors = {"bear": "rgba(239,68,68,0.12)",
                     "neutral": "rgba(234,179,8,0.08)",
                     "bull": "rgba(34,197,94,0.10)"}
    prev_regime, start_date = pred.iloc[0], pred.index[0]
    for date, regime in pred.items():
        if regime != prev_regime:
            fig.add_vrect(x0=start_date, x1=date,
                          fillcolor=regime_colors.get(prev_regime, "rgba(0,0,0,0)"),
                          layer="below", line_width=0)
            start_date, prev_regime = date, regime
    fig.add_vrect(x0=start_date, x1=pred.index[-1],
                  fillcolor=regime_colors.get(prev_regime, "rgba(0,0,0,0)"),
                  layer="below", line_width=0)
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Price with Ensemble Regime Signal", font=dict(size=12)),
                      yaxis_title="Price", height=320, showlegend=False)
    return fig

def plot_regime_probabilities(result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result["p_bear"],
                             name="P(Bear)", fill="tozeroy",
                             line=dict(color="#ef4444", width=0),
                             fillcolor="rgba(239,68,68,0.6)"))
    fig.add_trace(go.Scatter(x=result.index, y=result["p_neutral"],
                             name="P(Neutral)", fill="tozeroy",
                             line=dict(color="#eab308", width=0),
                             fillcolor="rgba(234,179,8,0.5)"))
    fig.add_trace(go.Scatter(x=result.index, y=result["p_bull"],
                             name="P(Bull)", fill="tozeroy",
                             line=dict(color="#22c55e", width=0),
                             fillcolor="rgba(34,197,94,0.5)"))
    layout = {**PLOTLY_LAYOUT}
    layout["yaxis"] = dict(range=[0, 1], gridcolor="#21262d", linecolor="#30363d", showgrid=True)
    fig.update_layout(**layout, title=dict(text="Regime Probabilities", font=dict(size=12)), height=220)
    return fig

def plot_weights(w_ensemble, w_baseline):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w_ensemble.index, y=w_ensemble.values,
                             name="Blended Ensemble", line=dict(color="#58a6ff", width=1.8)))
    fig.add_trace(go.Scatter(x=w_baseline.index, y=w_baseline.values,
                             name="HJB Baseline", line=dict(color="#8b949e", width=1.2, dash="dash")))
    layout = {**PLOTLY_LAYOUT}
    layout["yaxis"] = dict(range=[-0.05, 1.05], tickformat=".0%",
                           gridcolor="#21262d", linecolor="#30363d", showgrid=True)
    fig.update_layout(**layout, title=dict(text="Portfolio Weight — Risky Asset", font=dict(size=12)), height=220)
    return fig

def plot_equity_curves(eq_ensemble, eq_baseline):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq_ensemble.index, y=eq_ensemble.values,
                             name="Blended Ensemble", line=dict(color="#58a6ff", width=2)))
    fig.add_trace(go.Scatter(x=eq_baseline.index, y=eq_baseline.values,
                             name="HJB Baseline", line=dict(color="#8b949e", width=1.5, dash="dash")))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Equity Curves (Normalised)", font=dict(size=12)), height=260)
    return fig

def plot_confidence_conflict(result):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=result.index, y=result["confidence"],
                             name="Confidence", fill="tozeroy",
                             line=dict(color="#58a6ff", width=1),
                             fillcolor="rgba(88,166,255,0.2)"), row=1, col=1)
    fig.add_hline(y=CONFIDENCE_THRESHOLD, line_dash="dash", line_color="#f0883e", row=1, col=1)
    fig.add_trace(go.Scatter(x=result.index, y=result["conflict"],
                             name="Conflict", fill="tozeroy",
                             line=dict(color="#ef4444", width=1),
                             fillcolor="rgba(239,68,68,0.2)"), row=2, col=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=240,
                      title=dict(text="Confidence & Conflict Scores", font=dict(size=12)))
    fig.update_yaxes(range=[0, 1], gridcolor="#21262d", row=1, col=1)
    fig.update_yaxes(range=[0, 1], gridcolor="#21262d", row=2, col=1)
    return fig

# ============================================================
# MAIN DASHBOARD
# ============================================================
def main():
    # ── Sidebar
    with st.sidebar:
        st.markdown("## ⬡ REGIME DASHBOARD")
        st.markdown('<div class="info-box">Hybrid ensemble: GMM · HMM · RF · LSTM<br>Arithmetic mean · Pagan-Sossounov labels</div>',
                    unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### 📌 Asset")
        ticker     = st.text_input("Ticker symbol", value="^GSPC").strip().upper()
        vix_ticker = st.text_input("Volatility ticker", value="^VIX").strip().upper()

        st.markdown("---")
        st.markdown("### 📅 Data")
        lookback     = st.slider("Lookback (years)", 1, 10, 3)
        auto_refresh = st.checkbox("Auto-refresh (1hr)", value=False)
        if auto_refresh:
            st.cache_data.clear()

        st.markdown("---")
        st.markdown("### γ — Risk Aversion")
        st.markdown('<div class="info-box">w* = (μ−r) / (γσ²)<br>Higher γ → more conservative.<br>γ=4 is the dissertation baseline.</div>',
                    unsafe_allow_html=True)
        gamma = st.slider("Risk aversion (γ)", 1.0, 10.0, float(RISK_AVERSION_GAMMA), 0.5)

        st.markdown("---")
        st.markdown("### μ — Return Estimate Blend")
        st.markdown('<div class="info-box">mu_ann = α·μ_ST + (1−α)·μ_LT<br>Higher α → more weight on recent returns.</div>',
                    unsafe_allow_html=True)
        st_alpha = st.slider("Short-term weight (α)", 0.0, 1.0, 0.5, 0.05)
        lt_alpha = 1.0 - st_alpha

        st.markdown("---")
        st.markdown(f"Last updated: `{datetime.now().strftime('%Y-%m-%d %H:%M')}`")

    # ── Load models
    models, model_error = load_models()

    # ── Header
    ticker_display = ticker if ticker != "^GSPC" else "S&P 500 (^GSPC)"
    st.markdown("# HYBRID REGIME FRAMEWORK")
    st.markdown(f"*{ticker_display} · γ={gamma:.1f} · ST={st_alpha:.0%} · LT={lt_alpha:.0%}*")
    st.markdown("---")

    if model_error:
        st.error(f"⚠ {model_error}")
        st.info("Run `python dynamic_allocation.py` to train and save models, then refresh.")

    # ── Fetch data
    with st.spinner(f"Fetching {ticker}..."):
        px_series, vix_series, fetch_err = fetch_market_data(ticker, vix_ticker, lookback)
        if fetch_err or px_series is None or len(px_series) < 200:
            st.error(f"Failed to fetch data for {ticker}. Try a different ticker or longer lookback.")
            return

    macro_df, macro_err = fetch_macro_data(datetime.today().year - lookback - 1)
    if macro_err:
        st.warning("FRED macro data unavailable — using zero macro signal.")
        macro_df = None

    # ── Compute features
    with st.spinner("Computing features..."):
        X_df, sigma_ann, mu_ann, r_daily, w_hjb, ret = build_features(
            px_series, vix_series, macro_df,
            gamma=gamma, st_weight=st_alpha, lt_weight=lt_alpha
        )

    mu_st_disp = causal_ewm_mean(np.log(px_series).diff(), LOOKBACK_ST) * ANNUALIZATION
    mu_lt_disp = causal_ewm_mean(np.log(px_series).diff(), LOOKBACK_LT) * ANNUALIZATION

    # ── Run inference
    if models is not None:
        with st.spinner("Running ensemble inference..."):
            result = run_ensemble(X_df, models)
        w_ensemble, w_baseline = compute_allocation(result, sigma_ann, mu_ann, w_hjb, models, gamma)
    else:
        # Demo mode — uniform random probabilities
        n = len(X_df)
        np.random.seed(42)
        p_raw = np.random.dirichlet([1, 2, 3], size=n)
        result = pd.DataFrame({
            "p_bear": p_raw[:, 0], "p_neutral": p_raw[:, 1], "p_bull": p_raw[:, 2],
            "confidence": p_raw.max(axis=1), "conflict": 1 - p_raw.max(axis=1),
            "pred": p_raw.argmax(axis=1),
            "pred_name": [["bear","neutral","bull"][p] for p in p_raw.argmax(axis=1)],
        }, index=X_df.index)
        w_hjb_arr  = w_hjb.reindex(X_df.index).fillna(0)
        w_ensemble = pd.Series(w_hjb_arr.values * 0.9, index=X_df.index)
        w_baseline = w_hjb_arr

    # ── Latest snapshot
    latest        = result.iloc[-1]
    latest_price  = float(px_series.reindex(X_df.index).iloc[-1])
    prev_price    = float(px_series.reindex(X_df.index).iloc[-2])
    daily_chg     = (latest_price / prev_price - 1) * 100
    current_regime = latest["pred_name"]
    current_conf   = latest["confidence"]
    current_w      = float(w_ensemble.iloc[-1])
    current_w_base = float(w_baseline.iloc[-1])
    current_mu     = float(mu_ann.reindex(X_df.index).iloc[-1])
    current_sigma  = float(sigma_ann.reindex(X_df.index).iloc[-1])
    latest_date    = X_df.index[-1]

    # ── Top metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.metric(ticker if ticker != "^GSPC" else "S&P 500", f"{latest_price:,.2f}", f"{daily_chg:+.2f}%")
    with c2: st.metric("Current Regime", current_regime.upper())
    with c3: st.metric("Confidence", f"{current_conf:.1%}")
    with c4: st.metric("Ensemble Weight", f"{current_w:.1%}", f"{current_w - current_w_base:+.1%} vs HJB")
    with c5: st.metric("μ_ann (today)", f"{current_mu:.1%}")
    with c6: st.metric("σ_ann (today)", f"{current_sigma:.1%}")

    st.markdown("---")
    st.markdown(
        f'<span class="regime-badge {current_regime}">{current_regime.upper()}</span>'
        f'&nbsp;&nbsp;<span style="font-family:IBM Plex Mono;font-size:0.8rem;color:#8b949e;">'
        f'{latest_date.strftime("%d %b %Y")} · HJB w*={current_w_base:.1%} · γ={gamma:.1f} · α={st_alpha:.2f}</span>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # ── Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price & Regimes", "⚖️ Allocation", "📊 Signals", "🔬 Parameter Explorer"
    ])

    with tab1:
        st.plotly_chart(plot_price_with_regimes(px_series.reindex(X_df.index), result),
                        use_container_width=True)
        st.plotly_chart(plot_regime_probabilities(result), use_container_width=True)

    with tab2:
        eq_ens,  m_ens  = compute_performance(px_series, w_ensemble, r_daily)
        eq_base, m_base = compute_performance(px_series, w_baseline, r_daily)

        st.plotly_chart(plot_equity_curves(eq_ens, eq_base), use_container_width=True)
        st.plotly_chart(plot_weights(w_ensemble, w_baseline), use_container_width=True)

        st.markdown("#### Performance (displayed period)")
        perf_df = pd.DataFrame({
            "Strategy": ["Blended Ensemble", f"HJB Baseline (γ={gamma:.1f})"],
            "CAGR":   [f"{m_ens['CAGR']:.2%}",  f"{m_base['CAGR']:.2%}"],
            "Sharpe": [f"{m_ens['Sharpe']:.2f}", f"{m_base['Sharpe']:.2f}"],
            "Max DD": [f"{m_ens['MaxDD']:.2%}",  f"{m_base['MaxDD']:.2%}"],
        })
        st.dataframe(perf_df, hide_index=True, use_container_width=True)

    with tab3:
        st.plotly_chart(plot_confidence_conflict(result), use_container_width=True)

        st.markdown("#### Recent Regime History (last 20 days)")
        recent = result[["p_bear","p_neutral","p_bull","confidence","conflict","pred_name"]].tail(20).copy()
        recent.index = recent.index.strftime("%Y-%m-%d")
        recent.columns = ["P(Bear)","P(Neutral)","P(Bull)","Confidence","Conflict","Regime"]
        recent = recent.sort_index(ascending=False)
        for col in ["P(Bear)","P(Neutral)","P(Bull)","Confidence","Conflict"]:
            recent[col] = recent[col].apply(lambda x: f"{x:.3f}")
        st.dataframe(recent, use_container_width=True)

    with tab4:
        st.markdown("#### Parameter Explorer")
        st.markdown("Adjust γ and α in the sidebar to see how they affect the HJB weight and return estimate.")

        # γ sensitivity
        st.markdown("##### Risk Aversion Sensitivity")
        fig_g = go.Figure()
        excess_mu_s = (mu_ann - r_daily * ANNUALIZATION).reindex(X_df.index)
        sigma_s     = sigma_ann.reindex(X_df.index)
        for g, col in zip([2.0, 4.0, 6.0, 8.0], ["#22c55e","#58a6ff","#eab308","#ef4444"]):
            w_g = (excess_mu_s / (g * sigma_s**2)).clip(0,1).shift(1).fillna(0)
            fig_g.add_trace(go.Scatter(
                x=X_df.index, y=w_g.values,
                name=f"γ={g:.0f}" + (" ◄" if abs(g-gamma)<0.1 else ""),
                line=dict(color=col, width=2.2 if abs(g-gamma)<0.1 else 1.0,
                          dash="solid" if abs(g-gamma)<0.1 else "dot")
            ))
        layout_g = {**PLOTLY_LAYOUT}
        layout_g["yaxis"] = dict(tickformat=".0%", gridcolor="#21262d", linecolor="#30363d", showgrid=True)
        fig_g.update_layout(**layout_g,
                            title=dict(text="HJB Weight for Different Risk Aversion Levels", font=dict(size=12)),
                            height=280)
        st.plotly_chart(fig_g, use_container_width=True)

        # α sensitivity
        st.markdown("##### Return Estimate Blend Sensitivity")
        fig_m = go.Figure()
        for a, col in zip([0.0, 0.25, 0.5, 0.75, 1.0],
                          ["#ef4444","#eab308","#58a6ff","#22c55e","#a855f7"]):
            mu_a = a * mu_st_disp.reindex(X_df.index) + (1-a) * mu_lt_disp.reindex(X_df.index)
            fig_m.add_trace(go.Scatter(
                x=X_df.index, y=mu_a.values,
                name=f"α={a:.2f}" + (" ◄" if abs(a-st_alpha)<0.01 else ""),
                line=dict(color=col, width=2.2 if abs(a-st_alpha)<0.01 else 1.0,
                          dash="solid" if abs(a-st_alpha)<0.01 else "dot")
            ))
        fig_m.add_hline(y=0, line_color="#30363d", line_width=1)
        layout_m = {**PLOTLY_LAYOUT}
        layout_m["yaxis"] = dict(tickformat=".0%", gridcolor="#21262d", linecolor="#30363d", showgrid=True)
        fig_m.update_layout(**layout_m,
                            title=dict(text="Annualised Return Estimate for Different ST/LT Blends", font=dict(size=12)),
                            height=280)
        st.plotly_chart(fig_m, use_container_width=True)

        # Parameter snapshot
        st.markdown("##### Current Parameter Snapshot")
        snap_df = pd.DataFrame({
            "Parameter": ["γ","ST weight (α)","LT weight (1−α)","μ_ST today","μ_LT today",
                          "μ_ann today","σ_ann today","HJB w* today","Ensemble w today"],
            "Value": [
                f"{gamma:.1f}", f"{st_alpha:.2f}", f"{lt_alpha:.2f}",
                f"{float(mu_st_disp.reindex(X_df.index).iloc[-1]):.2%}",
                f"{float(mu_lt_disp.reindex(X_df.index).iloc[-1]):.2%}",
                f"{current_mu:.2%}", f"{current_sigma:.2%}",
                f"{current_w_base:.2%}", f"{current_w:.2%}",
            ]
        })
        st.dataframe(snap_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown(
        f'<div class="info-box">Ticker: {ticker} · Models trained on S&P 500 to 2015-01-01 · '
        'Ang &amp; Bekaert (2004) probability-weighted blending · '
        'Nystrup et al. (2017) HMM allocation framework</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()