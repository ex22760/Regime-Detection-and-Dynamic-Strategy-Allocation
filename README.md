# Real-Time Market Regime Detection and Dynamic Strategy Allocation Using Machine Learning

**Sujin Subanthran — MEng Engineering Mathematics, University of Bristol, 2026**

---

## Overview

This repository contains the full implementation of a hybrid ensemble framework for real-time market regime detection and dynamic portfolio allocation. The framework combines four complementary models — a Gaussian Mixture Model (GMM), Hidden Markov Model (HMM), Random Forest (RF), and Long Short-Term Memory network (LSTM) — within a stacked architecture that produces probabilistic regime estimates and adjusts portfolio allocation in real time.

The system is evaluated against a Hamilton-Jacobi-Bellman (HJB) stochastic control baseline over a strictly out-of-sample test period from January 2015 to February 2026.

---

## Key Findings

| Metric | Ensemble | Baseline (HJB) |
|---|---|---|
| Annual return (CAGR) | 3.99% | 4.38% |
| Risk-adjusted return (Sharpe) | **0.65** | 0.62 |
| Worst loss (Max Drawdown) | **−10.84%** | −14.60% |
| Transaction cost | 5.17 bps | 4.34 bps |

- **72% reduction** in mean equity exposure during confirmed bear days (0.070 vs 0.249 baseline), confirming the risk management mechanism operates correctly
- The Sharpe improvement is directional but not statistically significant at conventional levels (Jobson–Korkie z ≈ 1.2, p ≈ 0.23) given the 11-year test window
- HMM soft probabilities rank as the **second most important feature** in the Random Forest (Gini importance 0.179), validating the stacked architecture
- **Live validation**: the ensemble detected the April 2025 Liberation Day tariff shock with a one-day lag — P(bear) jumped from 0.026 to 0.643 on 3 April 2025

---

## Repository Structure

```
├── Models/
│   ├── baselinemodel.py           # HJB stochastic control baseline
│   ├── dashboard.py               # Streamlit live inference dashboard
│   ├── dynamicstratallocation.py  # Full ensemble backtest and dynamic allocation
│   ├── hybrid.py                  # Ensemble combination, confidence and conflict scoring
│   ├── loading_data_SP500.py      # Data download and feature engineering pipeline
│   ├── supervisedmodel.py         # Random Forest and LSTM training and evaluation
│   ├── unsupervisedmodel.py       # GMM and HMM training and evaluation
│   ├── lstm_model.pt              # Serialised trained LSTM weights
│   ├── lstm_input_size.txt        # Input dimension record for model loading
│   └── regime_models.pkl          # Serialised GMM, HMM and Random Forest models
└── README.md
```

---

## Running the Streamlit Dashboard

The dashboard fetches live market data, constructs features, runs the full four-model ensemble inference pipeline, and produces a regime classification and portfolio weight recommendation — all in under five seconds on a standard laptop CPU.

### 1. Prerequisites

Python 3.9 or later is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit yfinance pandas numpy scikit-learn torch hmmlearn matplotlib plotly
```

### 2. Clone the repository

```bash
git clone https://github.com/ex22760/Regime-Detection-and-Dynamic-Strategy-Allocation.git
cd Regime-Detection-and-Dynamic-Strategy-Allocation
```

### 3. Launch the dashboard

```bash
streamlit run Models/dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

### 4. Using the dashboard

Once running, you can:

- **Change the asset** — enter any Yahoo Finance ticker (default: `^GSPC` for S&P 500)
- **Adjust risk aversion** — use the γ slider to change the HJB baseline conservativeness (γ = 4 is the dissertation baseline)
- **Change the lookback window** — controls how much historical data is used for feature construction
- **Enable auto-refresh** — the dashboard will re-fetch and re-run inference every hour

The dashboard displays:
- Current regime classification (Bear / Neutral / Bull) with ensemble confidence score
- Recommended portfolio weight vs the HJB baseline weight
- Price chart with regime shading
- Stacked probability chart showing P(Bear), P(Neutral), P(Bull) over time
- Model conflict score and confidence threshold indicator

> **Note:** The models are trained on data up to January 2015. The dashboard operates in a strictly out-of-sample mode — no retraining occurs at runtime.

---

## Reproducing the Results

To reproduce the full backtest results from the dissertation:

```bash
# Download data and construct features
python Models/loading_data_SP500.py

# Train unsupervised models (GMM and HMM)
python Models/unsupervisedmodel.py

# Train supervised models (Random Forest and LSTM)
python Models/supervisedmodel.py

# Run the hybrid ensemble combination
python Models/hybrid.py

# Run the HJB baseline backtest
python Models/baselinemodel.py

# Run the full ensemble dynamic allocation backtest
python Models/dynamicstratallocation.py
```

All scripts use `random_state=42` and `torch.manual_seed(42)` for reproducibility.

---

## Model Architecture

```
Raw features (S&P 500 returns, VIX, CPI, unemployment, Fed Funds Rate)
        │
        ├── GMM  (3 Gaussian components, EM estimation)     ─┐
        ├── HMM  (3 states, Baum-Welch, causal forward filter) ─┤
        │                                                      │  Soft probability
        ├── Random Forest (depth=6, 300 trees, Platt-scaled) ─┤  vectors → weighted
        └── LSTM (hidden=32, dropout=0.2, seq=40)            ─┘  arithmetic mean
                                                               │
                                               Ensemble weights (softmax over
                                               validation macro-F1):
                                               GMM 0.312 | LSTM 0.252
                                               HMM 0.222 | RF  0.214
                                                               │
                                         Confidence scoring + conflict detection
                                                               │
                                          Regime-conditional HJB weight overlay
                                                               │
                                         Conflict-adaptive no-trade band filter
                                                               │
                                                   Portfolio weight wₜ ∈ [0, 1]
```

---

## Data Sources

| Series | Source | Frequency |
|---|---|---|
| S&P 500 price (OHLCV) | Yahoo Finance (`^GSPC`) | Daily |
| VIX implied volatility | Yahoo Finance (`^VIX`) | Daily |
| CPI inflation | FRED | Monthly → daily (21-day lag) |
| Unemployment rate | FRED | Monthly → daily (7-day lag) |
| Federal Funds Rate | FRED | Daily (1-day lag) |
| NBER recession dates | FRED (`USREC`) | Monthly |

All macroeconomic series are forward-filled to daily frequency after applying their empirical publication lags to prevent lookahead bias.

---

## Acknowledgements

This project was completed as part of the MEng Engineering Mathematics dissertation at the University of Bristol.

I would like to sincerely thank my supervisors:

- **Dr Yani Berdeni** (University of Bristol) — for guidance on the mathematical framework, stochastic control formulation, and dissertation structure throughout the project
- **Dr Nicolas** (University of Exeter) — for support on the machine learning methodology, ensemble design, and practical implementation aspects of the framework

Their expertise and feedback were invaluable in shaping both the theoretical rigour and empirical robustness of this work.

---

## Citation

If you use this code or framework in your own work, please cite:

```
Subanthran, S. (2026). Real-Time Market Regime Detection and Dynamic Strategy
Allocation Using Machine Learning. MEng Dissertation, University of Bristol.
```

---

## Licence

This repository is made available for academic and research purposes. Please contact the author before using any component in a commercial context.
