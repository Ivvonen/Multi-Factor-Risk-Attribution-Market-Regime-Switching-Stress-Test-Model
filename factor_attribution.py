import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture
import plotly.graph_objects as go

# --- 1. QUANTITATIVE & ML RISK CLASSES ---

class FundFactorRiskEngine:
    def __init__(self, fund_returns: np.ndarray, factor_dataframe: pd.DataFrame):
        """
        Decomposes investment fund returns into systematic risk factor exposures.
        """
        self.fund_rets = fund_returns
        self.factors = factor_dataframe
        
    def calculate_fama_french_attribution(self):
        """Runs an OLS regression to isolate Fund Alpha and Factor Betas."""
        # Calculate excess fund returns (Fund Return - Risk Free Rate)
        excess_fund_ret = self.fund_rets - self.factors['RF'].values
        
        # Define independent variables (Market, Size, Value factors)
        X = self.factors[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)  # Adds the intercept (Alpha)
        
        # Execute OLS Regression
        model = sm.OLS(excess_fund_ret, X).fit()
        
        return {
            "Alpha (Idiosyncratic Return)": model.params['const'] * 252,  # Annualized
            "Market Beta (Mkt-RF)": model.params['Mkt-RF'],
            "Size Exposure (SMB)": model.params['SMB'],
            "Value Exposure (HML)": model.params['HML'],
            "R-Squared (Systematic Fit)": model.rsquared
        }
        
    def classify_market_regimes(self, market_returns: np.ndarray, n_regimes=2):
        """Uses a Gaussian Mixture Model to separate returns into high/low volatility regimes."""
        rets_reshaped = market_returns.reshape(-1, 1)
        
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        gmm.fit(rets_reshaped)
        
        # Predict the hidden regime state for every single day
        regime_labels = gmm.predict(rets_reshaped)
        
        # Identify the high volatility crisis regime index
        regime_vols = [np.std(market_returns[regime_labels == i]) for i in range(n_regimes)]
        crisis_regime_idx = np.argmax(regime_vols)
        
        return regime_labels, crisis_regime_idx

# --- 2. STREAMLIT INTERFACE & DYNAMIC CONTROL DECK ---

st.set_page_config(page_title="Fund Factor Risk Attribution", layout="wide")
st.title("🕵️ Multi-Factor Risk Attribution & Regime-Switching Engine")
st.markdown("Decompose active fund performance into systematic factor betas and identify volatile market cycles dynamically.")

# 🕹️ Sidebar UI Widgets for Interactive Attributes
st.sidebar.header("🕹️ Dynamic Portfolio & Macro Controls")

# Input 1: Change Data Timeline
trading_days = st.sidebar.slider("Historical Lookback Window (Trading Days)", 100, 1000, 500, step=50)

st.sidebar.subheader("Fund Manager Portfolio Style Biases")
# Input 2: Dynamic Shift in Factor Exposures (Simulating user changing portfolio style)
true_beta = st.sidebar.slider("Target Market Beta (β)", 0.0, 2.5, 1.2, step=0.1)
true_smb = st.sidebar.slider("Small-Cap Tilt Bias (SMB)", -1.0, 1.5, 0.4, step=0.1)
true_hml = st.sidebar.slider("Value Style Tilt Bias (HML)", -1.0, 1.5, -0.2, step=0.1)
true_alpha_annual = st.sidebar.number_input("Manager Skill Target (Annualized Alpha %)", value=3.5, step=0.5) / 100

st.sidebar.subheader("Machine Learning Clustering Configuration")
# Input 3: Change ML Hyperparameters
regime_count = st.sidebar.selectbox("GMM Target Regimes Cluster Count", [2, 3], index=0)

# --- 3. DYNAMIC DATA SIMULATION ENGINE ---
# Generates realistic macroeconomic data tracks based on sidebar inputs
np.random.seed(42)
mkt_rf = np.random.normal(0.0004, 0.012, trading_days)
smb = np.random.normal(0.0001, 0.006, trading_days)
hml = np.random.normal(0.0001, 0.006, trading_days)
rf = np.full(trading_days, 0.00015)

# Inject structural volatility clusters (representing real-world crisis states)
crisis_start = int(trading_days * 0.75)
mkt_rf[crisis_start:] = mkt_rf[crisis_start:] * 3.5  # Sudden market crash regime shift

# Generate fund return using user-configured sidebar weights + idiosyncratic tracking noise
true_daily_alpha = true_alpha_annual / 252
idiosyncratic_noise = np.random.normal(0, 0.004, trading_days)
fund_returns = rf + true_daily_alpha + (true_beta * mkt_rf) + (true_smb * smb) + (true_hml * hml) + idiosyncratic_noise

factors_df = pd.DataFrame({'Mkt-RF': mkt_rf, 'SMB': smb, 'HML': hml, 'RF': rf})

# --- 4. ENGINE EXECUTION LOOP ---
engine = FundFactorRiskEngine(fund_returns=fund_returns, factor_dataframe=factors_df)
metrics = engine.calculate_fama_french_attribution()
regime_labels, crisis_idx = engine.classify_market_regimes(mkt_rf, n_regimes=regime_count)

# --- 5. RENDER DYNAMIC EXECUTIVE KPI RESULTS ---
st.markdown("### 📊 Live Quantitative Attribution Report Card")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Extracted Annualized Alpha", f"{metrics['Alpha (Idiosyncratic Return)']*100:+.2f}%")
col2.metric("Calculated Market Beta (β)", f"{metrics['Market Beta (Mkt-RF)']:.3f}")
col3.metric("Size Style Exposure (SMB)", f"{metrics['Size Exposure (SMB)']:.3f}")
col4.metric("Value Style Exposure (HML)", f"{metrics['Value Exposure (HML)']:.3f}")

# Model Explainer
st.markdown(f"**Regression R-Squared Summary Fit:** `{metrics['R-Squared (Systematic Fit)']*100:.2f}%` of this investment fund's return volatility is fully explained by systematic style factors.")
st.progress(metrics['R-Squared (Systematic Fit)'])

st.markdown("---")

# --- 6. RENDER DYNAMIC PLOTLY REGIME GRAPH ---
st.subheader("📈 Machine Learning Regime-Switching Factor Mapping")
st.markdown("This scatter plot automatically shifts color zones based on the Gaussian Mixture Model classifying normal vs high-volatility structural cycles.")

# Build interactive plotly grid
fig = go.Figure()

# Loop through each classified regime cluster to color-code scatter points
for regime in range(regime_count):
    mask = (regime_labels == regime)
    name = "🚨 Crisis / High Volatility State" if regime == crisis_idx else f"🟢 Normal Market Regime (State {regime})"
    color = '#d62728' if regime == crisis_idx else ['#1f77b4', '#2ca02c'][min(regime, 1)]
    
    fig.add_trace(go.Scatter(
        x=mkt_rf[mask] * 100,
        y=fund_returns[mask] * 100,
        mode='markers',
        name=name,
        marker=dict(size=8, color=color, opacity=0.7)
    ))

fig.update_layout(
    xaxis_title="Market Excess Returns (%)",
    yaxis_title="Active Fund Returns (%)",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    margin=dict(l=20, r=20, t=20, b=20),
    height=450
)

st.plotly_chart(fig, use_container_width=True)
