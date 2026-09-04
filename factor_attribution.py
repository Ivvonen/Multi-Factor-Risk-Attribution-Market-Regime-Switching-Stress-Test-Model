import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.mixture import GaussianMixture

class FundFactorRiskEngine:
    def __init__(self, fund_returns: np.ndarray, factor_dataframe: pd.DataFrame):
        """
        Decomposes investment fund returns into systematic risk factor exposures.
        factor_dataframe must contain columns: 'Mkt-RF', 'SMB', 'HML', 'RF'
        """
        self.fund_rets = fund_returns
        self.factors = factor_dataframe
        
    def calculate_fama_french_attribution(self):
        """Runs an OLS regression to isolate Fund Alpha and Factor Betas."""
        # Calculate excess fund returns (Fund Return - Risk Free Rate)
        excess_fund_ret = self.fund_rets - self.factors['RF'].values
        
        # Define independent variables (Market, Size, Value factors)
        X = self.factors[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X) # Adds the intercept (Alpha)
        
        # Execute OLS Regression
        model = sm.OLS(excess_fund_ret, X).fit()
        
        return {
            "Alpha (Idiosyncratic Return)": model.params['const'] * 252, # Annualized
            "Market Beta (Mkt-RF)": model.params['Mkt-RF'],
            "Size Exposure (SMB)": model.params['SMB'],
            "Value Exposure (HML)": model.params['HML'],
            "R-Squared (Systematic Fit)": model.rsquared
        }
        
    def classify_market_regimes(self, market_returns: np.ndarray, n_regimes=2):
        """Uses a Gaussian Mixture Model to separate returns into high/low volatility regimes."""
        # Reshape data for scikit-learn
        rets_reshaped = market_returns.reshape(-1, 1)
        
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        gmm.fit(rets_reshaped)
        
        # Predict the hidden regime state for every single day
        regime_labels = gmm.predict(rets_reshaped)
        
        # Identify the high volatility crisis regime index
        regime_vols = [np.std(market_returns[regime_labels == i]) for i in range(n_regimes)]
        crisis_regime_idx = np.argmax(regime_vols)
        
        return regime_labels, crisis_regime_idx

st.subheader("Fund Risk Factor Decomposition & Alpha Isolation")
st.markdown("Decompose active fund manager performance into systematic factor betas using multi-factor linear regressions.")

if st.button("Run Fama-French Risk Audit"):
    
    # 1. GENERATE OR FETCH YOUR DATA VECTORS FIRST (Example structural placeholders)
    # In production, these should be your real historical returns and factor dataframes
    np.random.seed(42)
    mock_fund_returns = np.random.normal(0.0005, 0.01, 252)
    mock_factors = pd.DataFrame({
        'Mkt-RF': np.random.normal(0.0004, 0.01, 252),
        'SMB': np.random.normal(0.0001, 0.005, 252),
        'HML': np.random.normal(0.0001, 0.005, 252),
        'RF': np.full(252, 0.0001)
    })
    
    # 2. INSTANTIATE THE ENGINE (This fixes the NameError)
    engine = FundFactorRiskEngine(fund_returns=mock_fund_returns, factor_dataframe=mock_factors)
    
    # 3. NOW CALL THE ENGINE METHODS DYNAMICALLY
    metrics = engine.calculate_fama_french_attribution()
    
    # 4. Display institutional risk report card metrics
    st.write(f"**Annualized Fund Alpha:** `{metrics['Alpha (Idiosyncratic Return)']*100:.2f}%`")
    st.write(f"**Systematic Market Beta (β):** `{metrics['Market Beta (Mkt-RF)']:.2f}`")
    st.write(f"**Size Style Tilt (SMB):** `{metrics['Size Exposure (SMB)']:.2f}`")
    st.write(f"**Value Style Tilt (HML):** `{metrics['Value Exposure (HML)']:.2f}`")
    st.progress(metrics['R-Squared (Systematic Fit)'])
