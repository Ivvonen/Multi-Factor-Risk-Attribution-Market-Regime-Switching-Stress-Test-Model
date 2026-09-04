import numpy as np
import pandas as pd
import statsmodels.api as sm
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
