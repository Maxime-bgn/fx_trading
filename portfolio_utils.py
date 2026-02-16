"""
portfolio_utils.py
Utility functions extracted from heatmap.py, markowitz.py, returns.py, and risque.py
"""

import pandas as pd
import numpy as np
import glob
import os
from scipy.optimize import minimize
from scipy.stats import norm

TRADING_DAYS = 252
MAX_WEIGHT = 0.20

# ========================================
# FROM heatmap.py
# ========================================

def load_fx_prices():
    """Load all FX price data from *=X.csv files."""
    prices = {}
    
    for file in glob.glob("*=X.csv"):
        pair = os.path.basename(file).replace("=X.csv", "")
        
        try:
            df = pd.read_csv(
                file,
                sep=";",
                decimal=",",
                index_col=0,
                parse_dates=True
            )
            
            if "Close" not in df.columns:
                continue
            
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            prices[pair] = df["Close"]
        except Exception:
            continue
    
    prices_df = pd.DataFrame(prices)
    prices_df = prices_df.dropna()
    
    return prices_df


# ========================================
# FROM markowitz.py
# ========================================

def load_fx_returns():
    """Load returns data from fx_returns.csv."""
    try:
        returns_df = pd.read_csv(
            "fx_returns.csv",
            sep=";",
            decimal=",",
            index_col=0,
            parse_dates=True
        )
        returns_df = returns_df.dropna(axis=1, thresh=200)
        returns_df = returns_df.dropna()
        return returns_df
    except Exception:
        return None


def port_return(w, mu):
    """Calculate portfolio return."""
    return np.dot(w, mu)


def port_vol(w, Sigma):
    """Calculate portfolio volatility."""
    return np.sqrt(w.T @ Sigma @ w)


def neg_sharpe(w, mu, Sigma):
    """Negative Sharpe ratio (for minimization)."""
    return -port_return(w, mu) / port_vol(w, Sigma)


def compute_markowitz_optimization(returns_df, n_frontier_points=30):
    """
    Compute Markowitz optimization and efficient frontier.
    
    Returns:
        dict with keys: mu, Sigma, w_minvar, w_maxsharpe, 
                       frontier_vol, frontier_ret, column_names
    """
    mu = returns_df.mean() * TRADING_DAYS
    Sigma = returns_df.cov() * TRADING_DAYS
    
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0, MAX_WEIGHT)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    
    # Minimum Variance
    res_minvar = minimize(
        lambda w: port_vol(w, Sigma), w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    w_minvar = res_minvar.x
    
    # Maximum Sharpe
    res_maxsharpe = minimize(
        lambda w: neg_sharpe(w, mu, Sigma), w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    w_maxsharpe = res_maxsharpe.x
    
    # Efficient Frontier
    target_returns = np.linspace(
        port_return(w_minvar, mu),
        port_return(w_maxsharpe, mu),
        n_frontier_points
    )
    
    frontier_vol = []
    frontier_ret = []
    
    for R in target_returns:
        cons = (
            constraints,
            {"type": "eq", "fun": lambda w, R=R: port_return(w, mu) - R}
        )
        
        res = minimize(
            lambda w: port_vol(w, Sigma), w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons
        )
        
        if res.success:
            frontier_vol.append(port_vol(res.x, Sigma))
            frontier_ret.append(port_return(res.x, mu))
    
    return {
        'mu': mu,
        'Sigma': Sigma,
        'w_minvar': w_minvar,
        'w_maxsharpe': w_maxsharpe,
        'frontier_vol': frontier_vol,
        'frontier_ret': frontier_ret,
        'column_names': returns_df.columns
    }


# ========================================
# FROM returns.py
# ========================================

def compute_fx_returns(start_date="2020-01-01"):
    """
    Compute log returns from price CSVs.
    Returns a DataFrame of returns.
    """
    returns = {}
    
    for file in glob.glob("*=X.csv"):
        pair = file.replace("=X.csv", "")
        
        df = pd.read_csv(
            file,
            sep=";",
            decimal=",",
            index_col=0,
            parse_dates=True
        )
        
        if "Close" not in df.columns:
            continue
        
        prices = pd.to_numeric(df["Close"], errors="coerce").dropna()
        prices = prices[prices.index >= start_date]
        
        if len(prices) < 200:
            continue
        
        r = np.log(prices / prices.shift(1)).dropna()
        returns[pair] = r
    
    returns_df = pd.DataFrame(returns)
    
    # Save to CSV
    returns_df.to_csv("fx_returns.csv", sep=";", decimal=",")
    print("[OK] fx_returns.csv saved")
    
    return returns_df


# ========================================
# FROM risque.py
# ========================================

def max_drawdown_signed(r):
    """Calculate signed max drawdown (negative)."""
    cum = np.exp(r.cumsum())
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def max_drawdown_positive(r):
    """Calculate positive max drawdown."""
    cum = np.exp(r.cumsum())
    peak = cum.cummax()
    dd = (peak - cum) / peak
    return dd.max()


def semi_deviation(r, target=0.0, trading_days=252):
    """Calculate semi-deviation (downside risk)."""
    downside = r[r < target]
    return downside.std() * np.sqrt(trading_days)


def var_gaussian(r, alpha=0.95):
    """Calculate Value at Risk (VaR) using Gaussian distribution."""
    mu = r.mean()
    sigma = r.std()
    z = norm.ppf(alpha)
    return -(mu - z * sigma)


def compute_risk_metrics(returns_df, trading_days=252):
    """
    Compute comprehensive risk metrics for all pairs.
    Returns a DataFrame with risk statistics.
    """
    stats = {}
    
    for pair in returns_df.columns:
        r = returns_df[pair].dropna()
        if len(r) < 200:
            continue
        
        ret_ann = r.mean() * trading_days
        vol_ann = r.std() * np.sqrt(trading_days)
        sharpe = ret_ann / vol_ann if vol_ann > 0 else np.nan
        var_95 = var_gaussian(r)
        
        semi = semi_deviation(r, trading_days=trading_days)
        sortino = ret_ann / semi if semi > 0 else np.nan
        
        mdd_signed = max_drawdown_signed(r)
        mdd_pos = max_drawdown_positive(r)
        
        calmar = ret_ann / mdd_pos if mdd_pos > 0 else np.nan
        
        stats[pair] = {
            "Return_ann": ret_ann,
            "Vol_ann": vol_ann,
            "Sharpe": sharpe,
            "Semi_dev": semi,
            "Sortino": sortino,
            "VaR_95_daily": var_95,
            "MaxDD_signed": mdd_signed,
            "MaxDD_positive": mdd_pos,
            "Calmar": calmar
        }
    
    risk_df = pd.DataFrame(stats).T.sort_values("Sortino", ascending=False)
    
    # Save to CSV
    risk_df.to_csv("fx_risk_metrics.csv", sep=";", decimal=",")
    print("[OK] fx_risk_metrics.csv saved")
    
    return risk_df