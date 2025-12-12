import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def get_data(tickers, start_date, end_date):
    """
    Fetches daily closing prices for a list of tickers.
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        return data
    except Exception as e:
        return pd.DataFrame()

def check_cointegration(series_a, series_b):
    """
    Performs the Augmented Engle-Granger two-step cointegration test.
    Returns: p-value, hedge_ratio, spread series
    """
    # 1. Calculate Hedge Ratio using OLS
    X = sm.add_constant(series_b) 
    model = sm.OLS(series_a, X).fit()
    hedge_ratio = model.params.iloc[1]
    
    # 2. Calculate the Spread
    spread = series_a - (hedge_ratio * series_b)
    
    # 3. Perform ADF test
    score, pvalue, _ = coint(series_a, series_b)
    
    return pvalue, hedge_ratio, spread

def calculate_zscore(spread, window=30):
    """
    Calculates the Rolling Z-Score.
    """
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    z_score = (spread - mean) / std
    return z_score

def calculate_metrics(daily_returns):
    """
    Computes Sharpe Ratio and Max Drawdown.
    """
    # Sharpe Ratio (Assuming 252 trading days, Risk-Free Rate = 0)
    if daily_returns.std() == 0:
        return 0, 0
    
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    # Max Drawdown
    cumulative = (1 + daily_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    return sharpe, max_drawdown

def run_backtest(data, window=30):
    """
    Runs the mean reversion strategy logic.
    """
    # 1. Check Cointegration
    p_value, hedge_ratio, spread = check_cointegration(data.iloc[:, 0], data.iloc[:, 1])
    
    # 2. Calculate Z-Score
    z_score = calculate_zscore(spread, window)
    
    # 3. Define Strategy Logic
    signals = pd.DataFrame(index=data.index)
    signals['z'] = z_score
    signals['positions'] = 0
    
    # Entry Logic: Short Spread (Z > 2), Long Spread (Z < -2)
    signals.loc[signals['z'] > 2, 'positions'] = -1 
    signals.loc[signals['z'] < -2, 'positions'] = 1 
    
    # --- UPGRADE: STOP LOSS ---
    # If spread diverges too much (> 4 sigma), exit position to prevent infinite loss
    signals.loc[signals['z'] > 4, 'positions'] = 0
    signals.loc[signals['z'] < -4, 'positions'] = 0
    
    # 4. Calculate Returns
    spread_returns = spread.pct_change()
    strategy_returns = signals['positions'].shift(1) * spread_returns
    strategy_returns.fillna(0, inplace=True)
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    return cumulative_returns, strategy_returns, p_value, z_score, spread