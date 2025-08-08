import numpy as np
import pandas as pd
import pandas_ta as ta


def rsi_reversion(
        historical_data: pd.DataFrame,
        rsi_period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        smoothing: int = 2
) -> pd.DataFrame:
    """
    Generate signals based on RSI mean reversion.
    Short when RSI is high (overbought) and long when RSI is low (oversold).
    Args:
        historical_data: Multi-index DataFrame with (symbol, timestamp) 
            index and OHLCV data
        rsi_period: Period for RSI calculation
        overbought: RSI level considered overbought
        oversold: RSI level considered oversold
        smoothing: Window for signal smoothing
    Returns:
        DataFrame with signals and RSI values
    """
    # Create a copy to avoid modifying the input
    df = historical_data.copy()
    
    # Ensure numeric data types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(np.float64)
    
    # Initialize new columns
    df['rsi'] = np.nan
    df['raw_signal'] = 0.0
    df['signal'] = 0.0
    
    # Group by symbol to calculate RSI for each
    for symbol in df.index.get_level_values(0).unique():
        # Get symbol data and ensure it's properly indexed
        symbol_data = df.loc[symbol].copy()
        symbol_data = symbol_data.sort_index()
        
        # Calculate RSI using float64 values
        close_prices = symbol_data['close'].values
        print(close_prices)
        rsi_values = ta.rsi(pd.Series(close_prices), length=int(rsi_period)).to_numpy()
        rsi_values = np.nan_to_num(rsi_values)
        # Generate raw signals (-1 for overbought, 1 for oversold)
        raw_signal = np.zeros_like(close_prices, dtype=np.float64)
        mask = ~np.isnan(rsi_values)
        raw_signal[mask & (rsi_values > overbought)] = -1.0  # Short
        raw_signal[mask & (rsi_values < oversold)] = 1.0     # Long
        
        # Store RSI and raw signals
        df.loc[symbol, 'rsi'] = rsi_values
        df.loc[symbol, 'raw_signal'] = raw_signal
        
        # Apply smoothing
        if smoothing > 1:
            signal = (
                pd.Series(raw_signal, index=symbol_data.index)
                .rolling(window=smoothing, center=True)
                .mean()
                .fillna(0)
            )
            df.loc[symbol, 'signal'] = signal.values
        else:
            df.loc[symbol, 'signal'] = raw_signal
    
    return df
