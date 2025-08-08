import pandas as pd


def momentum_reversal(
        historical_data: pd.DataFrame,
        momentum_window: int = 20,
        reversal_window: int = 5,
        momentum_weight: float = 0.7
) -> pd.DataFrame:
    """
    Generate signals based on momentum and mean reversion.
    Long-term momentum combined with short-term mean reversion.
    
    Args:
        historical_data: Multi-index DataFrame with (symbol, timestamp) 
            index and OHLCV data
        momentum_window: Window for momentum calculation
        reversal_window: Window for mean reversion calculation
        momentum_weight: Weight given to momentum vs reversal (0 to 1)
        
    Returns:
        DataFrame with signals and intermediate calculations
    """
    # Create a copy to avoid modifying the input
    df = historical_data.copy()
    
    # Calculate signals for each symbol
    for symbol in df.index.get_level_values(0).unique():
        symbol_data = df.loc[symbol]
        
        # Calculate momentum component (longer-term trend)
        returns = symbol_data['close'].pct_change()
        df.loc[symbol, 'momentum'] = returns.rolling(
            window=momentum_window
        ).mean()
        
        # Calculate reversal component (short-term mean reversion)
        ma = symbol_data['close'].rolling(window=reversal_window).mean()
        pct_diff = (symbol_data['close'] - ma) / ma
        # Negative of deviation from mean
        df.loc[symbol, 'reversal'] = -pct_diff
        
        # Combine signals
        df.loc[symbol, 'raw_signal'] = (
            momentum_weight * df.loc[symbol, 'momentum'] +
            (1 - momentum_weight) * df.loc[symbol, 'reversal']
        )
        
        # Normalize to [-1, 1] range
        df.loc[symbol, 'signal'] = (
            df.loc[symbol, 'raw_signal'] / 
            df.loc[symbol, 'raw_signal'].abs().rolling(
                window=momentum_window
            ).max().fillna(1)
        ).fillna(0)
    
    return df 