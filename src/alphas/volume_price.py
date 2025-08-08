import numpy as np
import pandas as pd


def volume_price_trend(
        historical_data: pd.DataFrame,
        volume_window: int = 10,
        price_window: int = 20,
        volume_weight: float = 0.5
) -> pd.DataFrame:
    """
    Generate signals based on volume and price trends.
    Combines volume trend with price momentum.
    
    Args:
        historical_data: Multi-index DataFrame with (symbol, timestamp) 
            index and OHLCV data
        volume_window: Window for volume trend calculation
        price_window: Window for price trend calculation
        volume_weight: Weight given to volume vs price trend (0 to 1)
        
    Returns:
        DataFrame with signals and intermediate calculations
    """
    # Create a copy to avoid modifying the input
    df = historical_data.copy()
    
    # Calculate signals for each symbol
    for symbol in df.index.get_level_values(0).unique():
        symbol_data = df.loc[symbol]
        
        # Calculate volume trend
        vol_ma = symbol_data['volume'].rolling(window=volume_window).mean()
        vol_ratio = symbol_data['volume'] / vol_ma
        df.loc[symbol, 'volume_signal'] = vol_ratio - 1  # Center around 0
        
        # Calculate price trend
        returns = symbol_data['close'].pct_change()
        df.loc[symbol, 'price_signal'] = returns.rolling(
            window=price_window
        ).mean()
        
        # Calculate high-low range expansion/contraction
        high_low_range = (
            (symbol_data['high'] - symbol_data['low']) / 
            symbol_data['close']
        )
        df.loc[symbol, 'range_signal'] = high_low_range.rolling(
            window=price_window
        ).mean()
        
        # Combine signals
        df.loc[symbol, 'raw_signal'] = (
            volume_weight * df.loc[symbol, 'volume_signal'] * 
            np.sign(df.loc[symbol, 'price_signal']) +
            (1 - volume_weight) * df.loc[symbol, 'price_signal'] * 
            df.loc[symbol, 'range_signal']
        )
        
        # Normalize to [-1, 1] range
        df.loc[symbol, 'signal'] = (
            df.loc[symbol, 'raw_signal'] / 
            df.loc[symbol, 'raw_signal'].abs().rolling(
                window=max(volume_window, price_window)
            ).max().fillna(1)
        ).fillna(0)
    
    return df 