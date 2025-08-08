import numpy as np
import pandas as pd
import pandas_ta as ta

def trend_momentum(
        historical_data: pd.DataFrame,
        fast_ma: int = 10,
        slow_ma: int = 30,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        smoothing: int = 2
) -> pd.DataFrame:
    """
    Generate signals based on trend and momentum indicators.
    Combines:
    1. Moving Average Crossover (trend)
    2. RSI (mean reversion)
    3. MACD (momentum)
    
    Args:
        historical_data: Multi-index DataFrame with (symbol, timestamp) 
            index and OHLCV data
        fast_ma: Fast moving average period
        slow_ma: Slow moving average period
        rsi_period: RSI calculation period
        macd_fast: MACD fast period
        macd_slow: MACD slow period
        macd_signal: MACD signal line period
        rsi_overbought: RSI overbought level
        rsi_oversold: RSI oversold level
        smoothing: Signal smoothing window
    
    Returns:
        DataFrame with signals and indicator values
    """
    df = historical_data.copy()
    
    # Initialize indicator columns with NaN
    indicator_cols = {
        'fast_ma': np.float64,
        'slow_ma': np.float64,
        'rsi': np.float64,
        'macd': np.float64,
        'macd_signal': np.float64,
        'macd_hist': np.float64,
        'raw_signal': np.float64,
        'signal': np.float64
    }
    
    for col, dtype in indicator_cols.items():
        # df[col] = pd.Series(np.nan, index=df.index, dtype=dtype)
        df[col] = 0.0
    
    for symbol in df.index.get_level_values(0).unique():
        symbol_data = df.loc[symbol].copy()
        close_prices = symbol_data['close'].to_numpy(dtype=np.float64)
        
        try:
            # Calculate indicators (pandas-ta)
            s = pd.Series(close_prices)
            fast = ta.ema(s, length=int(fast_ma)).to_numpy()
            slow = ta.ema(s, length=int(slow_ma)).to_numpy()
            rsi = ta.rsi(s, length=int(rsi_period)).to_numpy()
            macd_df = ta.macd(s, fast=int(macd_fast), slow=int(macd_slow), signal=int(macd_signal))
            macd_line = macd_df[macd_df.columns[0]].fillna(0).to_numpy()
            signal_line = macd_df[macd_df.columns[1]].fillna(0).to_numpy()
            macd_hist = macd_df[macd_df.columns[2]].fillna(0).to_numpy()
            
            # Fill NaN values with appropriate initial values
            fast_filled = pd.Series(fast).fillna(close_prices[0]).values
            slow_filled = pd.Series(slow).fillna(close_prices[0]).values
            rsi_filled = pd.Series(rsi).fillna(50).values  # Neutral RSI
            macd_hist_filled = pd.Series(macd_hist).fillna(0).values  # No momentum
            
            # Generate signals
            ma_signal = np.where(fast_filled > slow_filled, 1, -1)
            
            rsi_signal = np.zeros_like(rsi_filled)
            rsi_signal = np.where(rsi_filled < rsi_oversold, 1, rsi_signal)
            rsi_signal = np.where(rsi_filled > rsi_overbought, -1, rsi_signal)
            
            momentum_signal = np.sign(macd_hist_filled)
            
            # Combine signals with weights
            raw_signal = (
                0.4 * ma_signal +
                0.3 * rsi_signal +
                0.3 * momentum_signal
            )
            
            # Store results with MultiIndex
            idx = pd.MultiIndex.from_product([[symbol], symbol_data.index])
            df.loc[idx, 'fast_ma'] = fast_filled
            df.loc[idx, 'slow_ma'] = slow_filled
            df.loc[idx, 'rsi'] = rsi_filled
            df.loc[idx, 'macd'] = pd.Series(macd_line).fillna(0).values
            df.loc[idx, 'macd_signal'] = pd.Series(signal_line).fillna(0).values
            df.loc[idx, 'macd_hist'] = macd_hist_filled
            df.loc[idx, 'raw_signal'] = raw_signal
            
            # Apply smoothing if requested
            if smoothing > 1:
                signal = pd.Series(raw_signal)\
                    .rolling(window=int(smoothing), center=True)\
                    .mean()\
                    .fillna(method='ffill')\
                    .fillna(method='bfill')\
                    .fillna(0)
                df.loc[idx, 'signal'] = signal.values
            else:
                df.loc[idx, 'signal'] = raw_signal
                
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            raise
    
    return df