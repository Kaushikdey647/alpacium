import pandas as pd
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass


@dataclass
class AlphaSignals:
    """Container for alpha signals and metadata."""
    signals: pd.DataFrame  # DataFrame with signals and calculations
    metadata: Dict[str, Any]  # Additional info from alpha calculation
    parameters: Dict[str, Any]  # Parameters used to generate signals
    
    def __str__(self) -> str:
        # Basic signal info
        info = [
            f"Alpha Strategy: {self.metadata.get('function_name', 'Unknown')}",
            f"\nSignals Shape: {self.signals.shape}",
            f"Time Range: {self.signals.index.get_level_values(1).min()} to {self.signals.index.get_level_values(1).max()}",
            f"Symbols: {', '.join(self.metadata.get('symbols', []))}",
            "\nColumns:",
            f"{', '.join(self.signals.columns)}",
        ]
        
        # Signal statistics
        if 'signal' in self.signals.columns:
            signal_stats = self.signals['signal'].describe()
            info.extend([
                "\nSignal Statistics:",
                f"  Mean: {signal_stats['mean']:.4f}",
                f"  Std: {signal_stats['std']:.4f}",
                f"  Min: {signal_stats['min']:.4f}",
                f"  Max: {signal_stats['max']:.4f}",
                f"  Null Values: {self.signals['signal'].isna().sum()} ({(self.signals['signal'].isna().mean()*100):.1f}%)"
            ])
            
        # Check for any null values in other columns
        null_counts = self.signals.isna().sum()
        if null_counts.any():
            info.extend([
                "\nNull Values by Column:",
                *[f"  {col}: {count} ({(count/len(self.signals)*100):.1f}%)" 
                  for col, count in null_counts.items() if count > 0]
            ])
            
        # Parameters used
        info.extend([
            "\nParameters:",
            *[f"  {k}: {v}" for k, v in self.parameters.items()]
        ])
        
        # Memory usage
        memory_usage = self.signals.memory_usage(deep=True).sum()
        info.append(f"\nMemory Usage: {memory_usage / 1024 / 1024:.2f} MB")
        
        return "\n".join(info)


class AlphaEngine:
    """Engine for generating alpha signals from price data."""
    
    def __init__(self):
        """Initialize the alpha engine."""
        pass
    
    @staticmethod
    def _validate_data(
            historical_data: pd.DataFrame
    ) -> None:
        """
        Validate input data format.
        
        Args:
            historical_data: Multi-index DataFrame with (symbol, timestamp) 
                index and OHLCV data
        """
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if historical_data[list(required_columns)].isna().any().any():
            raise ValueError("Missing data in historical data")
    
    def generate_signals(
            self,
            historical_data: pd.DataFrame,
            alpha_function: Callable[..., pd.DataFrame],
            parameters: Optional[Dict[str, Any]] = None,
            show_progress: bool = True
    ) -> AlphaSignals:
        """
        Generate alpha signals using the provided function and parameters.
        
        Args:
            historical_data: Multi-index DataFrame with (symbol, timestamp) 
                index and OHLCV data
            alpha_function: Function that generates alpha signals
            parameters: Parameters to pass to the alpha function
            show_progress: Whether to show progress bar
            
        Returns:
            AlphaSignals containing signals and metadata
        """
        # Validate input data
        self._validate_data(historical_data)
        
        # Use empty dict if no parameters provided
        params = parameters or {}
        
        try:
            # Generate signals
            result_df = alpha_function(historical_data, **params)
            
            # Validate signal column exists
            if 'signal' not in result_df.columns:
                raise ValueError(
                    "Alpha function must return DataFrame with 'signal' column"
                )
            
            # Create metadata about the signal generation
            metadata = {
                'function_name': alpha_function.__name__,
                'symbols': list(
                    historical_data.index.get_level_values(0).unique()
                ),
                'data_length': len(historical_data),
                'signal_stats': {
                    'min': float(result_df['signal'].min()),
                    'max': float(result_df['signal'].max()),
                    'mean': float(result_df['signal'].mean()),
                    'std': float(result_df['signal'].std())
                },
                'columns_added': [
                    col for col in result_df.columns 
                    if col not in historical_data.columns
                ]
            }
            
            return AlphaSignals(
                signals=result_df,
                metadata=metadata,
                parameters=params
            )
            
        except Exception as e:
            msg = f"Error in {alpha_function.__name__}: {str(e)}"
            raise RuntimeError(msg) from e 