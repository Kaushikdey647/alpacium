from dataclasses import dataclass
from typing import List, Dict, Callable
from datetime import datetime

@dataclass
class Strategy:
    name: str
    alpha_function: Callable
    parameters: Dict
    description: str
    symbols: List[str]
    weight: float
    sector: str
    constraints: Dict = None

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
