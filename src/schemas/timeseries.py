from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from pydantic import field_validator, model_validator


# --- Timeframe -----------------------------------------------------------------


class TimeFrame(Enum):
    minute = "1Min"
    five_minutes = "5Min"
    fifteen_minutes = "15Min"
    hour = "1Hour"
    day = "1Day"
    week = "1Week"
    month = "1Month"

    @classmethod
    def from_minutes(cls, minutes: int) -> "TimeFrame":
        mapping = {
            1: cls.minute,
            5: cls.five_minutes,
            15: cls.fifteen_minutes,
            60: cls.hour,
            60 * 24: cls.day,
        }
        return mapping.get(minutes, cls.day)

    def to_alpaca(self) -> Any:
        try:
            from alpaca.data import TimeFrame as AlpacaTF  # type: ignore

            mapping = {
                TimeFrame.minute: AlpacaTF.Minute,
                TimeFrame.five_minutes: AlpacaTF(5, "Minute"),
                TimeFrame.fifteen_minutes: AlpacaTF(15, "Minute"),
                TimeFrame.hour: AlpacaTF.Hour,
                TimeFrame.day: AlpacaTF.Day,
                TimeFrame.week: AlpacaTF.Week,
                TimeFrame.month: AlpacaTF.Month,
            }
            return mapping[self]
        except Exception:
            return self.value


# --- Indicators ----------------------------------------------------------------


class Movement(Enum):
    rise = "rise"
    fall = "fall"
    freeze = "freeze"


BASIC_INDICATORS: Tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)


def _load_talib_function_names() -> List[str]:
    try:
        import talib  # type: ignore

        return sorted(set(talib.get_functions()))
    except Exception:
        # Minimal sensible default set if TA-Lib is unavailable at import time
        return [
            "SMA",
            "EMA",
            "WMA",
            "RSI",
            "MACD",
            "CCI",
            "ADX",
            "MFI",
            "OBV",
            "ATR",
            "STOCH",
            "STOCHRSI",
            "BBANDS",
        ]


TA_LIB_FUNCTIONS: Tuple[str, ...] = tuple(_load_talib_function_names())


class IndicatorSeries(BaseModel):
    """Generic indicator series: name + values aligned to dates.

    For TA-Lib multi-output indicators (e.g., MACD/BBANDS), you can represent
    each output as its own `IndicatorSeries` instance with a distinct name
    (e.g., "MACD_hist", "BBANDS_upper").
    """

    name: str = Field(..., description="Indicator name (OHLCV or TA-Lib function)")
    values: List[float] = Field(..., min_items=1)

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        normalized = v.strip()
        valid = set(BASIC_INDICATORS) | set(TA_LIB_FUNCTIONS) | {
            # Common derived outputs not direct TA-Lib function names
            "MACD_hist",
            "MACD_signal",
            "BBANDS_upper",
            "BBANDS_middle",
            "BBANDS_lower",
            "KDJ_K",
            "KDJ_D",
            "KDJ_J",
        }
        if normalized not in valid:
            raise ValueError(
                f"Unknown indicator '{v}'. Must be OHLCV, TA-Lib function, or a known derived output."
            )
        return normalized


class OHLCVWindow(BaseModel):
    recent_dates: List[date] = Field(..., min_items=1)
    open_list: Optional[List[float]] = None
    high_list: Optional[List[float]] = None
    low_list: Optional[List[float]] = None
    close_list: Optional[List[float]] = None
    adjusted_close_list: Optional[List[float]] = None
    volume_list: Optional[List[float]] = None

    @model_validator(mode="after")
    def validate_lengths(self) -> "OHLCVWindow":
        dates = self.recent_dates or []
        n = len(dates)
        for key in (
            "open_list",
            "high_list",
            "low_list",
            "close_list",
            "adjusted_close_list",
            "volume_list",
        ):
            series = getattr(self, key)
            if series is not None and len(series) != n:
                raise ValueError(f"{key} length {len(series)} must match dates length {n}")
        return self

    def ensure_adjusted_close_only(self) -> Dict[str, Any]:
        if self.adjusted_close_list is None:
            raise ValueError("adjusted_close_list is required for paper-compat JSON")
        return {
            "recent_date_list": [d.isoformat() for d in self.recent_dates],
            "adjusted_close_list": self.adjusted_close_list,
        }


class QueryBasic(BaseModel):
    query_stock: str
    query_date: date
    timeframe: TimeFrame = TimeFrame.day
    window: OHLCVWindow

    def to_paper_json(self) -> str:
        payload = {
            "query_stock": self.query_stock,
            "query_date": self.query_date.isoformat(),
            **self.window.ensure_adjusted_close_only(),
        }
        return json.dumps(payload, separators=(",", ":"))

    @classmethod
    def from_dataframe(
        cls,
        symbol: str,
        df: pd.DataFrame,
        as_of: date,
        lookback: int = 5,
        timeframe: TimeFrame = TimeFrame.day,
        use_adjusted_close: bool = True,
    ) -> "QueryBasic":
        if symbol not in df.index.get_level_values(0):
            raise ValueError(f"Symbol {symbol} not found in index")
        sdf = df.loc[symbol]
        sdf = sdf[sdf.index.date < as_of].tail(lookback)
        if len(sdf) < lookback:
            raise ValueError("Insufficient lookback data for query")
        window = OHLCVWindow(
            recent_dates=[d.date() if isinstance(d, datetime) else d for d in sdf.index],
            open_list=sdf["open"].tolist() if "open" in sdf.columns else None,
            high_list=sdf["high"].tolist() if "high" in sdf.columns else None,
            low_list=sdf["low"].tolist() if "low" in sdf.columns else None,
            close_list=sdf["close"].tolist() if "close" in sdf.columns else None,
            adjusted_close_list=(
                sdf.get("adjusted_close", sdf.get("adj_close", sdf.get("close"))).tolist()
                if use_adjusted_close
                else None
            ),
            volume_list=sdf["volume"].tolist() if "volume" in sdf.columns else None,
        )
        return cls(
            query_stock=symbol,
            query_date=as_of,
            timeframe=timeframe,
            window=window,
        )


class QueryTechnical(QueryBasic):
    indicators: Dict[str, List[float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_indicators(self) -> "QueryTechnical":
        dates = self.window.recent_dates if getattr(self, "window", None) else []
        for name, series in self.indicators.items():
            IndicatorSeries(name=name, values=series)
            if len(series) != len(dates):
                raise ValueError(
                    f"Indicator '{name}' length {len(series)} must match dates length {len(dates)}"
                )
        return self

    def to_rag_blocks(self) -> List[str]:
        blocks: List[str] = []
        for name, series in self.indicators.items():
            blocks.append(
                json.dumps(
                    {
                        "indicator": name,
                        "recent_date_list": [d.isoformat() for d in self.window.recent_dates],
                        f"{name}_list": series,
                    },
                    separators=(",", ":"),
                )
            )
        return blocks


class Candidate(BaseModel):
    candidate_stock: str
    candidate_date: date
    movement: Optional[Movement] = Field(
        default=None, description="Optional label; not required for inference"
    )
    timeframe: TimeFrame = TimeFrame.day
    recent_dates: List[date] = Field(..., min_items=1)
    indicator: IndicatorSeries

    @model_validator(mode="after")
    def align_indicator_length(self) -> "Candidate":
        dates = self.recent_dates or []
        if len(self.indicator.values) != len(dates):
            raise ValueError(
                f"Indicator length {len(self.indicator.values)} must match dates length {len(dates)}"
            )
        return self

    def to_paper_json(self) -> str:
        key = f"{self.indicator.name}_list" if self.indicator.name not in BASIC_INDICATORS else f"{self.indicator.name}_list"
        payload = {
            "candidate_stock": self.candidate_stock,
            "candidate_date": self.candidate_date.isoformat(),
            "candidate_movement": self.movement.value if self.movement else "",
            "recent_date_list": [d.isoformat() for d in self.recent_dates],
            key: self.indicator.values,
        }
        return json.dumps(payload, separators=(",", ":"))


# --- Helper builders ------------------------------------------------------------


def build_candidates_from_dataframe(
    symbol: str,
    df: pd.DataFrame,
    as_of: date,
    indicators: Iterable[Tuple[str, Iterable[float]]],
    lookback: int = 5,
    timeframe: TimeFrame = TimeFrame.day,
    movement: Optional[Movement] = None,
) -> List[Candidate]:
    if symbol not in df.index.get_level_values(0):
        raise ValueError(f"Symbol {symbol} not found in index")
    sdf = df.loc[symbol]
    sdf = sdf[sdf.index.date < as_of].tail(lookback)
    if len(sdf) < lookback:
        raise ValueError("Insufficient lookback data for candidate")

    recent_dates = [d.date() if isinstance(d, datetime) else d for d in sdf.index]
    cands: List[Candidate] = []
    for name, series in indicators:
        values = list(series)
        cands.append(
            Candidate(
                candidate_stock=symbol,
                candidate_date=as_of,
                movement=movement,
                timeframe=timeframe,
                recent_dates=recent_dates,
                indicator=IndicatorSeries(name=name, values=values),
            )
        )
    return cands


__all__ = [
    "TimeFrame",
    "Movement",
    "IndicatorSeries",
    "OHLCVWindow",
    "QueryBasic",
    "QueryTechnical",
    "Candidate",
    "build_candidates_from_dataframe",
]


