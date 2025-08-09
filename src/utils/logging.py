from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logging(level: Optional[str] = None, use_rich: bool = False) -> None:
    """Configure root logger for CLI-friendly output.

    - Level can come from argument, env var `ALPACIUM_LOG_LEVEL`, or default INFO
    - If Rich is available and use_rich=True, use RichHandler for nicer rendering
    - Idempotent: subsequent calls update level and keep one handler
    """
    lvl = (level or os.getenv("ALPACIUM_LOG_LEVEL") or "INFO").upper()
    lvl_num = getattr(logging, lvl, logging.INFO)

    root = logging.getLogger()
    root.setLevel(lvl_num)

    # Remove duplicate handlers to keep output clean in notebooks
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    handler: logging.Handler
    if use_rich:
        try:
            from rich.logging import RichHandler  # type: ignore

            handler = RichHandler(rich_tracebacks=False, show_time=True, show_path=False)
            fmt = "%(message)s"
        except Exception:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s | %(levelname)s | %(name)s: %(message)s"
    else:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(name)s: %(message)s"

    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    logging.getLogger(__name__).info("Logging configured: level=%s rich=%s", lvl, bool(use_rich))


__all__ = ["configure_logging"]


