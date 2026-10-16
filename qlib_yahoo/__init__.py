"""Explicit-universe Yahoo-to-Qlib research workflow (not a market-wide dataset)."""

import os
from pathlib import Path


def default_data_root() -> str:
    """Windows keeps D:\\Qlib; macOS/Linux use ~/Qlib. TUBBY_QLIB_DATA_ROOT overrides both."""
    configured = os.environ.get("TUBBY_QLIB_DATA_ROOT", "").strip()
    if configured:
        return str(Path(configured).expanduser())
    return r"D:\Qlib" if os.name == "nt" else str(Path.home() / "Qlib")


DEFAULT_DATA_ROOT = default_data_root()
DEFAULT_ALPHA = "($close / Ref($close, 20) - 1) * ($volume / Mean($volume, 20))"
DEFAULT_SYMBOLS = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "PG", "HD",
    "XOM", "KO", "NVDA", "TSLA",
)
BENCHMARK = "SPY"
QLIB_RELEASE = "v0.9.7"
QLIB_COMMIT = "da920b7f954f48ab1bb64117c976710de198373e"
