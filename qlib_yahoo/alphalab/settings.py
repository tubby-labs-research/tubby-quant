"""Simulation settings; importable without numpy so the command-line parser stays light."""

from __future__ import annotations

from dataclasses import dataclass, replace

from qlib_yahoo.dataset import DatasetError

NEUTRALIZATIONS = ("NONE", "MARKET", "SECTOR", "INDUSTRY", "COUNTRY")


@dataclass(frozen=True)
class Settings:
    universe: str = "TOP3000"
    delay: int = 1
    decay: int = 0
    neutralization: str = "MARKET"
    truncation: float = 0.08
    pasteurization: bool = True
    booksize: float = 20_000_000.0
    cost_bps: float = 0.0
    start: str | None = None
    end: str | None = None
    warmup: int = 252
    oos_start: str | None = None

    def validate(self) -> "Settings":
        if self.delay not in (0, 1):
            raise DatasetError("delay must be 0 or 1")
        if type(self.decay) is not int or not 0 <= self.decay <= 512:
            raise DatasetError("decay must be a whole number from 0 to 512")
        if self.neutralization.upper() not in NEUTRALIZATIONS:
            raise DatasetError(f"neutralization must be one of {', '.join(NEUTRALIZATIONS)}")
        if not (isinstance(self.truncation, (int, float)) and 0 <= self.truncation <= 1):
            raise DatasetError("truncation must be between 0 (off) and 1")
        if not (isinstance(self.booksize, (int, float)) and self.booksize > 0):
            raise DatasetError("booksize must be positive")
        if not (isinstance(self.cost_bps, (int, float)) and 0 <= self.cost_bps <= 1000):
            raise DatasetError("cost_bps must be between 0 and 1000")
        if type(self.warmup) is not int or not 0 <= self.warmup <= 2000:
            raise DatasetError("warmup must be a whole number of sessions from 0 to 2000")
        return replace(self, neutralization=self.neutralization.upper(), universe=self.universe.upper())
