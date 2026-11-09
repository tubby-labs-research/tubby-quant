"""Simulation settings; importable without numpy so the command-line parser stays light."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import math

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
    evaluation_sessions: list[str] | None = None

    def validate(self) -> "Settings":
        if type(self.delay) is not int or self.delay not in (0, 1):
            raise DatasetError("delay must be 0 or 1")
        if type(self.decay) is not int or not 0 <= self.decay <= 512:
            raise DatasetError("decay must be a whole number from 0 to 512")
        if (not isinstance(self.neutralization, str)
                or self.neutralization.upper() not in NEUTRALIZATIONS):
            raise DatasetError(f"neutralization must be one of {', '.join(NEUTRALIZATIONS)}")
        if not (type(self.truncation) in (int, float) and math.isfinite(self.truncation)
                and 0 <= self.truncation <= 1):
            raise DatasetError("truncation must be between 0 (off) and 1")
        if not (type(self.booksize) in (int, float) and math.isfinite(self.booksize) and self.booksize > 0):
            raise DatasetError("booksize must be finite and positive")
        if not (type(self.cost_bps) in (int, float) and math.isfinite(self.cost_bps)
                and 0 <= self.cost_bps <= 1000):
            raise DatasetError("cost_bps must be between 0 and 1000")
        if type(self.warmup) is not int or not 0 <= self.warmup <= 2000:
            raise DatasetError("warmup must be a whole number of sessions from 0 to 2000")
        if type(self.pasteurization) is not bool:
            raise DatasetError("pasteurization must be boolean")
        if not isinstance(self.universe, str) or not self.universe.strip():
            raise DatasetError("universe must be a nonempty name")
        sessions = self.evaluation_sessions
        if sessions is not None:
            if type(sessions) is not list or len(sessions) < 2:
                raise DatasetError("evaluation_sessions must be a list of at least two ISO dates")
            sessions = list(sessions)
            for value in [self.start, self.end, *sessions]:
                try:
                    valid = type(value) is str and date.fromisoformat(value).isoformat() == value
                except ValueError:
                    valid = False
                if not valid:
                    raise DatasetError("evaluation_sessions requires ISO YYYY-MM-DD dates and explicit start/end")
            if sessions != sorted(set(sessions)):
                raise DatasetError("evaluation_sessions must be sorted and unique")
            if self.start > self.end or sessions[0] < self.start or sessions[-1] > self.end:
                raise DatasetError("evaluation_sessions must be within the explicit start/end window")
        return replace(self, neutralization=self.neutralization.upper(), universe=self.universe.upper(),
                       evaluation_sessions=sessions)
