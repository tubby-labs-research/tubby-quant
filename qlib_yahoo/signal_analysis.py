"""Signal diagnostics from a run's saved scores and forward labels: IC, rank IC and quantile returns.

The runner saves ``signal_forward_labels.parquet`` (datetime, instrument, score,
label) where the label is the next open-to-open return a signal could earn when
traded at the next open. This module reads it in date-ordered batches (a
TOP3000 run holds about 14 million rows) and returns, per session:

* IC: the Pearson correlation of score and label across the day's names;
* rank IC: the same on ranks (Spearman), the usual headline;
* quantile means: names sorted into equal-count score buckets (1 = lowest
  score), the mean label of each bucket, and the top-minus-bottom spread.

Quantile paths compound the bucket means: an equal-weight portfolio of each
bucket rebalanced every session, gross of costs. Never imports Qlib.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

MIN_NAMES = 20
BATCH_ROWS = 1 << 18


def _per_day(frame: Any, groups: int) -> Any:
    """Per-date IC, rank IC, name count and bucket means for complete dates in ``frame``."""
    import numpy as np
    import pandas as pd

    frame = frame.dropna(subset=["score", "label"])
    frame = frame[np.isfinite(frame["score"]) & np.isfinite(frame["label"])]
    if frame.empty:
        return pd.DataFrame()
    by_day = frame.groupby("datetime", sort=True)
    count = by_day["score"].transform("size")
    frame = frame[count >= MIN_NAMES].copy()
    if frame.empty:
        return pd.DataFrame()
    by_day = frame.groupby("datetime", sort=True)
    frame["rs"] = by_day["score"].rank(method="average")
    frame["rl"] = by_day["label"].rank(method="average")
    frame["n"] = by_day["score"].transform("size")
    frame["bucket"] = np.minimum(((frame["rs"] - 1) / frame["n"] * groups).astype(int), groups - 1)

    def pearson(x: str, y: str) -> Any:
        part = frame[["datetime", x, y]].copy()
        part["xy"], part["xx"], part["yy"] = part[x] * part[y], part[x] ** 2, part[y] ** 2
        sums = part.groupby("datetime", sort=True).agg(n=(x, "size"), sx=(x, "sum"), sy=(y, "sum"),
                                                       sxy=("xy", "sum"), sxx=("xx", "sum"), syy=("yy", "sum"))
        cov = sums["sxy"] - sums["sx"] * sums["sy"] / sums["n"]
        vx = sums["sxx"] - sums["sx"] ** 2 / sums["n"]
        vy = sums["syy"] - sums["sy"] ** 2 / sums["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            return cov / np.sqrt(vx * vy)

    result = pd.DataFrame({"ic": pearson("score", "label"), "rank_ic": pearson("rs", "rl")})
    result["names"] = by_day["score"].size()
    buckets = frame.groupby(["datetime", "bucket"], sort=True)["label"].mean().unstack("bucket")
    for index in range(groups):
        result[f"q{index + 1}"] = buckets.get(index)
    return result


def analyze(path: Path, groups: int = 5) -> dict[str, Any]:
    """Daily IC series, summary statistics and quantile returns for one run's forward labels."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    missing = {"datetime", "score", "label"} - set(source.schema_arrow.names)
    if missing:
        raise ValueError(f"forward labels lack columns {sorted(missing)}")
    parts, carry = [], None
    for batch in source.iter_batches(batch_size=BATCH_ROWS, columns=["datetime", "score", "label"]):
        frame = batch.to_pandas()
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        last = frame["datetime"].iloc[-1]
        carry = frame[frame["datetime"] == last]  # the last date may continue in the next batch
        done = frame[frame["datetime"] != last]
        if not done.empty:
            parts.append(_per_day(done, groups))
    if carry is not None and not carry.empty:
        parts.append(_per_day(carry, groups))
    daily = pd.concat([part for part in parts if not part.empty]) if parts else pd.DataFrame()
    if daily.empty:
        raise ValueError("no session has scores and labels for at least 20 names")
    daily = daily.sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]

    def stats(series: Any) -> dict[str, Any]:
        values = series.dropna().to_numpy(dtype=float)
        if len(values) < 2:
            return {"mean": None, "std": None, "ir": None, "t_stat": None, "positive_share": None, "days": len(values)}
        mean, std = float(values.mean()), float(values.std(ddof=1))
        return {"mean": mean, "std": std, "ir": mean / std if std else None,
                "annualized_ir": mean / std * math.sqrt(252) if std else None,
                "t_stat": mean / std * math.sqrt(len(values)) if std else None,
                "positive_share": float((values > 0).mean()), "days": int(len(values))}

    quantile_names = [f"q{index + 1}" for index in range(groups)]
    means = daily[quantile_names].mean()
    spread = daily[quantile_names[-1]] - daily[quantile_names[0]]
    paths = {}
    for name in quantile_names:
        paths[name] = np.expm1(np.log1p(daily[name].fillna(0).clip(lower=-0.99)).cumsum()).tolist()
    paths["spread"] = np.cumsum(spread.fillna(0)).tolist()  # additive: a dollar-neutral long-short
    order = np.arange(groups)
    monotonic = None
    if means.notna().all():
        ranks = means.rank().to_numpy()
        monotonic = float(np.corrcoef(order, ranks)[0, 1]) if np.std(ranks) else None
    return _clean({
        "dates": [str(day)[:10] for day in daily.index],
        "ic": daily["ic"].tolist(), "rank_ic": daily["rank_ic"].tolist(), "names": daily["names"].astype(int).tolist(),
        "summary": {"ic": stats(daily["ic"]), "rank_ic": stats(daily["rank_ic"]),
                    "median_names": int(daily["names"].median()), "groups": groups,
                    "spread_mean": float(spread.mean()), "spread_annualized": float(spread.mean() * 252),
                    "spread_sharpe": float(spread.mean() / spread.std(ddof=1) * math.sqrt(252)) if spread.std(ddof=1) else None,
                    "monotonicity": monotonic},
        "daily": {name: daily[name].tolist() for name in quantile_names},
        "quantiles": {"mean": [float(means[name]) if pd.notna(means[name]) else None for name in quantile_names],
                      "annualized": [float(means[name] * 252) if pd.notna(means[name]) else None for name in quantile_names],
                      "paths": paths},
        "label": "next open-to-open return (signal on day t, traded at the open of t+1)",
    })


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value
