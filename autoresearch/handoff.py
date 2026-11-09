"""Read-only reporting and hand-off of campaign findings to Alpha Lab, Qlib or a reviewer.

Everything here reads the immutable record store. Nothing evaluates, calls a
model or writes research records; the only output is the file a caller asks for.
"""

from __future__ import annotations

import json
from pathlib import Path

from .records import ResearchStore

DECISION_ORDER = ("confirmed", "promote", "refine", "park", "not_confirmed", "insufficient", "reject", "repair")


def _by_time(records):
    return sorted(records, key=lambda r: (str(r.get("created_at", "")), r["id"]))


def _metric(value, key):
    number = (value or {}).get(key)
    return number if isinstance(number, (int, float)) and not isinstance(number, bool) else None


def campaign_report(store: ResearchStore, campaign_id: str | None = None) -> dict:
    """Counts, one row per hypothesis with its latest decision and metrics, and every final."""
    experiments = [e for e in _by_time(store.list("experiments"))
                   if campaign_id is None or e.get("campaign_id") == campaign_id]
    experiment_ids = {e["id"] for e in experiments}
    latest_experiment = {}
    for experiment in experiments:
        latest_experiment[experiment["hypothesis_id"]] = experiment
    reviews = [r for r in _by_time(store.list("reviews")) if r.get("experiment_id") in experiment_ids]
    latest_review = {}
    for review in reviews:
        latest_review[review["hypothesis_id"]] = review
    finals = [f for f in _by_time(store.list("finals")) if campaign_id is None or f.get("campaign_id") == campaign_id]
    final_by_hypothesis = {}
    for final in finals:
        final_by_hypothesis[final["hypothesis_id"]] = final
    rows = []
    for hypothesis in _by_time(store.list("hypotheses")):
        experiment = latest_experiment.get(hypothesis["id"])
        if experiment is None:
            continue
        review = latest_review.get(hypothesis["id"], {})
        validation = (experiment.get("result") or {}).get("metrics", {}).get("validation", {})
        final = final_by_hypothesis.get(hypothesis["id"])
        holdout = ((final or {}).get("result") or {}).get("metrics", {}).get("holdout", {})
        rows.append({
            "hypothesis_id": hypothesis["id"], "title": hypothesis.get("title", ""),
            "family": hypothesis.get("family", ""), "depth": hypothesis.get("depth", 0),
            "post_holdout": bool(hypothesis.get("post_holdout", False)),
            "expression": hypothesis.get("expression", ""), "experiment_id": experiment["id"],
            "outcome": (experiment.get("result") or {}).get("status"),
            "decision": review.get("decision"), "reason": review.get("reason", ""),
            "validation": {key: _metric(validation, key) for key in ("days", "sharpe", "returns", "turnover", "drawdown")},
            "final": None if final is None else {
                "final_id": final["id"], "decision": final.get("decision"), "reused": final.get("reused", False),
                "access_sequence": final.get("access_sequence"),
                "holdout": {key: _metric(holdout, key) for key in ("days", "sharpe", "returns", "turnover", "drawdown")}},
        })
    rows.sort(key=lambda row: (DECISION_ORDER.index(row["decision"]) if row["decision"] in DECISION_ORDER else len(DECISION_ORDER),
                               -(row["validation"]["sharpe"] if row["validation"]["sharpe"] is not None else -1e9)))
    return {"campaign_id": campaign_id,
            "counts": {name: len(store.list(name)) for name in ("sources", "hypotheses", "experiments", "reviews", "finals")},
            "decisions": {decision: sum(1 for row in rows if row["decision"] == decision)
                          for decision in sorted({row["decision"] for row in rows if row["decision"]})},
            "hypotheses": rows,
            "finals": [{"final_id": f["id"], "hypothesis_id": f["hypothesis_id"], "expression": f.get("expression", ""),
                        "decision": f.get("decision"), "reason": f.get("reason", ""), "reused": f.get("reused", False),
                        "access_sequence": f.get("access_sequence"),
                        "holdout": {key: _metric(((f.get("result") or {}).get("metrics", {})).get("holdout", {}), key)
                                    for key in ("days", "sharpe", "returns", "turnover", "drawdown")}}
                       for f in finals]}


def _fmt(value, spec=".2f"):
    return format(value, spec) if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"


def render_markdown(report: dict) -> str:
    """A compact human summary; the JSON report remains the complete record."""
    lines = ["# Research campaign report", ""]
    if report.get("campaign_id"):
        lines.append(f"Campaign `{report['campaign_id']}`")
        lines.append("")
    counts = report["counts"]
    lines.append("Records: " + ", ".join(f"{counts[name]} {name}" for name in ("sources", "hypotheses", "experiments", "reviews", "finals")))
    if report["decisions"]:
        lines.append("Decisions: " + ", ".join(f"{count} {name}" for name, count in report["decisions"].items()))
    lines += ["", "| Decision | Title | Validation Sharpe | Returns | Turnover | Drawdown | Final | Expression |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["hypotheses"]:
        v = row["validation"]
        final = row["final"]
        final_text = "-" if final is None else (f"{final['decision']}" + (" (reused)" if final["reused"] else "")
                                                + f", Sharpe {_fmt(final['holdout']['sharpe'])}")
        title = row["title"] + (" [post-holdout]" if row["post_holdout"] else "")
        lines.append(f"| {row['decision'] or row['outcome']} | {title} | {_fmt(v['sharpe'])} | {_fmt(v['returns'], '.1%')} | "
                     f"{_fmt(v['turnover'], '.1%')} | {_fmt(v['drawdown'], '.1%')} | {final_text} | `{row['expression']}` |")
    lines.append("")
    lines.append("Promotion and confirmation mean evidence for deeper research, not a trading recommendation.")
    return "\n".join(lines) + "\n"


def selected_candidates(report: dict, decision: str = "promote") -> list[dict]:
    """Rows to hand on: ``promote`` (latest review), ``confirmed`` (final holdout), or ``all`` successful rows."""
    if decision == "promote":
        return [row for row in report["hypotheses"] if row["decision"] == "promote"]
    if decision == "confirmed":
        return [row for row in report["hypotheses"] if row["final"] and row["final"]["decision"] == "confirmed"]
    if decision == "all":
        return [row for row in report["hypotheses"] if row["outcome"] == "ok"]
    raise ValueError("decision must be promote, confirmed or all")


def batch_text(report: dict, decision: str = "promote") -> tuple[str, int]:
    """Alpha Lab batch text (one FASTEXPR per line, provenance as comments) and the candidate count."""
    rows = selected_candidates(report, decision)
    lines = [f"# Research hand-off: {len(rows)} {decision} candidate(s)"
             + (f" from campaign {report['campaign_id']}" if report.get("campaign_id") else ""),
             "# Lab and Research share the FASTEXPR evaluator; validation metrics below are the campaign's own records.",
             ""]
    for row in rows:
        v = row["validation"]
        lines.append(f"# {row['title']} | hypothesis {row['hypothesis_id']} | {row['decision']}"
                     f" | validation Sharpe {_fmt(v['sharpe'])} returns {_fmt(v['returns'], '.1%')}"
                     f" turnover {_fmt(v['turnover'], '.1%')} drawdown {_fmt(v['drawdown'], '.1%')}"
                     + (f" | final {row['final']['decision']}" if row["final"] else ""))
        lines.append(row["expression"].replace("\n", " ").strip())
        lines.append("")
    return "\n".join(lines), len(rows)


def export_batch(report: dict, destination: Path, decision: str = "promote") -> Path:
    """Write an Alpha Lab batch file that ``python -m qlib_yahoo alpha batch`` and the console read.

    Comment lines carry the hypothesis identity and the recorded validation
    metrics so the hand-off stays traceable without the record store.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    text, _count = batch_text(report, decision)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def write_report(report: dict, destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return destination
