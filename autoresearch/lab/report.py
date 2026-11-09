"""Read-only campaign reports: funnel, leaderboard, generators, finals and ledger integrity.

Nothing here evaluates, trains or writes research records. Reports read the
immutable store, the checkpoint and the trial ledger of one campaign.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import CampaignListing, CampaignReport, FinalSummary, Funnel, LeaderboardRow, StoreReport

if TYPE_CHECKING:
    from autoresearch.records import ResearchStore

    from .campaign import LabCampaign

DECISION_ORDER = ("promote", "reject", "park", "repair", "redundant")


def _number(value: object) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _leaderboard(store: ResearchStore, cid: str, domain: str | None) -> tuple[list[LeaderboardRow], list[dict[str, Any]]]:
    hypotheses = {h["id"]: h for h in store.list("hypotheses") if h.get("campaign_id") == cid}
    experiments = {e["hypothesis_id"]: e for e in sorted(
        (e for e in store.list("experiments") if e.get("campaign_id") == cid), key=lambda e: e.get("created_at", ""))}
    reviews = {r["hypothesis_id"]: r for r in sorted(
        (r for r in store.list("reviews") if r.get("campaign_id") == cid), key=lambda r: r.get("created_at", ""))}
    finals = {f["hypothesis_id"]: f for f in sorted(
        (f for f in store.list("finals") if f.get("campaign_id") == cid), key=lambda f: f.get("created_at", ""))}
    rows: list[LeaderboardRow] = []
    for hid, hypothesis in hypotheses.items():
        experiment = experiments.get(hid) or {}
        review = reviews.get(hid) or {}
        result = experiment.get("result") or {}
        validation = (result.get("metrics") or {}).get("validation") or {}
        final = finals.get(hid)
        checks = review.get("checks") or {}
        signal_like = domain == "alpha" or "sharpe" in validation
        if signal_like:
            numbers: dict[str, Any] = {k: _number(validation.get(k)) for k in ("days", "sharpe", "returns", "turnover",
                                                                               "drawdown", "fitness", "ic_mean")}
        else:
            numbers = {k: _number(validation.get(k)) for k in ("score", "baseline", "relative_improvement",
                                                               "p_value", "rows")}
            numbers["metric"] = validation.get("metric")
        final_summary: FinalSummary | None = None
        if final is not None:
            holdout = ((final.get("result") or {}).get("metrics") or {}).get("holdout") or {}
            final_summary = {"decision": final["decision"], "reused": final.get("reused", False),
                             "holdout": {k: v for k, v in holdout.items() if _number(v) is not None}}
        row: LeaderboardRow = {"hypothesis_id": hid, "title": hypothesis.get("title", ""),
                               "family": hypothesis.get("family", ""), "generator": hypothesis.get("generator", ""),
                               "round": hypothesis.get("round"), "screen_fitness": _number(hypothesis.get("screen_fitness")),
                               "decision": review.get("decision") or ("error" if result.get("status") == "error" else None),
                               "reason": review.get("reason", ""), "outcome": result.get("status"),
                               "post_holdout": bool(hypothesis.get("post_holdout")),
                               "validation": numbers, "final": final_summary}
        if signal_like:
            row["deflated_sharpe"] = _number((checks.get("deflated_sharpe") or {}).get("dsr"))
        if "expression" in hypothesis:
            row["expression"] = hypothesis["expression"]
        if "recipe" in hypothesis:
            row["recipe"] = hypothesis["recipe"]
        rows.append(row)

    def order(row: LeaderboardRow) -> tuple[int, float]:
        decision = row["decision"]
        rank = DECISION_ORDER.index(decision) if decision is not None and decision in DECISION_ORDER else len(DECISION_ORDER)
        score = row["validation"].get("sharpe")
        if score is None:
            score = row["validation"].get("relative_improvement")
        return (rank, -(score if score is not None else -1e9))

    rows.sort(key=order)
    return rows, list(finals.values())


def summary(campaign: LabCampaign) -> CampaignReport:
    """Campaign report from the checkpoint, the record store and the trial ledger."""
    status = campaign.status()
    cid = campaign.campaign_id
    domain = campaign.config["domain"]
    rows, finals = _leaderboard(campaign.store, cid, domain)
    redundant = sum(1 for v in campaign.checkpoint["validated"].values() if v.get("decision") == "redundant")
    funnel: Funnel = {"screened": status["screen_trials"], "screen_errors": status["screen_errors"],
                      "redundant": redundant, "validated": status["validation_trials"], "promoted": status["promotions"],
                      "finalized": len(finals), "confirmed": sum(1 for f in finals if f.get("decision") == "confirmed")}
    return {"campaign_id": cid, "name": campaign.config["name"], "domain": domain, "status": status["status"],
            "round": status["round"], "budget": status["budget"], "limits": status["limits"], "funnel": funnel,
            "generators": status["generators"], "leaderboard": rows, "ledger": campaign.ledger.verify(),
            "holdout_accessed": status["holdout_accessed"], "lineage": campaign.lineage_summary()}


def store_report(research_root: str | Path, campaign_id: str | None = None) -> StoreReport:
    """Report from records and the trial ledger alone (no configuration or checkpoint needed).

    Without ``campaign_id`` it lists every campaign in the store; with it, the
    funnel is recounted from the ledger, which is the authoritative trial count.
    """
    from autoresearch.records import ResearchStore

    from .ledger import TrialLedger

    root = Path(research_root)
    if not root.is_dir():
        return {"campaigns": [], "campaign": None}
    store = ResearchStore(root)
    campaigns = sorted(store.list("campaigns"), key=lambda c: c.get("created_at", ""))
    listing: list[CampaignListing] = [{"campaign_id": c["id"], "name": c.get("name", ""), "domain": c.get("domain"),
                                       "created_at": c.get("created_at")} for c in campaigns]
    if campaign_id is None:
        return {"campaigns": listing, "campaign": None}
    record = store.get("campaigns", campaign_id)
    if record is None:
        raise ValueError("no campaign with that identifier in this store")
    ledger = TrialLedger(root / "ledgers" / (campaign_id + ".jsonl"))
    verified = ledger.verify()
    stages = verified.get("stages", {}) if verified.get("ok") else {}
    entries = ledger.entries() if verified.get("ok") else []
    rows, finals = _leaderboard(store, campaign_id, record.get("domain"))
    generators: dict[str, int] = {}
    for entry in entries:
        if entry["stage"] == "screen":
            generators[entry.get("generator", "unknown")] = generators.get(entry.get("generator", "unknown"), 0) + 1
    funnel: Funnel = {"screened": stages.get("screen", 0),
                      "screen_errors": sum(1 for e in entries if e["stage"] == "screen" and e.get("status") != "ok"),
                      "redundant": sum(1 for e in entries if e["stage"] == "note" and e.get("decision") == "redundant"),
                      "validated": stages.get("validate", 0),
                      "promoted": sum(1 for e in entries if e["stage"] == "review" and e.get("decision") == "promote"),
                      "finalized": len(finals), "confirmed": sum(1 for f in finals if f.get("decision") == "confirmed")}
    report: CampaignReport = {"campaign_id": campaign_id, "name": record.get("name", ""), "domain": record.get("domain"),
                              "status": "recorded", "round": max((e.get("round") or 0 for e in entries), default=0),
                              "budget": None, "limits": {}, "funnel": funnel,
                              "generators": {g: {"screened": n} for g, n in generators.items()},
                              "leaderboard": rows, "ledger": verified,
                              "holdout_accessed": any(e["stage"] == "holdout" for e in entries)}
    return {"campaigns": listing, "campaign": report}


def _fmt(value: object, spec: str = ".2f") -> str:
    return format(value, spec) if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"


def render_markdown(report: CampaignReport, limit: int = 25) -> str:
    funnel = report["funnel"]
    lines = [f"# {report['name']} — {report['domain']} campaign", "",
             f"Campaign `{report['campaign_id']}` · round {report['round']} · status {report['status']}"
             + (f" ({report['budget']})" if report["budget"] else ""), "",
             "| Screened | Errors | Redundant | Validated | Promoted | Finalized | Confirmed |",
             "| --- | --- | --- | --- | --- | --- | --- |",
             f"| {funnel['screened']} | {funnel['screen_errors']} | {funnel['redundant']} | {funnel['validated']} | "
             f"{funnel['promoted']} | {funnel['finalized']} | {funnel['confirmed']} |", ""]
    ledger = report["ledger"]
    lines.append("Trial ledger: " + (f"{ledger['entries']} entries, chain intact" if ledger.get("ok")
                                     else "BROKEN: " + ledger.get("problem", "")))
    lineage = report.get("lineage")
    if lineage:
        trials = lineage["trials"]
        retired = lineage.get("holdout_retired")
        exposed = lineage.get("holdout_exposed")
        registry_status = lineage.get("registry_status")
        lines.append("")
        lines.append(f"Lineage registry: `{lineage['registry']}` · other campaigns' validation trials "
                     f"{max(trials['registry'], trials['store'])} (registry {trials['registry']}, record store "
                     f"{trials['store']}, " + ("agree" if trials["agree"] else "DISAGREE: finalize is non-confirmatory only")
                     + f") · holdout confirmation {lineage['confirmation']} at level {lineage['level']:g}"
                     + ("" if lineage.get("canonical", True) else " · custom registry (not this machine's canonical one)")
                     + (f" · holdout retired {str(retired.get('at', ''))[:10]} by {retired.get('campaign_id')}"
                        + (" (batch voided after a read)" if retired.get("method") == "voided-after-access" else "") if retired else "")
                     + (f" · holdout exposed {str(exposed.get('at', ''))[:10]} by {exposed.get('campaign_id')} "
                        "(read outside a confirmatory batch)" if exposed else "")
                     + (f" · REGISTRY {registry_status}: finalize is non-confirmatory only"
                        if registry_status in ("registry_reset", "registry_shrunk") else "")
                     + (" · registry seeded from a record store" if lineage.get("seeded") else "")
                     + (f" · open batch of {len(lineage['open_batch']['keys'])} by {lineage['open_batch']['campaign_id']}"
                        if lineage.get("open_batch") else ""))
    lines.append("")
    if report["domain"] == "alpha" or any("sharpe" in r["validation"] for r in report["leaderboard"]):
        lines += ["| Decision | Generator | Validation Sharpe | Returns | Turnover | DSR | Final | Candidate |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in report["leaderboard"][:limit]:
            v = row["validation"]
            final = row["final"]
            final_text = "-" if final is None else final["decision"] + (" (reused)" if final["reused"] else "")
            label = row.get("expression") or row.get("title", "")
            lines.append(f"| {row['decision']} | {row['generator']} | {_fmt(v.get('sharpe'))} | {_fmt(v.get('returns'), '.1%')} | "
                         f"{_fmt(v.get('turnover'), '.1%')} | {_fmt(row.get('deflated_sharpe'), '.3f')} | {final_text} | "
                         f"`{label[:100]}` |")
    else:
        lines += ["| Decision | Generator | Recipe | Validation score | Baseline | Improvement | p-value | Final |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in report["leaderboard"][:limit]:
            v = row["validation"]
            final = row["final"]
            final_text = "-" if final is None else final["decision"]
            lines.append(f"| {row['decision']} | {row['generator']} | {row['title']} | {_fmt(v.get('score'), '.4f')} | "
                         f"{_fmt(v.get('baseline'), '.4f')} | {_fmt(v.get('relative_improvement'), '.1%')} | "
                         f"{_fmt(v.get('p_value'), '.4f')} | {final_text} |")
    lines += ["", "Promotion and confirmation are evidence for deeper research, not a trading recommendation.", ""]
    return "\n".join(lines)
