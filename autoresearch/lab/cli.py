"""``tubby-lab``: run, inspect, finalize and benchmark research-domain campaigns.

    tubby-lab domains
    tubby-lab demo alpha --root ~/tubby-lab-demo          # planted panel, discovery, holdout, report
    tubby-lab demo ml --root ~/tubby-lab-ml               # built-in Friedman #1 task
    tubby-lab run campaign.json --rounds 10
    tubby-lab report campaign.json --markdown
    tubby-lab submit campaign.json --expression "-rank(ts_delta(close, 5))" --rationale "short-term reversal"
    tubby-lab finalize campaign.json
    tubby-lab export campaign.json --output promoted.txt
    tubby-lab verify campaign.json
    tubby-lab benchmark alpha --root ~/bench
    tubby-lab benchmark null --root ~/bench-null --replicates 200   # false-promotion rate with an interval
    tubby-lab --blas-threads 4 benchmark ml --panel --root ~/bench-ml   # raise the one-thread BLAS cap

Nothing starts until invoked; no model calls, trades or downloads happen here.
Every command caps its own BLAS pool to one thread (``autoresearch.lab.runtime``)
unless ``--blas-threads`` or ``TUBBY_LAB_BLAS_THREADS`` says otherwise; the cap
changes wall time only.
Finalize is the only command that reads the reserved holdout, and only once
per candidate unless ``--allow-reuse`` records a labelled repeat.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

from .campaign import LabCampaign, atomic_json, load_lab_config
from .domain import catalog
from .runtime import limit_blas_threads
from .types import CampaignConfig, JSONObject, LeaderboardRow

#: What :func:`signal.signal` accepts and returns.
_SignalHandler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None


def _alpha_demo_config(root: Path, *, symbols: int, days: int, rounds: int) -> JSONObject:
    from autoresearch.domains.alpha.benchmark import benchmark_config
    from autoresearch.domains.alpha.planted import create_planted_dataset

    dataset = create_planted_dataset(root / "planted-data", symbols=symbols, days=days)
    config = benchmark_config("lab-alpha-demo", dataset, root, symbols=symbols, rounds=rounds, seed=3, proposals=32)
    return config


def _ml_demo_config(root: Path, *, rounds: int, task: str) -> JSONObject:
    from autoresearch.domains.ml.benchmark import panel_config, tabular_config

    if task == "panel":
        from autoresearch.domains.alpha.planted import create_planted_dataset

        dataset = create_planted_dataset(root / "planted-data", symbols=150, days=1400)
        return panel_config("lab-ml-panel-demo", dataset, root, symbols=150, rounds=rounds, seed=5, proposals=8)
    return tabular_config("lab-ml-demo", "friedman1", root, rounds=rounds, seed=5, proposals=10)


def create_demo(domain: str, root: Path, *, rounds: int, task: str = "tabular") -> Path:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign.json"
    if path.exists():
        return path
    if domain == "alpha":
        config = _alpha_demo_config(root, symbols=200, days=1600, rounds=rounds)
    else:
        config = _ml_demo_config(root, rounds=rounds, task=task)
    atomic_json(path, config)
    return path


def run(config: CampaignConfig, *, rounds: int | None, time_limit: float | None, quiet: bool = False) -> int:
    campaign = LabCampaign(config)
    stop = threading.Event()
    handlers: dict[signal.Signals, _SignalHandler] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, lambda *_: stop.set())
    started = time.time()
    try:
        done = 0
        while not stop.is_set():
            outcome = campaign.tick()
            if not quiet:
                printable: dict[str, object] = {k: v for k, v in outcome.items() if k != "decisions"}
                decisions = outcome["decisions"] if outcome["status"] == "round" else []
                printable["decisions"] = [{"decision": d["decision"], "score": d["score"]} for d in decisions]
                print(json.dumps(printable, sort_keys=True), flush=True)
            if outcome["status"] != "round":
                break
            done += 1
            if rounds is not None and done >= rounds:
                break
            if time_limit is not None and time.time() - started >= time_limit:
                break
        status = campaign.status()
        print(json.dumps({"status": "stopped", "campaign_id": campaign.campaign_id, "round": status["round"],
                          "screen_trials": status["screen_trials"], "validation_trials": status["validation_trials"],
                          "promotions": status["promotions"], "budget": status["budget"]}, sort_keys=True), flush=True)
        return 0
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        campaign.domain.release()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tubby-lab", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--blas-threads", type=int, metavar="N",
                        help="BLAS threads for this process and its workers (default: $TUBBY_LAB_BLAS_THREADS, "
                             "else 1; changes wall time only)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("domains", help="list the research domains")
    demo = commands.add_parser("demo", help="create and run an offline demonstration campaign")
    demo.add_argument("domain", choices=("alpha", "ml"))
    demo.add_argument("--root", type=Path, required=True)
    demo.add_argument("--rounds", type=int, default=8)
    demo.add_argument("--task", choices=("tabular", "panel"), default="tabular", help="ml demo task")
    for name in ("run", "status", "report", "finalize", "export", "submit", "verify"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "run":
            command.add_argument("--rounds", type=int, help="stop after this many rounds (budgets still apply)")
            command.add_argument("--time-limit", type=float, help="stop after this many seconds, between rounds")
            command.add_argument("--quiet", action="store_true")
        if name == "report":
            command.add_argument("--markdown", action="store_true")
            command.add_argument("--limit", type=int, default=25)
        if name == "finalize":
            command.add_argument("--key", action="append", default=[], help="explicit promoted candidate key")
            command.add_argument("--allow-reuse", action="store_true")
            command.add_argument("--non-confirmatory", action="store_true",
                                 help="read the holdout as exploratory evidence: never confirmed, never retiring it; "
                                      "required when the record store disagrees with the lineage registry, the "
                                      "registry does not match this campaign's last snapshot of it, the holdout was "
                                      "already retired, or it was read outside a confirmatory batch")
            command.add_argument("--void-batch", metavar="REASON",
                                 help="abandon this campaign's unfinished holdout batch with a recorded reason "
                                      "instead of completing it over the same family; a batch that had already read "
                                      "a key retires the lineage's holdout for confirmation")
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--decision", choices=("promote", "confirmed", "all"), default="promote")
        if name == "submit":
            command.add_argument("--expression", help="FASTEXPR candidate (alpha domain)")
            command.add_argument("--recipe", help='JSON recipe {"family": ..., "params": {...}} (ml domain)')
            command.add_argument("--rationale", default="")
            command.add_argument("--by", default="operator")
    bench = commands.add_parser("benchmark", help="measure a domain on synthetic tasks with known answers")
    bench.add_argument("domain", choices=("alpha", "ml", "null"),
                       help="alpha or ml benchmark; null: the alpha pipeline's false-promotion rate over N null panels")
    bench.add_argument("--root", type=Path, required=True)
    bench.add_argument("--rounds", type=int, help="rounds per campaign (null: default 8)")
    bench.add_argument("--panel", action="store_true", help="ml: include the planted cross-sectional panel")
    bench.add_argument("--replicates", type=int, help="null: number of independent null panels (required)")
    bench.add_argument("--names", type=int, default=80, help="null: names per panel (default 80)")
    bench.add_argument("--sessions", type=int, default=700, help="null: sessions per panel (default 700)")
    bench.add_argument("--proposals", type=int, default=24, help="null: proposals per round (default 24)")
    bench.add_argument("--study-seed", type=int, default=2026, help="null: seeds every replicate with its index")
    bench.add_argument("--workers", type=int, default=1, help="null: replicate processes (changes wall time only)")
    args = parser.parse_args(argv)
    try:
        limit_blas_threads(args.blas_threads)
        if args.command == "domains":
            print(json.dumps(catalog(), indent=2))
            return 0
        if args.command == "demo":
            path = create_demo(args.domain, args.root, rounds=args.rounds, task=args.task)
            config = load_lab_config(path)
            run(config, rounds=args.rounds, time_limit=None)
            campaign = LabCampaign(config)
            if campaign.state is None:
                campaign.status()
            checkpoint = campaign.checkpoint
            if checkpoint["accepted"] and not checkpoint["holdout_accessed"]:
                print(json.dumps({"status": "finalized", "finals": campaign.finalize()}, sort_keys=True), flush=True)
            from .report import render_markdown, summary
            print(render_markdown(summary(campaign)), end="")
            print(f"Campaign file: {path}")
            return 0
        if args.command == "benchmark":
            if args.domain == "null":
                from autoresearch.domains.alpha.null_calibration import REPORT_MARKDOWN, run_null_study
                if args.replicates is None:
                    raise ValueError("benchmark null needs --replicates")
                run_null_study(args.root, replicates=args.replicates, names=args.names, sessions=args.sessions,
                               rounds=args.rounds or 8, proposals=args.proposals, study_seed=args.study_seed,
                               workers=args.workers)
                print((Path(args.root).expanduser().resolve() / REPORT_MARKDOWN).read_text(), end="")
                return 0
            if args.domain == "alpha":
                from autoresearch.domains.alpha.benchmark import run_benchmark as run_alpha_benchmark
                run_alpha_benchmark(args.root, **({"rounds": args.rounds} if args.rounds else {}))
            else:
                from autoresearch.domains.ml.benchmark import run_benchmark as run_ml_benchmark
                run_ml_benchmark(args.root, panel=args.panel, **({"rounds": args.rounds} if args.rounds else {}))
            print((Path(args.root).resolve() / "benchmark.md").read_text(), end="")
            return 0
        config = load_lab_config(args.config)
        if args.command == "run":
            if args.rounds is not None and args.rounds < 1:
                raise ValueError("--rounds must be positive")
            return run(config, rounds=args.rounds, time_limit=args.time_limit, quiet=args.quiet)
        campaign = LabCampaign(config)
        if args.command == "status":
            print(json.dumps(campaign.status(), indent=2, sort_keys=True))
        elif args.command == "report":
            from .report import render_markdown, summary
            value = summary(campaign)
            print(render_markdown(value, args.limit) if args.markdown else json.dumps(value, indent=2, sort_keys=True), end="" if args.markdown else "\n")
        elif args.command == "finalize":
            # Requested keys that no longer pass the final selection are refused, recorded and reported here.
            eligibility = campaign.final_eligibility(args.key or None)
            outcomes = campaign.finalize(args.key or None, allow_reuse=args.allow_reuse,
                                         non_confirmatory=args.non_confirmatory, void_batch=args.void_batch)
            print(json.dumps({"campaign_id": campaign.campaign_id, "status": "finalized", "finals": outcomes,
                              "excluded": eligibility["excluded"],
                              "n_validation_trials": eligibility["n_validation_trials"],
                              "lineage": campaign.lineage_summary()},
                             indent=2, sort_keys=True))
        elif args.command == "export":
            if config["domain"] == "alpha":
                from autoresearch.handoff import campaign_report, export_batch
                path = export_batch(campaign_report(campaign.store, campaign.campaign_id), args.output, args.decision)
            else:
                from .report import summary

                def selected(row: LeaderboardRow) -> bool:
                    if args.decision == "all":
                        return row["outcome"] == "ok"
                    if args.decision == "confirmed":
                        final = row["final"]
                        return final is not None and final["decision"] == "confirmed"
                    return row["decision"] == "promote"

                rows = [r for r in summary(campaign)["leaderboard"] if selected(r)]
                if args.output.exists():
                    raise FileExistsError(f"refusing to overwrite {args.output}")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"campaign_id": campaign.campaign_id, "recipes": rows},
                                                  indent=2, sort_keys=True) + "\n")
                path = args.output
            print(json.dumps({"status": "written", "file": str(path), "decision": args.decision}))
        elif args.command == "submit":
            candidate: JSONObject
            if config["domain"] == "alpha":
                if not args.expression:
                    raise ValueError("alpha campaigns take --expression")
                candidate = {"expression": args.expression, "rationale": args.rationale}
            else:
                if not args.recipe:
                    raise ValueError("ml campaigns take --recipe")
                candidate = json.loads(args.recipe)
            print(json.dumps(campaign.submit(candidate, args.by), sort_keys=True))
        elif args.command == "verify":
            verified = campaign.verify()   # ledger chain, checkpoint counts, registry against its last snapshot
            print(json.dumps(verified, indent=2, sort_keys=True))
            return 0 if verified["ok"] else 1
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"tubby-lab: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
