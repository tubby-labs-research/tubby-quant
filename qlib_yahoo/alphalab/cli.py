"""``python -m qlib_yahoo alpha ...``: BRAIN-style simulation from the command line."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from qlib_yahoo.alphalab.settings import NEUTRALIZATIONS, Settings


def load_sec_config(path: Path) -> dict[str, Any]:
    """Read a dictionary or legacy SEC selection without loading facts."""
    from qlib_yahoo.sec_features import load_bindings

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("expected a JSON object")
        catalog = config.get("catalog")
        if not isinstance(catalog, str) or not catalog.strip():
            raise ValueError("catalog must be a nonempty path string")
        if config.get("mode") not in (None, "dictionary", "legacy"):
            raise ValueError("mode must be dictionary or legacy")
        if "bindings" in config and not isinstance(config["bindings"], dict):
            raise ValueError("bindings must be a JSON object")
        bindings = load_bindings(config) if "bindings" in config else None
        catalog_path = Path(catalog)
        if not catalog_path.is_absolute():
            catalog_path = path.parent / catalog_path
        return {**config, "catalog": str(catalog_path.resolve()),
                **({"bindings": bindings} if bindings is not None else {})}
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid SEC config {path}: {error}") from error


def add_parser(commands: Any) -> None:
    alpha = commands.add_parser("alpha", help="Alpha Lab: fast BRAIN-style simulation of FASTEXPR alphas, "
                                              "one at a time or many in parallel, with a local alpha library.")
    sub = alpha.add_subparsers(dest="alpha_command", required=True)

    def settings_arguments(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", type=Path, help="Snapshot (default: latest.json).")
        p.add_argument("--sec-config", type=Path, help="JSON SEC catalog (dictionary) or explicit legacy field bindings.")
        p.add_argument("--group-overlay", type=Path, help="Dated industry/sector/country classification folder bound to this snapshot.")
        p.add_argument("--universe", default="TOP3000", help="TOP3000 (default), TOP2000, TOP1000, TOP500, TOP200.")
        p.add_argument("--delay", type=int, default=1, choices=(0, 1))
        p.add_argument("--decay", type=int, default=0, help="Linear decay window in days (default 0).")
        p.add_argument("--neutralization", default="MARKET", type=str.upper, choices=NEUTRALIZATIONS)
        p.add_argument("--truncation", type=float, default=0.08, help="Largest weight per stock (default 0.08; 0 off).")
        p.add_argument("--no-pasteurization", action="store_true", help="Allow positions outside the universe.")
        p.add_argument("--booksize", type=float, default=20_000_000.0)
        p.add_argument("--cost-bps", type=float, default=0.0, help="Cost per dollar traded, bps (default 0 like BRAIN).")
        p.add_argument("--start", help="First test date (default: first full-universe session after warm-up).")
        p.add_argument("--end", help="Last test date (default: last session).")
        p.add_argument("--warmup", type=int, default=252, help="History sessions before --start for time-series operators.")
        p.add_argument("--oos-start", help="Also report in-sample / out-of-sample metrics split at this date.")

    run = sub.add_parser("run", help="Simulate one alpha and print its metrics, yearly table and checks.")
    run.add_argument("expression", help="FASTEXPR alpha, e.g. \"-rank(ts_delta(close, 5))\".")
    settings_arguments(run)
    run.add_argument("--json", type=Path, help="Write the full result (with daily PnL) as JSON.")
    run.add_argument("--save", action="store_true", help="Also add it to the alpha library.")

    batch = sub.add_parser("batch", help="Simulate every alpha in a file in parallel; save to the library.")
    batch.add_argument("file", type=Path, help="One alpha per line; lines starting with # are ignored.")
    settings_arguments(batch)
    batch.add_argument("--workers", type=int, default=4, help="Worker processes (default 4).")
    batch.add_argument("--force", action="store_true", help="Simulate again even when the library has the result.")
    batch.add_argument("--csv", type=Path, help="Write the leaderboard as CSV.")

    library = sub.add_parser("library", help="List saved alphas.")
    library.add_argument("--sort", default="fitness", choices=("fitness", "sharpe", "returns", "turnover", "created_at"))
    library.add_argument("--limit", type=int, default=30)

    sub.add_parser("operators", help="List operators, data fields and groups.")


def settings_from(args: argparse.Namespace) -> Settings:
    return Settings(universe=args.universe, delay=args.delay, decay=args.decay, neutralization=args.neutralization,
                    truncation=args.truncation, pasteurization=not args.no_pasteurization, booksize=args.booksize,
                    cost_bps=args.cost_bps, start=args.start, end=args.end, warmup=args.warmup,
                    oos_start=args.oos_start).validate()


def _fmt(value: Any, spec: str) -> str:
    return format(value, spec) if isinstance(value, (int, float)) and math.isfinite(value) else "-"


def print_result(result: dict[str, Any]) -> None:
    s = result["summary"]
    print(f"{result['expression']}")
    print(f"  {s['start']} -> {s['end']} ({s['days']} sessions) | {result['settings']['universe']} delay "
          f"{result['settings']['delay']} decay {result['settings']['decay']} {result['settings']['neutralization']} "
          f"truncation {result['settings']['truncation']}")
    print(f"  Sharpe {_fmt(s.get('sharpe'), '.2f')}  Fitness {_fmt(s.get('fitness'), '.2f')}  Returns "
          f"{_fmt(s.get('returns'), '.1%')}  Turnover {_fmt(s.get('turnover'), '.1%')}  Drawdown "
          f"{_fmt(s.get('drawdown'), '.1%')}  Margin {_fmt(s.get('margin_bps'), '.1f')} bps  "
          f"Long/short {s['long_count']:.0f}/{s['short_count']:.0f}")
    print("  year  sharpe  returns  turnover  fitness  drawdown")
    for row in result["yearly"]:
        print(f"  {row['year']}  {_fmt(row.get('sharpe'), '6.2f')}  {_fmt(row.get('returns'), '7.1%')}  "
              f"{_fmt(row.get('turnover'), '8.1%')}  {_fmt(row.get('fitness'), '7.2f')}  {_fmt(row.get('drawdown'), '8.1%')}")
    for name, stats in (result.get("periods") or {}).items():
        print(f"  {name}: Sharpe {_fmt(stats.get('sharpe'), '.2f')} over {stats.get('days')} sessions")
    print("  checks: " + ", ".join(f"{c['name']} {c['result']}" for c in result["checks"]))


def main(args: argparse.Namespace) -> None:
    from qlib_yahoo.dataset import resolve_dataset

    if args.alpha_command == "operators":
        from qlib_yahoo.alphalab.engine import GROUPS, operator_names

        print("operators: " + ", ".join(operator_names()))
        print("fields: open high low close volume returns vwap dollar_volume adv<d> (+ close_raw dividend cap when present; vwap is stored when the snapshot has it, else (high+low+close)/3)")
        print("groups: " + " ".join(GROUPS))
        return
    from qlib_yahoo.alphalab.batch import Library, batch_correlations, read_expressions, run_batch

    if args.alpha_command == "library":
        for row in Library(args.data_root / "alphalab" / "library.sqlite").rows(sort=args.sort, limit=args.limit):
            print(f"{row['id']}  sharpe {_fmt(row['sharpe'], '5.2f')} fitness {_fmt(row['fitness'], '5.2f')} "
                  f"turnover {_fmt(row['turnover'], '6.1%')} selfcorr {_fmt(row['self_correlation'], '5.2f')}  "
                  f"{row['status']}  {row['expression'][:80]}")
        return
    settings = settings_from(args)
    sec_path = getattr(args, "sec_config", None)
    sec_options = {"sec": load_sec_config(sec_path)} if sec_path is not None else {}
    if getattr(args, "group_overlay", None):
        sec_options["group_overlay"] = args.group_overlay
    dataset = resolve_dataset(args.data_root, args.dataset)
    if args.alpha_command == "run":
        if args.save or sec_options:
            result = run_batch(dataset, [args.expression], settings, data_root=args.data_root, workers=1,
                               **sec_options)[0]
            if "error" in result:
                raise SystemExit(f"error: {result['error']}")
        else:
            from qlib_yahoo.alphalab.simulate import Workspace, json_safe, simulate

            result = json_safe(simulate(args.expression, settings, Workspace(dataset)))
        print_result(result)
        if args.json:
            args.json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        return
    results = run_batch(dataset, read_expressions(args.file), settings, data_root=args.data_root,
                        workers=args.workers, force=args.force, progress=lambda m: print(m, flush=True),
                        **sec_options)
    ok = sorted((r for r in results if "error" not in r), key=lambda r: -(r["summary"].get("fitness") or -1e9))
    print("\nrank  fitness  sharpe  returns  turnover  selfcorr  id                alpha")
    for n, r in enumerate(ok, 1):
        s = r["summary"]
        print(f"{n:>4}  {_fmt(s.get('fitness'), '7.2f')}  {_fmt(s.get('sharpe'), '6.2f')}  {_fmt(s.get('returns'), '7.1%')}  "
              f"{_fmt(s.get('turnover'), '8.1%')}  {_fmt(r.get('self_correlation'), '8.2f')}  {r['id']}  {r['expression'][:70]}")
    for r in results:
        if "error" in r:
            print(f"ERROR  {r['expression'][:70]}: {r['error']}")
    ids, matrix = batch_correlations(results)
    pairs = sorted(((matrix[i][j], ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))
                    if matrix[i][j] is not None), reverse=True)
    if pairs:
        print("\nmost correlated pairs in this batch: " + "; ".join(f"{a}~{b} {c:.2f}" for c, a, b in pairs[:5]))
    if args.csv:
        import csv

        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "expression", "sharpe", "fitness", "returns", "turnover", "drawdown", "margin_bps",
                             "self_correlation", "error"])
            for r in results:
                s = r.get("summary") or {}
                writer.writerow([r.get("id"), r["expression"], s.get("sharpe"), s.get("fitness"), s.get("returns"),
                                 s.get("turnover"), s.get("drawdown"), s.get("margin_bps"), r.get("self_correlation"),
                                 r.get("error", "")])

