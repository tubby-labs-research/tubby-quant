"""One setup command, then one command or prompt per OHLCV alpha."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from qlib_yahoo import DEFAULT_ALPHA, DEFAULT_DATA_ROOT, DEFAULT_SYMBOLS
from qlib_yahoo.backtest import run_backtest
from qlib_yahoo.dataset import DatasetError, parse_symbols, prepare, resolve_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Yahoo/Stooq OHLCV snapshots and Qlib backtests with a compact net ledger and optional HTML.")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT), help=f"Local storage root (default {DEFAULT_DATA_ROOT}; set TUBBY_QLIB_DATA_ROOT to change).")
    commands = parser.add_subparsers(dest="command", required=True)
    from qlib_yahoo.workflow_cli import add_parser as add_workflow_parser
    add_workflow_parser(commands)
    dictionary = commands.add_parser("sec-dictionary", help="Build, search or inspect a SEC dictionary.")
    dictionary.add_argument("dictionary_command", choices=("build", "search", "status", "prepare-execution"))
    dictionary.add_argument("--catalog", required=True, type=Path)
    dictionary.add_argument("--query", default="")
    dictionary.add_argument("--offset", type=int, default=0)
    dictionary.add_argument("--limit", type=int, default=50)
    dictionary.add_argument("--include-custom", action="store_true")
    prep = commands.add_parser("prepare", help="Download explicit US tickers, normalize bars and build Qlib bins.")
    prep.add_argument("--start", default="2022-01-01", help="First included session.")
    prep.add_argument("--end", default="2025-01-01", help="Exclusive end date.")
    ticker_group = prep.add_mutually_exclusive_group()
    ticker_group.add_argument("--symbols", help="Comma-separated trade symbols to include in the dataset.")
    ticker_group.add_argument("--symbols-file", type=Path, help="One trade symbol per line.")
    prep.add_argument("--attempts", type=int, default=3)
    prep.add_argument("--pause", type=float, default=1.0, help="Seconds between Yahoo requests; also used in retry backoff.")
    prep.add_argument("--resume-snapshot", type=Path, help="Resume an incomplete snapshot; pass its original symbols/dates again.")

    stooq = commands.add_parser("import-stooq", help="Read local Stooq daily files without modifying them; create a new Qlib snapshot.")
    stooq.add_argument("--source", type=Path, default=Path(r"D:\QuantData\Stooq_US_Hist"))
    stooq.add_argument("--start", required=True, help="First included date (YYYY-MM-DD).")
    stooq.add_argument("--end", required=True, help="Exclusive end date (YYYY-MM-DD).")
    selection = stooq.add_mutually_exclusive_group(required=True)
    selection.add_argument("--symbols", help="Explicit comma-separated trade tickers to include in the dataset.")
    selection.add_argument("--symbols-file", type=Path, help="One trade ticker per line; blank lines and # comments ignored.")
    stooq.add_argument("--make-latest", action="store_true", help="After successful validation, select this snapshot as the console default.")
    stooq.add_argument("--max-workers", type=int, default=2)

    table = commands.add_parser(
        "import-stooq-table",
        help="Read a consolidated Stooq table (Parquet/CSV) or daily US ASCII zip read-only; build a Qlib snapshot with a dated TOP-N universe.")
    table.add_argument("--table", type=Path, required=True,
                       help="Long table (.parquet/.csv) with ticker, exchange, type, date, open, high, low, close, volume, "
                            "or Stooq's daily US ASCII download (d_us_txt.zip).")
    table.add_argument("--start", required=True, help="First included date (YYYY-MM-DD).")
    table.add_argument("--end", required=True, help="Exclusive end date (YYYY-MM-DD).")
    table.add_argument("--universe-size", type=int, default=3000, help="Members per monthly ranking (default 3000).")
    table.add_argument("--universe-name", help="Universe file name (default top<size>).")
    table.add_argument("--adv-sessions", type=int, default=20, help="Trailing dollar-volume window (default 20).")
    table.add_argument("--min-history", type=int, default=252,
                       help="Valid bars required before a stock can be ranked (default 252).")
    table.add_argument("--min-session-coverage", type=float, default=0.5,
                       help="Drop sessions with fewer bars than this share of the trailing median (default 0.5).")
    table.add_argument("--corporate-actions", type=Path,
                       help="Yahoo chart archive from fetch-yahoo-charts: repair unadjusted splits, drop non-equity "
                            "instruments, add close_raw and dividend fields.")
    table.add_argument("--groups", type=Path,
                       help="Nasdaq screener folder (fetch-nasdaq-screener) or Sharadar folder: record "
                            "sector/industry/country per symbol.")
    table.add_argument("--min-price", type=float, default=0.0,
                       help="Rank only stocks whose unadjusted close on the ranking session is at least this (default 0: off).")
    table.add_argument("--keep-normalized", action="store_true",
                       help="Keep the intermediate normalized CSVs (removed after verification by default; ~65%% of the snapshot).")
    table.add_argument("--make-latest", action="store_true", help="After successful validation, select this snapshot as the console default.")
    table.add_argument("--max-workers", type=int, default=min(8, os.cpu_count() or 2))

    charts = commands.add_parser(
        "fetch-yahoo-charts",
        help="Fetch Yahoo daily charts with split/dividend events for the table's common stocks into a new raw archive.")
    charts.add_argument("--table", type=Path, required=True, help="The Stooq table or zip whose common stocks to fetch.")
    charts.add_argument("--start", required=True, help="First date (YYYY-MM-DD).")
    charts.add_argument("--end", required=True, help="Exclusive end date (YYYY-MM-DD).")
    charts.add_argument("--destination", type=Path,
                        help="Archive folder; an existing incomplete archive resumes (default <data root>/sources/yahoo-chart-<UTC time>).")
    charts.add_argument("--workers", type=int, default=6, help="Concurrent requests (default 6).")

    screener = commands.add_parser("fetch-nasdaq-screener",
                                   help="Save Nasdaq's stock screener (security names, sector, industry) as raw evidence.")
    screener.add_argument("--destination", type=Path,
                          help="New folder (default <data root>/sources/nasdaq-screener-<UTC time>).")
    build = commands.add_parser("build-yahoo-table",
                                help="Build the consolidated daily table (adjusted OHLCV, factor, raw close, dividends, "
                                     "security types) from a Yahoo chart archive and a Nasdaq screener.")
    build.add_argument("--charts", type=Path, required=True, help="Archive from fetch-yahoo-charts.")
    build.add_argument("--screener", type=Path, required=True, help="Folder from fetch-nasdaq-screener.")
    build.add_argument("--output", type=Path, required=True, help="New Parquet file; import it with import-stooq-table.")

    from qlib_yahoo.alphalab.cli import add_parser as add_alpha_parser

    add_alpha_parser(commands)

    sharadar = commands.add_parser(
        "fetch-sharadar",
        help="Download Sharadar bulk tables (prices incl. delisted stocks, tickers, actions, optional daily market cap) "
             "with your own API key from SHARADAR_API_KEY or --api-key-file.")
    sharadar.add_argument("--destination", type=Path,
                          help="New folder, or an interrupted download to resume (default <data root>/sources/sharadar-<UTC time>).")
    sharadar.add_argument("--api-key-file", type=Path, help="File whose first line is the key (default: SHARADAR_API_KEY).")
    sharadar.add_argument("--tables", default="tickers,actions,stocks",
                          help="Comma-separated: tickers,actions,stocks and optionally daily (market cap; Fundamentals/Bundle plans).")
    sharadar.add_argument("--years", choices=("5", "10", "full"), default="full", help="History depth of the plan (default full).")
    sharadar_build = commands.add_parser(
        "build-sharadar-table",
        help="Build the consolidated daily table (adjusted OHLCV, raw close, dividends, security types, delisted "
             "stocks, market cap when downloaded) from a Sharadar folder; import it with import-stooq-table.")
    sharadar_build.add_argument("--source", type=Path, required=True,
                                help="Folder from fetch-sharadar, or one holding tickers/actions/stocks[/daily] CSV or zip files.")
    sharadar_build.add_argument("--output", type=Path, required=True, help="New Parquet file.")
    sharadar_build.add_argument("--start", help="Drop bars before this date (YYYY-MM-DD; default keep all).")

    hd_build = commands.add_parser(
        "build-historicaldata-table",
        help="Build the consolidated daily table (adjusted OHLCV, VWAP, raw close, dividends, security types, delisted "
             "lifecycles) from an unpacked HistoricalData.net daily archive; import it with import-stooq-table.")
    hd_build.add_argument("--source", type=Path, required=True,
                          help="Archive folder holding symbols.csv and day_by_symbol/ (its manifest.json is checked when present).")
    hd_build.add_argument("--output", type=Path, required=True, help="New Parquet file.")
    hd_build.add_argument("--start", help="Drop bars before this date (YYYY-MM-DD; default keep all).")

    refresh = commands.add_parser(
        "refresh-market-data",
        help="All data steps in new dated folders: Nasdaq screener -> Yahoo charts -> table -> snapshot -> quality scan.")
    refresh.add_argument("--previous-table", type=Path, action="append", default=[],
                         help="Earlier consolidated table(s) whose tickers are fetched too (repeatable).")
    refresh.add_argument("--import-start", default="2010-01-01", help="First snapshot date (default 2010-01-01).")
    refresh.add_argument("--min-price", type=float, default=1.0, help="Universe price floor on the traded close (default 1).")
    refresh.add_argument("--universe-size", type=int, default=3000)
    refresh.add_argument("--workers", type=int, default=6, help="Concurrent chart requests (default 6).")
    refresh.add_argument("--make-latest", action="store_true", help="Make the new snapshot the default.")

    quality = commands.add_parser("data-quality", help="Scan a snapshot's prices for split artifacts, extreme and stale "
                                                         "prices, penny stocks and coverage; read-only.")
    quality.add_argument("--dataset", type=Path, help="Snapshot to scan (default: latest.json).")
    quality.add_argument("--universe", help="Dated universe to judge member-days by (default: the snapshot's first).")
    quality.add_argument("--output", type=Path, help="Also write the full JSON report here.")

    pack = commands.add_parser("pack-dataset", help="Create a new verified packed snapshot for faster small-file I/O; preserve the source.")
    pack.add_argument("--dataset", type=Path, required=True, help="Original prepared dataset to read.")
    pack.add_argument("--destination", type=Path, required=True, help="New snapshot directory; must not exist.")

    test = commands.add_parser("backtest", help="Run one expression and save a compact net ledger with optional HTML reports.")
    test.add_argument("--dataset", type=Path, help="Prepared snapshot; defaults to <data root>/latest.json.")
    test.add_argument("--sec-config", type=Path, help="JSON SEC catalog (dictionary) or explicit legacy field bindings.")
    test.add_argument("--alpha", help="Qlib daily expression; if absent, prompt interactively.")
    test.add_argument("--execution-profile", choices=("fast", "reference"), default="fast",
                      help="Fast daily adapters (default), or original Qlib adapters for parity checks; same trading rules.")
    test.add_argument("--history-mode", choices=("stream", "memory"),
                      help="Signed history: fast runs default to stream; memory retains original reference copies.")
    test.add_argument("--runtime-profile", choices=("auto", "native"), default="auto",
                      help="Accepted platform-specific signed accelerators (default), or native rollback; same trading rules.")
    test.add_argument("--alphas-file", type=Path,
                      help="One expression per line: independent trials in one prepared worker; cannot combine with --alpha.")
    test.add_argument("--start", help="First backtest session; default after --warmup-sessions and, for a dated universe, its first member signal.")
    test.add_argument("--end", help="Last backtest session; default penultimate dataset session.")
    test.add_argument("--topk", type=int, default=5)
    test.add_argument("--n-drop", type=int, default=1)
    test.add_argument("--cash", type=float, default=100_000)
    test.add_argument("--cost-per-side", type=float, default=0.0005)
    test.add_argument("--cash-rate", type=float, default=0.0, help="Annual cash opportunity rate for metrics; cash earns no interest in simulation.")
    test.add_argument("--strategy-mode", choices=("long-only", "signed"), default="long-only",
                      help="Portfolio construction; signed uses normalized raw alpha weights.")
    test.add_argument("--gross-exposure", type=float, default=1.0,
                      help="Signed mode gross exposure, from 0 to 1 (default 1).")
    test.add_argument("--borrow-rate", type=float, default=0.004,
                      help="Annual stock-borrow rate for signed mode (default 0.004).")
    test.add_argument("--signal-transform", choices=("raw", "rank", "zscore"), default="raw",
                      help="Signed mode: raw scores, cross-sectional ranks or z-scores become the weights "
                           "(rank and zscore are dollar-neutral; default raw).")
    test.add_argument("--universe", help="Dated universe recorded by the dataset, e.g. top3000 (default: all symbols).")
    test.add_argument("--warmup-sessions", type=int, default=252, help="Pre-evaluation sessions; must cover alpha lookback.")
    test.add_argument("--risk-degree", type=float, default=0.95)
    test.add_argument("--report-level", choices=("summary", "overview", "detailed"), default="summary")
    test.add_argument("--holdings", choices=("auto", "always", "never"), default="auto")
    test.add_argument("--threshold", type=float, default=1.3)
    test.add_argument("--min-observations", type=int, default=252)
    test.add_argument("--open-report", action="store_true", help="Open report in the system browser; selects overview from default summary.")
    replay = commands.add_parser("render-report", help="Render a report using only a completed saved run.")
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--level", choices=("summary", "overview", "detailed"), default="overview")
    replay.add_argument("--holdings", choices=("auto", "always", "never"), default="auto")
    replay.add_argument("--threshold", type=float, default=1.3)
    replay.add_argument("--min-observations", type=int, default=252)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "workflow":
            from qlib_yahoo.workflow_cli import main as workflow_main
            workflow_main(args)
        elif args.command == "prepare":
            if args.symbols_file:
                lines = args.symbols_file.read_text(encoding="utf-8").splitlines()
                symbols = parse_symbols(line for line in lines if line.strip() and not line.lstrip().startswith("#"))
            else:
                symbols = parse_symbols(args.symbols or DEFAULT_SYMBOLS)
            prepare(args.data_root, args.start, args.end, symbols, args.attempts, args.pause, args.resume_snapshot)
        elif args.command == "import-stooq":
            from qlib_yahoo.stooq import prepare_stooq

            if args.symbols_file:
                lines = args.symbols_file.read_text(encoding="utf-8").splitlines()
                symbols = tuple(line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#"))
            else:
                symbols = tuple(args.symbols.split(","))
            prepare_stooq(args.source, args.data_root, symbols, args.start, args.end,
                          make_latest=args.make_latest, max_workers=args.max_workers)
        elif args.command == "import-stooq-table":
            from qlib_yahoo.stooq_table import prepare_stooq_table
            from qlib_yahoo.universe import UniverseRule

            rule = UniverseRule(name=args.universe_name or f"top{args.universe_size}", size=args.universe_size,
                                adv_sessions=args.adv_sessions, min_history_sessions=args.min_history,
                                min_price=args.min_price)
            prepare_stooq_table(args.table, args.data_root, args.start, args.end, rule=rule,
                                min_session_coverage=args.min_session_coverage,
                                make_latest=args.make_latest, max_workers=args.max_workers,
                                keep_normalized=args.keep_normalized, corporate_actions=args.corporate_actions,
                                groups=args.groups)
        elif args.command == "fetch-yahoo-charts":
            from datetime import datetime, timezone

            from qlib_yahoo.stooq_table import common_stock_tickers
            from qlib_yahoo.yahoo_chart import fetch_archive

            tickers = common_stock_tickers(args.table, args.start, args.end)
            destination = args.destination or (args.data_root / "sources" /
                                               f"yahoo-chart-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
            print(f"Fetching {len(tickers):,} tickers into {destination}", flush=True)
            print(fetch_archive(tickers, destination, args.start, args.end, workers=args.workers,
                                progress=lambda message: print(message, flush=True)))
        elif args.command == "fetch-nasdaq-screener":
            from datetime import datetime, timezone

            from qlib_yahoo.reference import fetch_screener

            print(fetch_screener(args.destination or (args.data_root / "sources" /
                                                      f"nasdaq-screener-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")))
        elif args.command == "build-yahoo-table":
            from qlib_yahoo.yahoo_table import build_table

            manifest = build_table(args.charts, args.screener, args.output, progress=lambda m: print(m, flush=True))
            kinds = {reason: record["count"] for reason, record in manifest["classification"].items()}
            print(f"{args.output}: {manifest['rows']:,} rows; securities by type {kinds}; "
                  f"isolated spikes removed: {manifest['isolated_spikes']['rows']}")
        elif args.command == "sec-dictionary":
            import json
            from qlib_yahoo.sec_registry import build_dictionary, dictionary_status, search_dictionary
            from qlib_yahoo.sec_execution import build_execution_index, execution_status

            if args.dictionary_command == "build":
                if dictionary_status(args.catalog)["status"] != "ready":
                    build_dictionary(args.catalog, progress=lambda message: print(message, flush=True))
                build_execution_index(args.catalog, progress=lambda message: print(message, flush=True))
            elif args.dictionary_command == "prepare-execution":
                build_execution_index(args.catalog, progress=lambda message: print(message, flush=True))
            result = (search_dictionary(args.catalog, query=args.query, offset=args.offset,
                                        limit=args.limit, include_custom=args.include_custom)
                      if args.dictionary_command == "search" else dictionary_status(args.catalog))
            if args.dictionary_command != "search":
                result["execution"] = execution_status(args.catalog)
            print(json.dumps(result, indent=2, default=str))
        elif args.command == "alpha":
            from qlib_yahoo.alphalab.cli import main as alpha_main

            alpha_main(args)
        elif args.command == "fetch-sharadar":
            from datetime import datetime, timezone

            from qlib_yahoo.sharadar import api_key_from_environment, fetch_bulk

            destination = args.destination or (args.data_root / "sources" /
                                               f"sharadar-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
            print(fetch_bulk(destination, api_key_from_environment(args.api_key_file),
                             tables=[name.strip() for name in args.tables.split(",") if name.strip()],
                             years=args.years, progress=lambda message: print(message, flush=True)))
        elif args.command == "build-sharadar-table":
            from qlib_yahoo.sharadar import build_table as build_sharadar_table

            manifest = build_sharadar_table(args.source, args.output, start=args.start,
                                            progress=lambda message: print(message, flush=True))
            kinds = {kind: record["count"] for kind, record in manifest["classification"].items()}
            print(f"{args.output}: {manifest['rows']:,} rows; securities by type {kinds}; delisted stocks "
                  f"{manifest['delisted_stocks']['count']:,}; off-exchange bars dropped "
                  f"{manifest['listing_filter']['bars_dropped']:,}; isolated spikes removed {manifest['isolated_spikes']['rows']}")
        elif args.command == "build-historicaldata-table":
            from qlib_yahoo.historicaldata import build_table as build_historicaldata_table

            manifest = build_historicaldata_table(args.source, args.output, start=args.start,
                                                  progress=lambda message: print(message, flush=True))
            kinds = {kind: record["count"] for kind, record in manifest["classification"].items()}
            print(f"{args.output}: {manifest['rows']:,} rows from {manifest['files']:,} files; securities by type {kinds}; "
                  f"delisted stocks {manifest['delisted_stocks']['count']:,}; split-only adjusted "
                  f"{manifest['split_only_adjusted']['count']}; isolated spikes removed {manifest['isolated_spikes']['rows']}")
        elif args.command == "refresh-market-data":
            from qlib_yahoo.refresh import refresh

            record = refresh(args.data_root, previous_tables=args.previous_table, import_start=args.import_start,
                             min_price=args.min_price, universe_size=args.universe_size, make_latest=args.make_latest,
                             workers=args.workers, progress=lambda message: print(message, flush=True))
            print(record["snapshot"])
        elif args.command == "data-quality":
            import json

            from qlib_yahoo.data_quality import scan, summary_lines

            report = scan(resolve_dataset(args.data_root, args.dataset), args.universe)
            print("\n".join(summary_lines(report)))
            if args.output:
                args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        elif args.command == "pack-dataset":
            from qlib_yahoo.packed import pack_dataset
            print(pack_dataset(args.dataset, args.destination))
        elif args.command == "render-report":
            from qlib_yahoo.analysis import render_report

            print(render_report(args.run, level=args.level, holdings=args.holdings,
                                threshold=args.threshold, min_observations=args.min_observations))
        else:
            sec_options = {}
            if args.sec_config is not None:
                from qlib_yahoo.alphalab.cli import load_sec_config

                sec_options["sec"] = load_sec_config(args.sec_config)
            dataset = resolve_dataset(args.data_root, args.dataset)
            alpha = args.alpha
            if args.alphas_file and alpha is not None:
                raise DatasetError("use only one of --alpha and --alphas-file")
            if alpha is None and not args.alphas_file:
                if not sys.stdin.isatty():
                    raise DatasetError("use --alpha when running without an interactive terminal")
                alpha = input(f"Qlib alpha expression [{DEFAULT_ALPHA}]: ").strip() or DEFAULT_ALPHA
            settings = dict(
                start=args.start, end=args.end, topk=args.topk, n_drop=args.n_drop,
                initial_cash=args.cash, cost_per_side=args.cost_per_side, open_report=args.open_report,
                annual_cash_rate=args.cash_rate, warmup_sessions=args.warmup_sessions,
                risk_degree=args.risk_degree, report_level=args.report_level,
                holdings=args.holdings, threshold=args.threshold,
                min_observations=args.min_observations,
                strategy_mode=args.strategy_mode, gross_exposure=args.gross_exposure,
                annual_borrow_rate=args.borrow_rate, universe=args.universe,
                signal_transform=args.signal_transform,
                execution_profile=args.execution_profile,
                history_mode=args.history_mode,
                runtime_profile=args.runtime_profile,
                **sec_options,
            )
            if args.alphas_file:
                from qlib_yahoo.preparation import run_backtest_batch

                alphas = [line.strip() for line in args.alphas_file.read_text(encoding="utf-8").splitlines()
                          if line.strip() and not line.lstrip().startswith("#")]
                print(run_backtest_batch(dataset, alphas, data_root=args.data_root, **settings))
            else:
                run_backtest(dataset, alpha, **settings)
    except (DatasetError, OSError, ValueError, RuntimeError) as error:
        print(f"Qlib workflow failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
