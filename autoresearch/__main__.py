"""Command-line research worker. Nothing starts until explicitly invoked."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

from .config import default_research_root, load_config
from .controller import Controller
from .records import ResearchStore


def report(root):
    store = ResearchStore(root)
    return {"sources": len(store.list("sources")), "hypotheses": len(store.list("hypotheses")),
            "experiments": len(store.list("experiments")), "campaigns": len(store.list("campaigns")),
            "findings": [{key: row.get(key) for key in
                          ("hypothesis_id", "experiment_id", "decision", "reason", "question", "expected_change")}
                         for row in store.list("reviews")]}


def run(config, *, continuous=False, max_steps=100, allow_model=False):
    if config["model"]["provider"] != "fixture" and not allow_model:
        raise ValueError("live model calls require --allow-model and explicit model/environment configuration")
    controller = Controller(config)
    stop = threading.Event()
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
    try:
        steps = 0
        outcome = {"status": "stopped"}
        while not stop.is_set():
            outcome = controller.tick()
            print(json.dumps(outcome, sort_keys=True), flush=True)
            steps += 1
            state = controller.status()
            if outcome["status"] == "complete":
                break
            if outcome["status"] == "budget" and "daily" not in outcome.get("reason", ""):
                break
            if not continuous:
                if (steps >= max_steps or not state["queued"]
                        or outcome["status"] in {"budget", "idle"}):
                    break
            elif outcome["status"] in {"idle", "error", "budget"}:
                stop.wait(config.get("idle_seconds", 30))
        print(json.dumps({"status": "stopped", "campaign_id": controller.campaign_id,
                          "evaluations": controller.status()["evaluations"],
                          "queued": controller.status()["queued"]}), flush=True)
        return 1 if outcome["status"] == "error" else 0
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Daily alpha research with portable findings")
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run", help="run a bounded campaign pass, or a continuous worker")
    execute.add_argument("config", type=Path)
    mode = execute.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="drain the available queue (default)")
    mode.add_argument("--continuous", action="store_true", help="stay running, with configured idle pauses")
    execute.add_argument("--max-steps", type=int, default=100)
    execute.add_argument("--allow-model", action="store_true", help="enable explicitly configured live model calls")
    demo = commands.add_parser("demo", help="offline fixtures with real Alpha Lab synthetic-data simulations")
    demo.add_argument("--root", type=Path, required=True)
    for name in ("report", "reindex"):
        command = commands.add_parser(name)
        command.add_argument("--research-root", type=Path, default=default_research_root())
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import create_demo
            return run(load_config(create_demo(args.root)))
        if args.command == "run":
            if args.max_steps < 1:
                raise ValueError("max-steps must be positive")
            return run(load_config(args.config), continuous=args.continuous,
                       max_steps=args.max_steps, allow_model=args.allow_model)
        if args.command == "report":
            print(json.dumps(report(args.research_root), indent=2))
        elif args.command == "reindex":
            path = ResearchStore(args.research_root).rebuild_index()
            print(json.dumps({"status": "rebuilt", "index": str(path)}))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"autoresearch: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
