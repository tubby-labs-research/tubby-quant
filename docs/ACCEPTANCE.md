# Acceptance test, 2026-11-08

This is the public copy of the private acceptance record; scripts and screenshots live in the private repository's session records, and the criteria, results and evidence are reproduced in full.

Scope: the final integrated tree (`claude/integration-check`, installed locally as the release tree) launched as the loopback server and as the native app bundle; macOS arm64
(12 cores), CPython 3.12.12, numpy 1.26.4, pandas 2.2.3, Qlib 0.9.7. Artificial demonstration
data throughout; nothing here is market evidence and the tool places no trades.

Each criterion is pass/fail with the evidence that decided it. "Pass" means observed on this
machine on the stated tree, not inferred from code. Known limitations are listed at the end.

## A. Launch, security and environment

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| A1 | The server binds a loopback port only, prints one start URL with a one-time token, and a visit exchanges it for an HttpOnly, SameSite=Strict cookie | pass | `GET /?token=…` → 303 with `Set-Cookie: tqc=…; Path=/; HttpOnly; SameSite=Strict`; the server prints exactly one start URL on a 127.0.0.1 port (`acceptance script a_security.py`) |
| A2 | A request whose Host header is not this server is refused (DNS-rebinding guard) | pass | `Host: evil.example.com` → 421 Misdirected Request, `{"error": "Unexpected host"}` (raw HTTP; the machine's HTTP proxy returns 502 for the same request, so the check bypasses it) |
| A3 | A request without the cookie gets no data; a wrong token is refused | pass | `GET /` without the cookie → 403 "Open the app from its launcher"; `GET /api/session` without the cookie → 403; wrong token → 403; `/../etc/passwd` → 404 |
| A4 | A state-changing POST without the token header or from a foreign Origin is refused | pass | `POST /api/settings` with the cookie but no `X-Token` → 403; with the token but `Origin: https://evil.example.com` → 403; same origin with the token → 200 |
| A5 | The native app bundle launches from Finder, records the final tree's hash as its build version, logs to `~/Library/Logs/Tubby Qlib Console/` and exits with its window | pass | `scripts/macos/build_app.sh` rebuilt `~/Applications/Tubby Qlib Console.app` from the release tree (bundle version = commit `e63b429`, ad-hoc signature, recorded checkout `~/TubbyQuant`); the bundle's launcher starts `python -m qlib_yahoo.webapp` from that checkout and logs to `~/Library/Logs/Tubby Qlib Console/launcher.log`; launch verified at the end of the run (see the final report) |

## B. Every view renders, light and dark, without script errors

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| B1 | Start, Alpha Lab, Discover, Validation, Data, Research and Evidence render at 1600x1000 in light and dark with zero console or page errors | pass | `acceptance script b_views.py`: seven views × light/dark × 1600×1000 and 1280×800, zero console errors and zero page errors on both runs (before and after the fixes below); the only console entries in the whole run are the browser's own log lines for deliberately refused requests (one 400 from `lab-submit` when an idea is refused by name) |
| B2 | The same at 1280x800 without horizontal overflow | pass | `scrollWidth − clientWidth` = 0 for every view at both widths in both schemes |
| B3 | The appearance toggle cycles system / light / dark, persists across a reload, and redraws charts | pass | toggle cycles light → dark → system (`data-theme` set, set, removed), survives a reload (`persisted: true`), charts redraw on change |
| B4 | Every text token clears WCAG AA contrast on every surface (gated by `tests/test_console_contrast.py`) | pass | `tests/test_console_contrast.py` in the suite (gate) |

## C. Guided first session on an empty data folder

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| C1 | On an empty data folder Start shows the three steps and no other view errors | pass (after fix) | first run: one `400 Bad Request` from `POST /api/lab-plan` fired by the Discover form on an empty folder; fixed in console-design `b1b6c32` (no request without prices; the plan box says what to do). Second run on a new empty folder: zero 4xx responses and zero console errors across all views |
| C2 | "Create demonstration data" produces a dataset within a minute and the checklist advances | pass | `Create demonstration data` → `datasets/demo-planted-200` (200 symbols, 1,500 sessions, provider `console-demo-planted-v1`) within seconds; checklist 1/3 |
| C3 | "Simulate these three now" runs the starter alphas in Alpha Lab and the step is marked done | pass (after fix) | first run simulated the two expressions left in the editor by an earlier session instead of the three starters; fixed in `b1b6c32` (the button now runs exactly the three starters and says when it replaced other text). Second run: three starters simulated, checklist 2/3 |
| C4 | "Run a short demo campaign" starts a Discover campaign and the step is marked done | pass (after fix) | `Run a short demo campaign` → quick alpha campaign complete, 6 rounds, 2 promoted, checklist 3/3. First run: the new-campaign form still showed "No ready prices" after the data was created (catalog cached once); fixed in `b1b6c32` (catalog read on every refresh, form redrawn when prices arrive). Second run: the form offers `TOP200 · Unpacked` with no plan error and no reload |

## D. Alpha Lab

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| D1 | A batch of three expressions completes with metrics and a detail view per alpha | pass | batch of three expressions completes in 1.1 s on the demonstration data; leaderboard with fitness, Sharpe, returns, turnover, drawdown, checks; correlated pairs listed; batch file written under `alphalab/batches/`; clicking a row opens the detail (`acceptance script d_lab.py`, screenshots `d1-lab-batch.png`, `d1-lab-detail.png`) |
| D2 | A malformed expression is named in a readable error without a traceback | pass | `rank(ts_mean(returns, 10)` → inline "Line 1: expected ')' at position 26, found the end" and the toast "Alpha Lab syntax check failed"; no traceback anywhere on the page |

## E. Discover (autonomous research)

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| E1 | An alpha campaign created from a preset runs to completion; the stage timeline and budget tiles update while it runs | pass | `acceptance script e_discover.py`: quick preset from the form, `Create & run` → stage timeline (6 stages marked) and 6 budget tiles update while running; campaign reached round 3 with 55 screened, 6 validated, 1 promoted |
| E2 | The leaderboard gives a reason for every decision and the candidate drawer opens | pass | 6 leaderboard rows, each with decision and reason; the drawer opens with family, round, screen fitness, decision and the expression |
| E3 | Finalize asks for confirmation, scores the holdout once, and the lineage line reports the registry state | pass | first campaign on the root: dialog "Score the promoted candidates on the reserved holdout?" → "Holdout scored", 1 confirmed, lineage line "8 validation trials by other campaigns in this registry · holdout confirmation batch at level 0.050 … canonical registry"; a second campaign on the same lineage gets "Read the holdout as evidence only?" because the holdout is retired (see Known limitations for the reuse case) |
| E4 | Verify reports the ledger intact and the checkpoint consistent with it | pass | "Ledger intact: 81 entries, head 4a6d295dadd5…, screen and validation counts match the checkpoint" |
| E5 | Export writes the promoted candidates to a file and reports the path | pass | "Exported 1 to …/autoresearch/lab/alpha/exports/acceptance-alpha-…-promote-….txt" with a Copy path button |
| E6 | A machine-learning campaign (tabular) runs to completion and the feature-importance chart is readable | pass | quick ML campaign (panel task on the demonstration prices) complete: 22 screened, 5 validated, 5 promoted, 5/5 rows with evidence; the importance chart is drawn for promoted tree models (seen on the demo ML campaign, `discover-ml.png`) |
| E7 | A seed idea the dataset cannot evaluate is refused by name before anything is written | pass (after fix) | first run: `rank(ts_mean(nonsense_field, 5))` was queued; fixed in `b1b6c32` (queued ideas pass the seed check; test added). Second run: toast "Cannot queue the idea … field 'nonsense_field' is not in this dataset …", the idea box keeps the text, nothing written |

## F. Validation, Research and Evidence

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| F1 | A Qlib validation run on the demonstration dataset completes and Overview, Returns, Signal, Details and Holdings render | pass | `acceptance script f_validation.py`: the saved Validation dataset of this folder was unavailable and the view said so ("Saved dataset is unavailable or cannot supply trading prices. Your saved expression is unchanged. Choose ordinary OHLCV prices…"); Data → `demo-planted-200` → *Use for Qlib Validation*, then `-1 * ($close / Ref($close, 5) - 1)` with ⌘R → a signed run completed in 8 s (runs 4 → 5, Sharpe 0.68 on artificial data); Overview, Returns, Signal, Details render; Holdings lists 33 positions |
| F2 | A Research demonstration campaign runs and its attempts appear in Evidence | pass | Research *Try demo* → campaign "Liquidity reversal · first evidence screen" recorded (offline, deterministic); Evidence lists the attempts (88 → 91 after the Lab and validation runs of this acceptance) |

## G. Keyboard and recovery

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| G1 | Cmd+K opens the palette with searchable commands and datasets; Escape closes it; Cmd+1..5 switch views | pass | ⌘K opens the palette (7 results for "discover": commands, datasets, examples), Escape closes it, ⌘5 switches to Discover (`acceptance script b_views.py`); the full shortcut table is exercised by the console harness tests |
| G2 | Killing the campaign worker mid-run leaves a consistent campaign: the console reports it, the ledger verifies, and the campaign resumes | pass | Thorough campaign running; `kill -TERM` of the worker: it stopped after the round in progress ("Stopped after round 2"), no error, ledger intact (121 entries), checkpoint matches; `lab-start` resumed one round (round 3) and verified again. `kill -KILL`: the console reports "The lab worker exited with code -9", the campaign row stays consistent (round 17), ledger intact (704 entries, no torn tail), checkpoint matches; resumed to round 18 and verified (`acceptance script g_recovery.py kill`) |
| G3 | Restarting the server mid-session shows a recovery note that can be dismissed per view | pass | server stopped while the worker ran round 25: the worker exited with it (no orphan process); after the restart the worker is idle, the campaign row is consistent (round 29), the ledger verifies (1,056 entries), one round resumes (round 30) and verifies again (`acceptance script g_recovery.py after-restart`); view-level notices (saved selection unavailable) appear once per view and dismiss with × (`acceptance script b_views.py`, `theme_check.py`) |

## H. Stability

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| H1 | Soak: at least 30 minutes with a campaign running and the status endpoints polled every 2 s: no 5xx, p95 latency of status calls under 500 ms, resident memory growth under 20 %, no traceback in the server log | pass | `acceptance script h_soak.py`: 30 min, a Standard alpha campaign (15 rounds of 48) running on the same server, six status endpoints polled every 2 s → 5,262 requests, 0 errors or 5xx; p95 latency 3.8–36 ms per endpoint (max 148 ms on `lab-worker`, 54 ms on `overview`); server resident memory 220 MB at start, 69 MB at the end (no growth; the peak was the start); no traceback in the server log |
| H2 | Two campaigns requested at once: the second is refused with "A campaign is already running", nothing is corrupted | pass | while the soak's standard campaign ran, `POST /api/lab-start` for a second campaign → 400 `{"error": "A lab campaign is already running. Stop it or wait for it to finish."}`; the first campaign kept running (`acceptance script g_recovery.py doublestart`) |

## I. Quality gates and calibration

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| I1 | `scripts/check_static.py`, `scripts/check_types.py`, `scripts/check_repository.py` pass on the tree | pass | on `e63b429`: `check_static` passed, `check_types` passed (mypy, strict for `autoresearch.lab`), `check_repository` passed once the public acceptance overlay existed (its only complaint was that missing link target) |
| I2 | The offline suite passes apart from documented environmental cases | pass | `scripts/run_tests.py` on `e63b429`: 2,421 tests, 208 skipped, 2 failures: the known environmental `test_workflow_recovery.test_live_owner_and_exited_worker`, and `test_framework_release.test_allowlist_is_explicit_and_omits_private_tooling`, which caught duplicate entries in the export allowlist produced by merging the release-preparation branch (389 entries, 350 unique); fixed in the release kit by de-duplicating the manifest, the test passes again. The real-worker integration test (`TUBBY_LAB_INTEGRATION=1`) passed |
| I3 | Benchmarks recorded on the tree (`docs/LAB_BENCHMARKS.md`): planted signals recovered, null panels promote nothing | pass | `docs/LAB_BENCHMARKS.md`, run alone on `e63b429` with one BLAS thread: alpha 168 s, 3/3 planted signals recovered, 7/7 finals confirmed, 5 null panels 2,106 screens / 17 validations / 0 promotions; machine learning 410 s, friedman1 9/9, interaction 13/13, null regression 0, planted panel 5/5 (Sharpe 5.39 on an artificial signal), null panel 0 |

## J. Installer

| # | Criterion | Result | Evidence |
| --- | --- | --- | --- |
| J1 | `scripts/install.sh` installs into a fresh prefix from a local source (`--dir`, `--data-root`, `--no-app`, `--source`), builds the pinned environment, prepares the data folder and the upstream Qlib script, and the installed console starts | pass | clean prefix: 48 s with cached wheels (first-ever installs download about 1.2 GB), `qlib_yahoo 0.2.0` / `qlib 0.9.7` import, `tubby-lab domains` lists the domains, the console starts on a loopback port and its start URL answers 303 with the cookie; a second run updates in place in 6 s (`acceptance script installer_test.sh`). First attempt failed its own post-install check because it ran the module outside the checkout; fixed (`0f3f935`) |

## Known limitations

- The lineage registry is kept under the machine's default data root (`~/Qlib/autoresearch/lineage`), not under the data folder a server was started with (`--data-root`). Two folders holding the same artificial dataset therefore share one lineage, which is why a campaign on a scratch folder showed "lineage disagrees" (14 trials in its own store, 24 in the registry). This is the harness's current design; making the registry authority independent of the data root is an open owner decision recorded in `docs/reviews/GPT_REVIEW_2_2026-11.md`.
- Continuous integration does not run for the private repository (organisation billing); every gate above ran locally.
- The synthetic demonstration data is the only data exercised here; real-market downloads need network access and were not part of this run.
