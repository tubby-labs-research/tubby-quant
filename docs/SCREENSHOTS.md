# Screenshot examples

The README images are captures of the actual Alpha Lab console and its own
charts, using **artificial data only**. They are not UI mockups, real-market
backtests, investment recommendations or evidence of trading readiness.
No private strategy, provider data, portfolio or saved research result is used.

## What is shown

- **Console:** expression editor, saved alpha library and simulation controls.
- **Positive result:** `rank(ts_delta(close, 5))` on the artificial fixture.
- **Negative result:** the same expression with its sign reversed.

Both expressions are generic demonstration examples. The pair was selected to
show both result states, not to measure investment merit. The graphs are the
console's cumulative PnL and drawdown, not Qlib Validation charts. PnL is after
the selected turnover cost; Alpha Lab does **not** charge stock-borrow expense.
Its fixed-book accounting is not Qlib's cash/position ledger.

## Reproduce the examples

After [installation](INSTALL.md), run from the source checkout in PowerShell:

```powershell
$demoRoot = Join-Path $env:LOCALAPPDATA 'TubbyLabs\readme-synthetic-demo'
.\env\qlib\Scripts\python.exe scripts\prepare_readme_demo.py --output $demoRoot
.\env\qlib\Scripts\python.exe -m qlib_yahoo.webapp --browser --data-root $demoRoot
```

Use a **new directory outside the source checkout**. The preparation script
refuses an existing destination and never downloads data or makes model calls.
For another preparation, choose another new directory; to reopen the existing
results, run only the second command. On macOS substitute `env/qlib/bin/python`
and shell-appropriate paths; these captures were verified on Windows only.

Open Alpha Lab, choose either `SYNTHETIC` entry in the library, then **Use saved
settings**. Hide the bottom console panel for more room. Scroll within the result
pane to see cumulative PnL, drawdown and the year-by-year table. The expression
editor and settings panel may be resized/hidden with the normal UI controls.

### Fixture and settings

The script reuses the existing deterministic artificial generator
[`create_demo_dataset`](../autoresearch/backend.py), seed **7403**. To fit the
console's TOP200 selector, it makes 25 replicas of each of the eight original
synthetic paths: **200 synthetic instruments, only eight distinct price paths**.
This is a display fixture, not a diversified universe. No real trading calendar,
historical membership or cross-sectional market structure is implied.

| Setting | Value |
| --- | --- |
| Input calendar | Weekdays, 2023-01-02 through 2024-12-31; not exchange holidays |
| Evaluation | 2024-01-02 through 2024-12-31, 261 sessions |
| Universe | TOP200 artificial replicas |
| Signal delay / decay | 1 / 4 sessions |
| Neutralization / cap | MARKET / 1% per instrument |
| Warm-up | 252 sessions (the console default) |
| Turnover cost | 5 basis points |
| Book size | Fixed $20 million, not starting cash |

Both runs use identical settings and dates. Their PnL is not exactly opposite
because both pay costs. The UI's **Returns** and **Drawdown** percentages use
half the fixed book; see the [user guide](USER_GUIDE.md) and
[limitations](LIMITATIONS.md) before comparing them with another engine.
The charts automatically scale their axes, so compare the labeled values,
not the apparent steepness of the two lines.

## Capture provenance and upkeep

Captured on Windows, 2026-10-17, using the public console/evaluator source at
`864cee92dc66be9fb060dc067424c74f2fc542ee`, plus the demonstration preparation
script added with this gallery. No application behavior, metrics or chart data
was changed for the screenshots. The script saves `screenshot-demo.json` beside
the generated data, including the exact settings, expressions and summaries.
The generated snapshot also retains its input hashes.

The JPEG captures use the browser's default 1280-pixel-wide viewport. Only the
bottom status strip (which includes the local data-folder location) is excluded.
The editor is resized and panels are hidden/scrolled with ordinary UI controls;
no screenshot pixels or chart values are retouched.

Only reviewed screenshots and the recipe belong in this repository. Keep the
generated data, local paths, session tokens, libraries and receipts outside it.
Capture through the real UI; do not alter numbers, redraw lines or hide failed
checks to imply a better result. Review every replacement image for private
information. Recheck the recipe and this record when the evaluator changes.
