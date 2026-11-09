# Console design system

The console (`qlib_yahoo/webapp/static`) draws every view from one set of
tokens in `app.css`. View sheets (`home.css`, `discover.css`, `research.css`,
`workflow.css`) only compose those tokens; none defines its own colours or
sizes. This note records the decisions so that new views stay consistent.

## Appearance

The stylesheet follows the operating system by default. The toolbar toggle
(or the ⌘K entries under *Appearance*) pins light or dark by setting
`html[data-theme]`; the choice is kept in `localStorage` under `tq-theme`.
Tokens are defined three times with identical values: the dark default,
`@media (prefers-color-scheme: light)` for the system preference, and
`:root[data-theme=light]` for the pinned mode. Charts read the tokens when
they draw, so a change re-renders the open view.

## Surfaces and text

| Token | Role |
| --- | --- |
| `--canvas` | window chrome: toolbar, activity bar, status bar |
| `--panel` | side panels, cards, table headers |
| `--center` | the work area |
| `--raised` | controls, inputs, inset code blocks |
| `--overlay` | menus, dialogs, the command palette, toasts |
| `--border` / `--border-strong` | hairlines; the strong one outlines controls |
| `--card-shadow` | one inset highlight on dark, one soft drop shadow on light |
| `--t1` … `--t4` | text from primary to faint; `--t3` keeps 4.5:1 on panels |
| `--accent` / `--accent-strong` / `--accent-soft` | the single accent, its pressed shade and its wash |
| `--pos` / `--neg` / `--warn` and their `-soft` washes | direction and state only, never decoration |

Chart colours (`--series`, `--cat-1..6`, `--seq-1..5`, `--div-*`) are
separate from the interface colours so a chart never competes with a button.

## Type

The body is 13px / 1.5 in the system sans (`--sans`); numbers everywhere use
tabular figures. The scale is `--fs-xs` 11, `--fs-sm` 12, `--fs-md` 13,
`--fs-lg` 15, `--fs-xl` 20, `--fs-2xl` 26. Page titles are 22px at weight 650
with slightly tight tracking; section eyebrows are `--fs-xs`, uppercase,
tracked .06em, in `--t3`. Monospace (`--mono`) is reserved for expressions,
code and the console log, not for numbers or labels.

## Controls and spacing

Buttons, inputs, selects and segmented controls share one height
(`--ctl`, 30px; `--ctl-sm`, 26px for compact bars) and one radius family
(`--r-sm` 6, `--r-md` 10, `--r-lg` 14). The primary button carries a subtle
gradient and inset highlight; secondary buttons are outlined on `--raised`.
Spacing follows a 4px grid; cards use 14–16px padding, view headers 22–24px.

## Patterns

- **Stat tiles** (`.kpis`, `.dv-stats`, `.ar-budget`): one card split by
  hairlines, 22px values, an eyebrow label above and a one-line note below.
- **Stage timeline** (`.dv-stages`, `.ar-workflow`): numbered circles joined
  by a rule; done stages are green, the current stage is filled with the
  accent and is the only one that shows its subtitle.
- **Tables**: uppercase `--fs-xs` headers, 9px row padding, numeric cells in
  tabular figures, a 3px accent or green bar on the selected or promoted row.
- **Empty states**: a framed icon, a 15px title and one sentence with the
  next action.
- **Notices**: the top-of-view banner is scoped to the view it concerns and
  can be dismissed for the session; notes inside a result are quiet cards.
- **Motion**: 140ms for hover and toggles, 220ms for panels and dialogs,
  all with `--ease`; `prefers-reduced-motion` disables them.

## Checks

`tests/test_qlib_recovery_console.py` and `tests/test_sec_console.py` pin a
few layout rules that recovery and the SEC dialog rely on; keep those
selectors when editing `app.css`. The UI version string in `index.html`
(three places), `app.js` and the two SEC frontend tests must stay identical.
