# Third-party code and runtime provenance

The project's Apache 2.0 licence does not replace licences for third-party
components or files carrying an existing MIT notice. This inventory describes
the source distribution; it is not a full transitive-dependency/legal clearance.

## Qlib and derived source

Microsoft Qlib is MIT licensed, Copyright (c) Microsoft Corporation. The full
notice is retained in [licenses/QLIB-MIT.txt](licenses/QLIB-MIT.txt).

Upstream base: Qlib 0.9.7, commit
[`da920b7f954f48ab1bb64117c976710de198373e`](https://github.com/microsoft/qlib/tree/da920b7f954f48ab1bb64117c976710de198373e).
The separate [Tubby Labs derivative](https://github.com/tubby-labs-research/qlib)
retains MIT and identifies its additions in its source history. Platform pins
are in [the runtime descriptor](docs/qlib-derivative-pin.json): Windows tubby.4
`8ab6249aa6143cb0675259d5ff986d294e918ef4`, macOS tubby.3
`d10a5a8ca397ea088da8f5c996e8bd042eea27e0`.

Existing MIT headers are retained in `qlib_yahoo/fast_signed.py`,
`flat_order_indicator.py`, `quote_construction_v2.py`, `quote_construction_v3.py`
and `session_loops.pyx`. The latter is retained source for an explicit experimental
path, not the native helper selected by default. These files derive from the
Qlib signed strategy, reporting and index-data implementations identified in
their source comments. Some headers also credit Tubby Labs contributors.

The signed numeric adapter/kernel (`ordered_signed.py` and
`ordered_signed_kernel.py`) retain formula/behaviour provenance to the MIT-licensed
signed ledger in the derivative. Preserve this attribution and the Qlib MIT
notice when redistributing derived portions. No claim of exclusive authorship or
blanket relicensing is made.

## Bundled native helper

`runtime-artifacts/qlib-date-range-buffer-cp312-win_amd64-public-v1.zip` contains only
the pinned Windows extension and sanitized provenance. Its source is
`qlib_yahoo/session_range_buffer.pyx`; build inputs and verification are retained
in `scripts/build_qlib_session_loops.py`, `scripts/build_public_runtime_asset.py`
and the runtime descriptor. The public rebuild retains the implementation's
exact source bytes but uses relative source names and reproducible compiler
path mapping instead of distributing private checkout paths. No compiler,
market dataset or Python environment is bundled.

Cython 3.1.8 generated the native source. Its upstream copying statement expressly
exempts compiled output, including embedded snippets, from Cython licensing
restrictions. The [verbatim statement](licenses/CYTHON-COPYING.txt) is retained;
[upstream source](https://github.com/cython/cython/blob/3.1.8/COPYING.txt).
The compiler itself is Apache 2.0 and is not bundled.

The extension is limited to CPython 3.12 / Windows x64, retains private CPython
imports and was built with an acknowledged incomplete toolchain installation.
Hash verification and corresponding source do not establish a public-API-only,
healthy-toolchain or cross-platform certificate. See [limitations](docs/LIMITATIONS.md).
Maintainer build details are in [the helper guide](docs/NATIVE_HELPER.md).

## Dependencies installed separately

Setup obtains these packages from their distributions; this source distribution does
not redistribute their wheels. Keep their own licence files and bundled notices
if creating a future application/environment bundle. Observed direct pins:

| Package | Version | Package-level licence observed |
| --- | --- | --- |
| NumPy | 1.26.4 | BSD-3-Clause; wheel includes additional component notices |
| pandas | 2.2.3 | BSD-3-Clause |
| SciPy | 1.14.1 | BSD-3-Clause; wheel includes additional component notices |
| MLflow | 3.16.1 | Apache-2.0 |
| pyqlib | 0.9.7 plus separate pinned derivative | MIT; use the upstream notice, not missing installed metadata |
| yahooquery | 2.4.1 | MIT |
| Plotly | 5.24.1 | MIT |
| statsmodels | 0.14.4 | BSD-3-Clause |
| pywebview | 6.2.1 | BSD-3-Clause |
| PyArrow | 25.0.1 | Apache-2.0; includes a separate third-party NOTICE |
| Numba | 0.65.1 | BSD; includes LICENSES.third-party |
| llvmlite | 0.47.0 | BSD-2-Clause and LLVM-related Apache-2.0-with-LLVM-exception material |
| isal (python-isal project) | 1.8.0 | PSF-2.0; retain packaged licence/component notices |

Versions and declared licences were inspected in the preparation environment;
this is not a promise that every transitive dependency shares those terms.
Desktop platform prerequisites, including WebView2 on Windows, have their own
distribution terms and are not supplied by this distribution. Report generation
copies Plotly JavaScript from the installed package; generated reports are not
bundled here. Before redistributing a report, accompany it with the appropriate
version-matched Plotly licence and applicable bundled-component notices. The
JavaScript copyright header alone is not a complete licence bundle, and this
framework does not currently assemble those report-sharing notices automatically.
See [Plotly's licence](https://github.com/plotly/plotly.py/blob/v5.24.1/LICENSE.txt)
and the notices for the exact JavaScript build being shared.

## Data, examples and assets

The included test datasets are synthetic fixtures, not redistributed market or
SEC data. Users supply their own datasets and are responsible for applicable
access/redistribution terms. The field names and importer code do not grant
rights to source datasets, taxonomy documentation or third-party data products.

The standalone website, its external fonts/images, article/course material,
private research and experiment results are excluded. The workbench's local
HTML/CSS/JavaScript and procedural icon are included; any subsequently copied
assets require their own provenance review.
