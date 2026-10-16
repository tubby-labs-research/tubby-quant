# Windows helper: source and rebuild boundary

Normal installation uses the bundled, hash-checked Windows native helper; users
do not need a compiler. This page is for maintainers reviewing or rebuilding it.

The helper accelerates date-range predicates, not financial arithmetic. Its
implementation is `qlib_yahoo/session_range_buffer.pyx`. The runtime descriptor
pins that source, the corresponding signed caller and original build recipe.
`scripts/build_public_runtime_asset.py` adapts that recipe with relative source
filenames, compiler path mapping and deterministic build controls. It does not
edit executable bytes after compilation. The public helper has a separate
installation folder and code-owned artifact digest from the historical build.
The [portable build receipt](qlib-public-native-build.v1.json) records tools,
source digests, flags and limitations; its digest is linked from ZIP provenance.

The build requires CPython 3.12 Windows x64, the exact Cython 3.1.8 Windows wheel,
and compatible MSVC/Windows SDK tools already installed. It downloads nothing
and changes neither installed runtimes nor runtime pins. Create an ignored
`release-candidates` folder first, then run with the installed Python:

```powershell
.\env\qlib\Scripts\python.exe scripts/build_public_runtime_asset.py --workspace release-candidates/public-native-local --cython-wheel <path-to-reviewed-wheel>
```

The workspace must be new. An incomplete compiler installation fails unless
the maintainer explicitly supplies `--allow-incomplete-toolchain`; this is not
an instruction to waive that check automatically. Build logs and local evidence
can contain machine paths and must remain private. Only the inspected native
artifact, sanitized provenance and portable receipt belong in a distribution.

The prepared build used acknowledged incomplete MSVC tools and retains private
CPython imports. It is not a healthy-toolchain or public-API-only certificate.
Native artifact reproducibility is distinct from receipt/archive identity:
generated C metadata can depend on the build folder even when executable bytes
match. Never replace a configured helper, update hashes, or claim platform or
performance equivalence solely because compilation succeeds. Review source,
binary-path scans, exact hashes and behavioral/simulation evidence first.

See [runtime setup](QLIB_FAST_RUNTIME.md), [limitations](LIMITATIONS.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md).
