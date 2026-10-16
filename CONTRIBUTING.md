# Contributing

Focused fixes and reproducible examples are welcome. Use
[GitHub issues](https://github.com/tubby-labs-research/tubby-quant/issues) for bugs
and proposed changes. Discuss substantial behavior changes before implementing them.

Keep changes small, preserve unrelated work and include the problem, resulting
behavior and relevant checks in your pull request. Use synthetic data when reporting
bugs. Do not attach strategies, portfolios, credentials, proprietary data or
generated campaign records. Security reports follow [SECURITY.md](SECURITY.md).

Create a branch from the current default branch, make a focused change, run the
relevant offline checks, and open a pull request. Include reproduction steps and
any remaining limitations. A maintainer reviews changes before merging; passing
checks do not replace review.

## License and sign-off

Project-owned contributions are accepted under Apache License 2.0, the same
license as the project: inbound equals outbound. Third-party files retain their
existing licenses and notices. Identify their source and license when adding or
modifying them. Contributors retain their copyright; no corporate copyright
assignment or separate contributor license agreement is required.

Use the [Developer Certificate of Origin 1.1](https://developercertificate.org/)
to certify your right to contribute. Add a sign-off to each contribution commit
with `git commit -s`, using your own name and email:

```text
Signed-off-by: Your Name <your-email@example.com>
```

## Local checks

From the source folder, run the offline tests with the installed Python:

```powershell
.\env\qlib\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

On macOS use `env/qlib/bin/python`. Read skips: optional packages, native runtime
and platform requirements can exclude checks. CI runs a bounded core suite on
Linux and Windows import/installer checks; it does not certify native signed fills,
desktop windows or clean-machine installs. Keep model calls and data downloads
out of offline tests. Do not weaken verification to accommodate local setup.

Review the diff and remove sensitive material before opening an issue or pull
request. Include relevant test results and skips.
