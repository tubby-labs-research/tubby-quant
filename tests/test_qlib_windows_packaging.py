"""Windows packaging: launchers, setup script, icon, shortcuts and the workbench's key labels.

These run on any platform; the PowerShell and WebView2 parts themselves need a Windows machine.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = [ROOT / "launch-qlib-console.cmd"]
SETUP = ROOT / "scripts" / "windows" / "setup.ps1"


def _crlf_only(data: bytes) -> bool:
    return data.count(b"\n") == data.count(b"\r\n") > 0


class LauncherTests(unittest.TestCase):
    def test_batch_files_use_crlf(self):
        # cmd.exe misreads labels and goto targets in LF-only batch files
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher.name):
                self.assertTrue(_crlf_only(launcher.read_bytes()))

    def test_git_keeps_crlf(self):
        attributes = (ROOT / ".gitattributes").read_text()
        self.assertIn("*.cmd text eol=crlf", attributes)
        self.assertIn("*.ps1 text eol=crlf", attributes)
        checked = subprocess.run(["git", "-C", str(ROOT), "check-attr", "eol", "--", "launch-qlib-console.cmd", "scripts/windows/setup.ps1"],
                                 capture_output=True, text=True)
        if checked.returncode:
            self.skipTest("not a git checkout")
        self.assertEqual(checked.stdout.count("eol: crlf"), 2, checked.stdout)

    def test_launchers_open_the_right_programs(self):
        main = (ROOT / "launch-qlib-console.cmd").read_text()
        self.assertIn('"%PYW%" -m qlib_yahoo.webapp %*', main)
        self.assertIn("import webview", main)
        self.assertRegex(main, r"(?m)^:browser\r?$")
        self.assertIn("-m qlib_yahoo.webapp --browser %*", main)
        self.assertFalse((ROOT / "launch-qlib-console-classic.cmd").exists())
        self.assertFalse((ROOT / "launch-workbench.cmd").exists())
        self.assertFalse((ROOT / "qlib_yahoo" / "console.py").exists())

    def test_requirements_match_macos(self):
        windows = (ROOT / "requirements-qlib-windows.txt").read_text()
        macos = (ROOT / "requirements-qlib-macos.txt").read_text()
        self.assertIn("-r requirements-qlib-yahoo.txt", windows)
        self.assertIn("-r requirements-qlib-table.txt", windows)
        pins = [re.search(r"(?m)^pywebview==(\S+)", text) for text in (windows, macos)]
        self.assertTrue(all(pins))
        self.assertEqual(pins[0].group(1), pins[1].group(1))


class SetupScriptTests(unittest.TestCase):
    def setUp(self):
        self.data = SETUP.read_bytes()
        self.text = self.data.decode("utf-8-sig")

    def test_encoding(self):
        # Windows PowerShell 5.1 reads BOM-less scripts in the ANSI code page
        self.assertTrue(self.data.startswith(b"\xef\xbb\xbf"))
        self.assertTrue(_crlf_only(self.data))

    def test_parameters_and_steps(self):
        for parameter in ("[switch]$Signed", "[switch]$NoShortcuts", "[string]$Python", "[string]$DataRoot", "[switch]$Recreate"):
            self.assertIn(parameter, self.text)
        self.assertIn("requirements-qlib-windows.txt", self.text)
        self.assertIn("sys.version_info[:2] == (3, 12)", self.text)
        self.assertIn("ensure_upstream", self.text)
        self.assertIn("install_fast_runtime.py", self.text)
        self.assertIn('"--data-root", $DataRoot', self.text)
        self.assertNotIn("install_signed_runtime.py", self.text)
        self.assertIn('"--create-shortcuts"', self.text)
        self.assertNotRegex(self.text, r"(?i)tkinter|\bTk\b|console-classic")

    def test_native_failures_stop_the_script(self):
        self.assertIn('$ErrorActionPreference = "Continue"', self.text)
        self.assertRegex(self.text, r"if \(\$LASTEXITCODE -ne 0\) \{ Fail")
        self.assertNotIn("Invoke-Expression", self.text)

    def test_package_install_progress_is_visible(self):
        install = next(line for line in self.text.splitlines()
                       if "Invoke-Checked" in line and "requirements-qlib-windows.txt" in line)
        self.assertNotIn('"--quiet"', install)

    def test_nothing_is_deleted(self):
        self.assertNotRegex(self.text, r"(?i)\bRemove-Item\b|\brmdir\b|\brd /s\b|\bdel\b")
        self.assertIn("Move-Item -Path $EnvDir", self.text)

    @unittest.skipUnless(shutil.which("pwsh") or shutil.which("powershell"), "PowerShell unavailable")
    def test_parses(self):
        shell = shutil.which("pwsh") or shutil.which("powershell")
        command = ("$errors = $null; [System.Management.Automation.Language.Parser]::ParseFile("
                   f"'{SETUP}', [ref]$null, [ref]$errors) | Out-Null; exit $errors.Count")
        self.assertEqual(subprocess.run([shell, "-NoProfile", "-Command", command], capture_output=True).returncode, 0)

    @unittest.skipUnless(os.name == "nt" and (shutil.which("pwsh") or shutil.which("powershell")),
                         "Windows PowerShell paths required")
    def test_selected_root_survives_shortcuts_and_fresh_shell_guidance(self):
        # Execute only the preparation/completion tail. Only the read-only
        # Python root query is real; installs, data writes and shortcuts are stubbed.
        fragment = self.text[self.text.index('Step "Preparing the data folder'):]
        special = "O'Brien 研究 space [fixture]"
        cases = [
            ("custom", rf"C:\{special}\custom", rf"C:\ignored", True, rf"C:\{special}\custom"),
            ("inherited", "", rf"C:\{special}\inherited", True, rf"C:\{special}\inherited"),
            ("default", "", "", True, r"D:\Qlib"),
            ("fallback", "", "", False, rf"C:\Users\{special}\Qlib"),
            ("inherited-unavailable", "", r"MissingSetupDrive:\Qlib", False, rf"C:\Users\{special}\Qlib"),
            ("relative", rf"data {special}", "", True, str(ROOT / f"data {special}")),
            ("inherited-relative", "", rf"data {special}", True, str(ROOT / f"data {special}")),
        ]
        fixtures = [dict(name=name, chosen=chosen, inherited=inherited, drive=drive,
                         expected=expected, noShortcuts=no_shortcuts)
                    for name, chosen, inherited, drive, expected in cases
                    for no_shortcuts in (False, True)]
        fixtures.append(dict(name="query-failure", chosen="", inherited="", drive=True,
                             noShortcuts=False, queryFailure=True))
        payload = base64.b64encode(json.dumps(dict(fragment=fragment, root=str(ROOT),
                                                  python=sys.executable, cases=fixtures))
                                   .encode("utf-8")).decode("ascii")
        script = r'''
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$fixture = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('PAYLOAD')) | ConvertFrom-Json
# A session-only drive exercises quoting of the checkout too, without creating files.
New-PSDrive -Name "Fixture O'Brien 研究 [checkout]" -PSProvider FileSystem -Root $fixture.root | Out-Null
$results = foreach ($case in $fixture.cases) {
  & {
    $Root = "Fixture O'Brien 研究 [checkout]:\"
    Microsoft.PowerShell.Management\Set-Location -LiteralPath $Root
    $EnvPython = "C:\Fixture O'Brien 研究 [checkout]\env\Scripts\python.exe"
    $DataRoot = $case.chosen
    $NoShortcuts = $case.noShortcuts
    $Signed = $false
    $env:USERPROFILE = "C:\Users\O'Brien 研究 space [fixture]"
    $env:TUBBY_QLIB_DATA_ROOT = $case.inherited
    $env:PYTHONIOENCODING = 'cp1252'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $script:messages = @()
    $script:calls = @()
    $script:created = $null
    $script:launched = $null
    function Step([string]$Message) {}
    function Fail([string]$Message) { throw $Message }
    function Write-Host([string]$Object) { $script:messages += $Object }
    function Test-Path([string]$Path) { return $case.drive }
    function New-Item {
      [CmdletBinding()]
      param([string]$ItemType, [switch]$Force, [string]$Path)
      $script:created = $Path
    }
    function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
      $script:calls += ,@($Arguments)
    }
    function Stub-Interpreter {
      if ($args[0] -eq '-c') {
        # Exercise the actual ASCII-safe Python -> PowerShell path protocol,
        # even when the child uses a code page that cannot encode the path.
        if ($case.queryFailure) { & $fixture.python -c 'raise SystemExit(23)' }
        else { & $fixture.python @args }
        return
      }
      $script:launched = @{ cwd = (Get-Location).ProviderPath; arguments = @($args) }
    }
    Set-Alias -Name $EnvPython -Value Stub-Interpreter
    try { . ([scriptblock]::Create($fixture.fragment)) }
    catch {
      if (-not $case.queryFailure) { throw }
      [pscustomobject]@{ failure = $_.Exception.Message; created = $script:created;
        calls = @($script:calls); messages = $script:messages }
      return
    }
    if ($case.queryFailure) { throw 'Expected the failed root query to stop setup' }
    $selected = $env:TUBBY_QLIB_DATA_ROOT
    $commands = @($script:messages | Where-Object { $_.StartsWith('Set-Location ') })
    if ($commands.Count -ne 1) { throw 'Expected one copyable launch command' }
    # Simulate a fresh terminal: no setup environment and a different real cwd.
    $env:TUBBY_QLIB_DATA_ROOT = $null
    Microsoft.PowerShell.Management\Set-Location -LiteralPath $env:SystemRoot
    . ([scriptblock]::Create($commands[0]))
    [pscustomobject]@{ name = $case.name; noShortcuts = $NoShortcuts;
      selected = $selected; created = $script:created; calls = @($script:calls);
      messages = $script:messages; launched = $script:launched;
      environmentAfterLaunch = $env:TUBBY_QLIB_DATA_ROOT }
  }
}
ConvertTo-Json -InputObject @($results) -Depth 8 -Compress
'''.replace("PAYLOAD", payload)
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        shells = dict.fromkeys(filter(None, (shutil.which("powershell"), shutil.which("pwsh"))))
        for shell in shells:
            with self.subTest(shell=shell):
                completed = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                           capture_output=True, encoding="utf-8", timeout=45)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                results = json.loads(completed.stdout.lstrip("\ufeff"))
                self.assertEqual(len(results), len(fixtures))
                for case, result in zip(fixtures, results):
                    with self.subTest(shell=shell, case=case["name"], noShortcuts=case["noShortcuts"]):
                        if case.get("queryFailure"):
                            self.assertEqual(result["failure"], "Could not determine the default data folder (exit 23).")
                            self.assertIsNone(result["created"])
                            self.assertEqual(result["calls"], [])
                            self.assertEqual(result["messages"], [])
                            continue
                        self.assertEqual(result["selected"], case["expected"])
                        self.assertEqual(result["created"], case["expected"])
                        self.assertEqual(Path(result["launched"]["cwd"]), ROOT)
                        self.assertEqual(result["launched"]["arguments"], [
                            "-m", "qlib_yahoo.webapp", "--data-root", case["expected"]])
                        self.assertFalse(result["environmentAfterLaunch"])
                        shortcuts = [call for call in result["calls"] if "--create-shortcuts" in call]
                        self.assertEqual(shortcuts, [] if case["noShortcuts"] else [[
                            "-m", "qlib_yahoo.webapp", "--create-shortcuts", "--data-root", case["expected"]]])
                        shortcut_guidance = [message for message in result["messages"]
                                             if "shortcut (Start menu or Desktop)" in message]
                        self.assertEqual(len(shortcut_guidance), int(not case["noShortcuts"]))
                        self.assertFalse(any("launch-qlib-console.cmd" in message for message in result["messages"]))


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is optional outside the Qlib environment")
class IconTests(unittest.TestCase):
    def test_ico_structure(self):
        from qlib_yahoo.icon import ico_bytes

        sizes = (16, 32, 256)
        data = ico_bytes(sizes)
        self.assertEqual(struct.unpack("<HHH", data[:6]), (0, 1, len(sizes)))
        end = 6 + 16 * len(sizes)
        for index, size in enumerate(sizes):
            width, height, colours, _, planes, bits, length, offset = struct.unpack("<BBBBHHII", data[6 + 16 * index:22 + 16 * index])
            with self.subTest(size=size):
                self.assertEqual((width, height), (size % 256, size % 256))  # 0 means 256
                self.assertEqual((colours, planes, bits), (0, 1, 32))
                self.assertEqual(offset, end)
                self.assertEqual(data[offset:offset + 8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(struct.unpack(">II", data[offset + 16:offset + 24]), (size, size))
                end = offset + length
        self.assertEqual(end, len(data))

    def test_icon_file_is_rewritten_only_when_it_changes(self):
        from qlib_yahoo.desktop import windows_icon_path
        from qlib_yahoo.icon import ico_bytes

        with tempfile.TemporaryDirectory() as tmp:
            path = windows_icon_path(Path(tmp) / "settings")
            self.assertEqual(path.read_bytes(), ico_bytes())
            os.utime(path, (1, 1))
            windows_icon_path(Path(tmp) / "settings")
            self.assertEqual(path.stat().st_mtime, 1)
            path.write_bytes(b"old drawing")
            windows_icon_path(Path(tmp) / "settings")
            self.assertEqual(path.read_bytes(), ico_bytes())
            self.assertEqual([p.name for p in path.parent.iterdir()], [path.name])


class ShortcutTests(unittest.TestCase):
    def test_script_quotes_every_value(self):
        from qlib_yahoo.desktop import SHORTCUT_NAME, shortcut_script

        script = shortcut_script(Path(r"C:\Users\O'Brien\Quant\env\qlib\Scripts\pythonw.exe"), '-m qlib_yahoo.webapp --data-root "D:\\Qlib"',
                                 Path(r"C:\Users\O'Brien\Quant"), Path(r"C:\icons\app.ico"))
        self.assertIn(r"$link.TargetPath = 'C:\Users\O''Brien\Quant\env\qlib\Scripts\pythonw.exe'", script)
        self.assertIn(r"""$link.Arguments = '-m qlib_yahoo.webapp --data-root "D:\Qlib"'""", script)
        self.assertIn(r"$link.IconLocation = 'C:\icons\app.ico,0'", script)
        self.assertIn(f"Join-Path $folder '{SHORTCUT_NAME}'", script)
        self.assertIn("GetFolderPath('Desktop')", script)
        self.assertNotIn("Desktop", shortcut_script(Path("t"), "", Path("w"), Path("i"), desktop=False))

    def _create(self, returncode=0, stdout="", stderr="", **kwargs):
        from qlib_yahoo import desktop

        calls = []

        def runner(command, **options):
            calls.append((command, options))
            return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

        with tempfile.TemporaryDirectory() as tmp, patch.object(desktop, "windows_icon_path", return_value=Path(tmp) / "app.ico"):
            result = desktop.create_windows_shortcuts(Path(tmp) / "Quant", runner=runner, **kwargs)
        return result, calls

    def test_creates_both_shortcuts(self):
        paths, calls = self._create(stdout="C:\\Start\\Tubby Qlib Console.lnk\r\nC:\\Desktop\\Tubby Qlib Console.lnk\r\n\r\n")
        self.assertEqual(paths, [Path("C:\\Start\\Tubby Qlib Console.lnk"), Path("C:\\Desktop\\Tubby Qlib Console.lnk")])
        (command, options), = calls
        self.assertEqual(command[:6], ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"])
        self.assertEqual(options, {"capture_output": True, "text": True})
        script = command[6]
        self.assertIn(str(Path("env") / "qlib" / "Scripts" / "pythonw.exe"), script)
        self.assertIn("$link.Arguments = '-m qlib_yahoo.webapp'", script)
        self.assertIn("GetFolderPath('Desktop')", script)

    def test_data_root_argument(self):
        _, calls = self._create(data_root=Path("E:\\Research\\Qlib"), desktop=False)
        self.assertIn("""$link.Arguments = '-m qlib_yahoo.webapp --data-root "E:\\Research\\Qlib"'""", calls[0][0][6])
        self.assertNotIn("Desktop", calls[0][0][6])
        # a drive root keeps its backslash without escaping the closing quote
        _, calls = self._create(data_root=Path("D:\\"))
        self.assertIn("""--data-root "D:\\\\"'""", calls[0][0][6])

    def test_failure_is_reported(self):
        with self.assertRaisesRegex(OSError, "access denied"):
            self._create(returncode=1, stderr="access denied\n")

    @unittest.skipIf(os.name == "nt", "these check the behaviour off Windows")
    def test_off_windows(self):
        from qlib_yahoo.desktop import create_windows_shortcuts, windows_dark_mode
        from qlib_yahoo.webapp.__main__ import main

        self.assertFalse(windows_dark_mode())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OSError):
                create_windows_shortcuts(Path(tmp))
            with self.assertRaises(OSError):
                main(["--create-shortcuts"])

    def test_command_line_passes_only_an_explicit_data_root(self):
        from qlib_yahoo.webapp.__main__ import main

        with tempfile.TemporaryDirectory() as tmp, patch("qlib_yahoo.desktop.create_windows_shortcuts", return_value=[]) as create:
            self.assertEqual(main(["--create-shortcuts"]), 0)
            self.assertEqual(main(["--create-shortcuts", "--data-root", tmp]), 0)
        (root, first), _ = create.call_args_list[0]
        self.assertEqual((root / "qlib_yahoo" / "webapp" / "__main__.py").is_file(), True)
        self.assertIsNone(first)
        self.assertEqual(Path(create.call_args_list[1][0][1]), Path(tmp))


@unittest.skipUnless(shutil.which("node"), "node unavailable")
class KeyLabelTests(unittest.TestCase):
    def test_mac_labels_read_as_windows_keys(self):
        core = (ROOT / "qlib_yahoo" / "webapp" / "static" / "core.js").read_text()
        signs = re.search(r"^const KEY_SIGNS = .*$", core, re.M).group(0)
        function = re.search(r"^function platformText\(.*?^}$", core, re.M | re.S).group(0)
        cases = ["\u2318R", "\u2318-click, Shift-click, \u2318A", "\u2318/", "\u232B", "\u2325\u2318I", "Run backtest", ""]
        script = f"{signs}\n{function}\nconsole.log(JSON.stringify({json.dumps(cases)}.map((t) => [platformText(t, false), platformText(t, true)])))"
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        converted = json.loads(result.stdout)
        self.assertEqual([windows for windows, _ in converted],
                         ["Ctrl+R", "Ctrl-click, Shift-click, Ctrl+A", "Ctrl/", "Del", "Alt+Ctrl+I", "Run backtest", ""])
        self.assertEqual([mac for _, mac in converted], cases)


if __name__ == "__main__":
    unittest.main()
