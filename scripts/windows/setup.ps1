<#
.SYNOPSIS
  One-time Windows setup for the Qlib workbench.

.DESCRIPTION
  Creates env\qlib with Python 3.12, installs the pinned packages
  (Qlib 0.9.7, pandas, plotly, pyarrow, pywebview), prepares the data folder with the
  pinned Qlib source used to convert datasets, optionally builds the signed engine,
  and adds Start menu and Desktop shortcuts that open the workbench in its own window.
  Re-running is safe; nothing is deleted (-Recreate moves the old environment aside).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1 -Signed
#>
param(
  [switch]$Signed,          # also build and verify the pinned signed (long/short) Qlib engine
  [switch]$NoShortcuts,     # do not add Start menu / Desktop shortcuts
  [string]$Python = "",     # Python 3.12 to use (default: auto-detect)
  [string]$DataRoot = "",   # data folder (default: D:\Qlib, or %USERPROFILE%\Qlib without a D: drive)
  [switch]$Recreate         # move an existing env\qlib aside and create a fresh one
)
$ErrorActionPreference = "Continue"  # native tools write progress to stderr; exit codes decide instead
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

function Step([string]$Message) { Write-Host ""; Write-Host "==> $Message" -ForegroundColor Cyan }
function Fail([string]$Message) { Write-Host $Message -ForegroundColor Red; exit 1 }
function Test-Python([string]$Exe) {
  if (-not $Exe -or -not (Test-Path $Exe)) { return $false }
  & $Exe -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>$null | Out-Null
  return ($LASTEXITCODE -eq 0)
}
function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
  & $Exe @Arguments
  if ($LASTEXITCODE -ne 0) { Fail "Failed (exit $LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

Step "Finding Python 3.12"
$candidates = @()
if ($Python) { $candidates += $Python }
if (Get-Command py -ErrorAction SilentlyContinue) {
  $fromLauncher = & py -3.12 -c "import json, sys; print(json.dumps(sys.executable))" 2>$null
  if ($LASTEXITCODE -eq 0 -and $fromLauncher) { $candidates += ($fromLauncher | ConvertFrom-Json -ErrorAction Stop) }
}
$candidates += @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "$env:ProgramFiles\Python312\python.exe")
$onPath = Get-Command python -ErrorAction SilentlyContinue
if ($onPath) { $candidates += $onPath.Source }
$PythonExe = $null
foreach ($candidate in $candidates) { if (Test-Python $candidate) { $PythonExe = $candidate; break } }
if (-not $PythonExe) {
  Fail "Python 3.12 is required. Install it from python.org, then re-run."
}
Write-Host "Using $PythonExe"

$EnvDir = Join-Path $Root "env\qlib"
$EnvPython = Join-Path $EnvDir "Scripts\python.exe"
if ((Test-Path $EnvDir) -and $Recreate) {
  $aside = "$EnvDir.old-$(Get-Date -Format yyyyMMddHHmmss)"
  Move-Item -Path $EnvDir -Destination $aside -ErrorAction Stop
  Write-Host "Moved the previous environment to $aside"
}
if ((Test-Path $EnvPython) -and -not (Test-Python $EnvPython)) {
  Fail "env\qlib exists but is not Python 3.12; re-run with -Recreate."
}
if (-not (Test-Path $EnvPython)) {
  Step "Creating env\qlib"
  Invoke-Checked $PythonExe @("-m", "venv", $EnvDir)
}

Step "Installing pinned packages (Qlib 0.9.7, pandas, plotly, pyarrow, pywebview)"
Invoke-Checked $EnvPython @("-m", "pip", "install", "--quiet", "--upgrade", "pip")
Invoke-Checked $EnvPython @("-m", "pip", "install", "--no-cache-dir", "-r", (Join-Path $Root "requirements-qlib-windows.txt"))
Invoke-Checked $EnvPython @("-m", "pip", "check")
Invoke-Checked $EnvPython @((Join-Path $Root "scripts\verify_framework_environment.py"))
Invoke-Checked $EnvPython @("-c", "import qlib, pandas, pyarrow, plotly, webview; print('qlib', qlib.__version__, '| pywebview ok')")

Step "Preparing the data folder and pinned Qlib source"
if (-not $DataRoot) {
  $defaultRootJson = & $EnvPython -c "import json; from qlib_yahoo import DEFAULT_DATA_ROOT; print(json.dumps(DEFAULT_DATA_ROOT))"
  if ($LASTEXITCODE -ne 0) { Fail "Could not determine the default data folder (exit $LASTEXITCODE)." }
  $DataRoot = $defaultRootJson | ConvertFrom-Json -ErrorAction Stop
  $dataDrive = Split-Path -Qualifier $DataRoot -ErrorAction SilentlyContinue
  if ($dataDrive -and -not (Test-Path ($dataDrive + "\"))) {
    $DataRoot = Join-Path $env:USERPROFILE "Qlib"
    Write-Host "Default drive unavailable: using $DataRoot."
  }
}
$DataRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DataRoot)
New-Item -ItemType Directory -Force -Path $DataRoot -ErrorAction Stop | Out-Null
$env:TUBBY_QLIB_DATA_ROOT = $DataRoot  # this process only
Invoke-Checked $EnvPython @("-c", "from pathlib import Path; from qlib_yahoo import DEFAULT_DATA_ROOT; from qlib_yahoo.dataset import ensure_upstream; print('dump_bin:', ensure_upstream(Path(DEFAULT_DATA_ROOT)))")

if ($Signed) {
  Step "Installing and verifying the accepted signed runtime"
  Invoke-Checked $EnvPython @((Join-Path $Root "scripts\install_fast_runtime.py"), "--data-root", $DataRoot)
}

if (-not $NoShortcuts) {
  Step "Adding Start menu and Desktop shortcuts"
  $shortcutArguments = @("-m", "qlib_yahoo.webapp", "--create-shortcuts", "--data-root", $DataRoot)
  Invoke-Checked $EnvPython $shortcutArguments
}

Step "Done"
if (-not $NoShortcuts) {
  Write-Host "Launch: the 'Tubby Qlib Console' shortcut (Start menu or Desktop)."
}
Write-Host "Launch from PowerShell (copy this command):"
$launchCommand = "Set-Location -LiteralPath '$($Root.Replace("'", "''"))'; & '$($EnvPython.Replace("'", "''"))' -m qlib_yahoo.webapp --data-root '$($DataRoot.Replace("'", "''"))'"
Write-Host $launchCommand
Write-Host "Data folder: $DataRoot"
