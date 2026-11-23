$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (Test-Path -LiteralPath $venvPython) {
    & $venvPython (Join-Path $PSScriptRoot 'run_ui.py')
} elseif (Test-Path -LiteralPath $bundledPython) {
    & $bundledPython (Join-Path $PSScriptRoot 'run_ui.py')
} else {
    python (Join-Path $PSScriptRoot 'run_ui.py')
}
exit $LASTEXITCODE
