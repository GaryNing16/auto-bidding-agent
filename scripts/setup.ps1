param([string]$Python = "py")
$ErrorActionPreference = "Stop"
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($Python -eq 'py') { & $Python -3.14 -m venv .venv }
    else { & $Python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create project venv' }
}
& '.\.venv\Scripts\python.exe' -X utf8 -m pip install -r requirements-harmony-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency install failed' }
& '.\.venv\Scripts\python.exe' -X utf8 -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
