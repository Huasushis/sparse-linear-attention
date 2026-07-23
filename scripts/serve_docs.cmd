@echo off
setlocal
REM Resolve the repository from this script so the checkout can move.
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "DOCS_PYTHON=%REPO_ROOT%\.venv-docs\Scripts\python.exe"
if not exist "%DOCS_PYTHON%" (
  echo Missing %DOCS_PYTHON%. Create the docs environment from README.md first.
  exit /b 1
)
start "sla-mkdocs" /b "%DOCS_PYTHON%" -m mkdocs serve --config-file "%REPO_ROOT%\mkdocs.yml" --dev-addr 127.0.0.1:8000
exit /b 0
