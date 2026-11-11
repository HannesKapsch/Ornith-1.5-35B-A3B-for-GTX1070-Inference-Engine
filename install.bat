@echo off
rem ornith-serve installer for Windows. Usage:  install.bat --dir D:\ornith   (all options: install.bat --help)
rem Uses Python 3.10-3.12 if one is installed; otherwise it fetches uv (one file, github.com/astral-sh/uv) and a private
rem Python 3.12 into .tools next to this file. Nothing else on the PC is changed.
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "PY="
if defined ORNITH_FORCE_PRIVATE_PYTHON goto private
for %%v in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%v -c "import venv" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys, venv; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1 && set "PY=python"
)
if defined PY goto run

:private
set "TOOLS=%~dp0.tools"
set "UV_PYTHON_INSTALL_DIR=%TOOLS%\python"
set "UV_CACHE_DIR=%TOOLS%\cache"
if not exist "%TOOLS%\uv.exe" (
  echo No Python 3.10-3.12 found: downloading uv to get a private Python 3.12 ...
  if not exist "%TOOLS%" mkdir "%TOOLS%"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '%TOOLS%\uv.zip'; Expand-Archive '%TOOLS%\uv.zip' '%TOOLS%' -Force; Remove-Item '%TOOLS%\uv.zip'"
  if errorlevel 1 ( echo ERROR: could not download uv. Install Python 3.12 from python.org and run this again. & exit /b 1 )
)
"%TOOLS%\uv.exe" python install 3.12
if errorlevel 1 ( echo ERROR: uv could not install Python 3.12 & exit /b 1 )
rem the private one under .tools, never a Python that happens to be installed elsewhere
for /f "delims=" %%p in ('dir /s /b "%TOOLS%\python\python.exe" 2^>nul') do if not defined PY set "PY=%%p"
if not defined PY ( echo ERROR: the private Python was not found & exit /b 1 )
set "PY="!PY!""

:run
echo Using Python: !PY!
!PY! "%~dp0install.py" %*
exit /b %errorlevel%
