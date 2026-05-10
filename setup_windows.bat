@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE="
set "SETUP_TEMP=%PROJECT_ROOT%.setup_tmp"

call :probe_path "C:\zed_mapping\python313\python.exe"
if not defined PYTHON_EXE call :probe_cmd "py -3.10"
if not defined PYTHON_EXE call :probe_cmd "py -3.11"
if not defined PYTHON_EXE call :probe_cmd "py -3.12"
if not defined PYTHON_EXE call :probe_cmd "py -3.13"
if not defined PYTHON_EXE call :probe_cmd "py -3"
if not defined PYTHON_EXE call :probe_cmd "python"

if not defined PYTHON_EXE (
  echo Install Python 3.10 or newer, then rerun this script.
  exit /b 1
)

if not exist "%SETUP_TEMP%" mkdir "%SETUP_TEMP%"
set "TEMP=%SETUP_TEMP%"
set "TMP=%SETUP_TEMP%"

if exist "%PROJECT_ROOT%.venv" rmdir /s /q "%PROJECT_ROOT%.venv"

if /i "%PYTHON_EXE:~0,3%"=="py " (
  %PYTHON_EXE% -m venv "%PROJECT_ROOT%.venv"
) else (
  "%PYTHON_EXE%" -m venv "%PROJECT_ROOT%.venv"
)
if errorlevel 1 exit /b 1

"%PROJECT_ROOT%.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%PROJECT_ROOT%.venv\Scripts\python.exe" -m pip install -r "%PROJECT_ROOT%requirements.txt"
if errorlevel 1 exit /b 1

if exist "C:\zed_mapping\venv\Lib\site-packages\pyzed" (
  > "%PROJECT_ROOT%.venv\Lib\site-packages\zed_shared_env.pth" echo C:\zed_mapping\venv\Lib\site-packages
  echo Linked pyzed from C:\zed_mapping\venv\Lib\site-packages via .pth file.
)

if exist "C:\Users\fyp34\AppData\Local\Programs\Python\Python310\Lib\site-packages\pyzed" (
  >> "%PROJECT_ROOT%.venv\Lib\site-packages\zed_shared_env.pth" echo C:\Users\fyp34\AppData\Local\Programs\Python\Python310\Lib\site-packages
  echo Linked pyzed from C:\Users\fyp34\AppData\Local\Programs\Python\Python310\Lib\site-packages via .pth file.
)

exit /b %errorlevel%

:probe_path
if exist "%~1" (
  "%~1" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if not errorlevel 1 set "PYTHON_EXE=%~1"
)
exit /b 0

:probe_cmd
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set "PYTHON_EXE=%~1"
exit /b 0
