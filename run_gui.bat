@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "PROJECT_SCRIPT=%PROJECT_ROOT%src\zed_mapping_svo_gui.py"
set "PRIMARY_PY=C:\zed_mapping\venv\Scripts\python.exe"
set "LOCAL_PY=%PROJECT_ROOT%.venv\Scripts\python.exe"
set "MPLCONFIGDIR=%PROJECT_ROOT%.mplconfig"

if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"

call :try_python "%PRIMARY_PY%"
if not errorlevel 1 exit /b %errorlevel%

call :try_python "%LOCAL_PY%"
if not errorlevel 1 exit /b %errorlevel%

echo No compatible Python environment found.
echo Expected a Python 3.12+ environment with PySide6, pyvista, and pyvistaqt installed.
echo For real ZED playback, install the ZED SDK so pyzed is available too.
echo Run setup_windows.bat or install the ZED SDK bindings into a supported Python interpreter.
exit /b 1

:try_python
set "PYTHON_EXE=%~1"
if not exist "%PYTHON_EXE%" exit /b 1
"%PYTHON_EXE%" -c "import importlib.util, sys; mods=('PySide6','pyvista','pyvistaqt'); sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" "%PROJECT_SCRIPT%"
exit /b %errorlevel%
