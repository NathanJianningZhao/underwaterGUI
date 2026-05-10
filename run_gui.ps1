$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectScript = Join-Path $ProjectRoot 'src\zed_mapping_svo_gui.py'
$MplConfigDir = Join-Path $ProjectRoot '.mplconfig'

New-Item -ItemType Directory -Force -Path $MplConfigDir | Out-Null
$env:MPLCONFIGDIR = $MplConfigDir

function Test-GuiPython {
    param([string]$PythonExe)
    if (-not $PythonExe -or -not (Test-Path $PythonExe)) {
        return $false
    }
    & $PythonExe -c "import importlib.util, sys; mods=('PySide6','pyvista','pyvistaqt'); sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)"
    return ($LASTEXITCODE -eq 0)
}

$candidates = @(
    'C:\zed_mapping\venv\Scripts\python.exe',
    (Join-Path $ProjectRoot '.venv\Scripts\python.exe')
)

$PythonExe = $null
foreach ($candidate in $candidates) {
    if (Test-GuiPython $candidate) {
        $PythonExe = $candidate
        break
    }
}

if (-not $PythonExe) {
    Write-Host 'No compatible Python environment found.'
    Write-Host 'Expected a Python 3.12+ environment with PySide6, pyvista, and pyvistaqt installed.'
    Write-Host 'For real ZED playback, install the ZED SDK so pyzed is available too.'
    Write-Host 'Run setup_windows.ps1 or install the ZED SDK bindings into a supported Python interpreter.'
    exit 1
}

& $PythonExe $ProjectScript
exit $LASTEXITCODE
