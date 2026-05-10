$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot '.venv'
$SetupTemp = Join-Path $ProjectRoot '.setup_tmp'

function Test-PythonVersion {
    param([string[]]$CommandParts)
    try {
        $args = @()
        if ($CommandParts.Length -gt 1) {
            $args += $CommandParts[1..($CommandParts.Length - 1)]
        }
        $args += '-c', "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
        & $CommandParts[0] @args *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

$PythonCommand = $null
$Candidates = @(
    @('C:\zed_mapping\python313\python.exe'),
    @('py', '-3.10'),
    @('py', '-3.11'),
    @('py', '-3.12'),
    @('py', '-3.13'),
    @('py', '-3'),
    @('python')
)

foreach ($candidate in $Candidates) {
    if (Test-PythonVersion $candidate) {
        $PythonCommand = $candidate
        break
    }
}

if (-not $PythonCommand) {
    Write-Host 'Install Python 3.10 or newer, then rerun this script.'
    exit 1
}

New-Item -ItemType Directory -Force -Path $SetupTemp | Out-Null
$env:TEMP = $SetupTemp
$env:TMP = $SetupTemp

if (Test-Path $VenvPath) {
    Remove-Item -Recurse -Force $VenvPath
}

$venvArgs = @()
if ($PythonCommand.Length -gt 1) {
    $venvArgs += $PythonCommand[1..($PythonCommand.Length - 1)]
}
$venvArgs += '-m', 'venv', $VenvPath
& $PythonCommand[0] @venvArgs

$ProjectPython = Join-Path $VenvPath 'Scripts\python.exe'
& $ProjectPython -m pip install --upgrade pip
& $ProjectPython -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')

if (Test-Path 'C:\zed_mapping\venv\Lib\site-packages\pyzed') {
    $pthPath = Join-Path $VenvPath 'Lib\site-packages\zed_shared_env.pth'
    Set-Content -Path $pthPath -Value 'C:\zed_mapping\venv\Lib\site-packages' -Encoding ASCII
    Write-Host 'Linked pyzed from C:\zed_mapping\venv\Lib\site-packages via .pth file.'
}

if (Test-Path 'C:\Users\fyp34\AppData\Local\Programs\Python\Python310\Lib\site-packages\pyzed') {
    $pthPath = Join-Path $VenvPath 'Lib\site-packages\zed_shared_env.pth'
    Add-Content -Path $pthPath -Value 'C:\Users\fyp34\AppData\Local\Programs\Python\Python310\Lib\site-packages' -Encoding ASCII
    Write-Host 'Linked pyzed from C:\Users\fyp34\AppData\Local\Programs\Python\Python310\Lib\site-packages via .pth file.'
}

exit $LASTEXITCODE
