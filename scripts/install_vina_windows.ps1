# Install AutoDock Vina (Windows) for AutoScreen
# Downloads official vina_1.2.7_win.exe into tools/bin/vina.exe
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Bin = Join-Path $Root "tools\bin"
New-Item -ItemType Directory -Force -Path $Bin | Out-Null
$Dest = Join-Path $Bin "vina.exe"
$Url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_win.exe"
Write-Host "Downloading $Url"
curl.exe -L --retry 3 -o $Dest $Url
& $Dest --version
Write-Host "Installed: $Dest"
Write-Host "Also need OpenBabel on PATH (e.g. pip install openbabel-wheel)."
Write-Host "Demo receptor: data/receptors/1iep_receptor.pdbqt (see docs/vina_setup.md)."
