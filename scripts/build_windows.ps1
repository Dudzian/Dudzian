param(
  [string]$OutputRoot = "build/output",
  [string]$LogPath = "build/reports/build.log"
)
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repo
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$distPath = Join-Path $OutputRoot "CryptoHunter"
$workPath = "build/pyinstaller-work"
if (Test-Path $distPath) { Remove-Item -Recurse -Force $distPath }
if (Test-Path $workPath) { Remove-Item -Recurse -Force $workPath }
$arguments = @(
  "--noconfirm",
  "--clean",
  "--distpath", $OutputRoot,
  "--workpath", $workPath,
  "CryptoHunter.spec"
)
& python -m PyInstaller @arguments *>&1 | Tee-Object -FilePath $LogPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$exe = Join-Path $distPath "CryptoHunter.exe"
if (-not (Test-Path $exe)) { throw "Missing build output: $exe" }
