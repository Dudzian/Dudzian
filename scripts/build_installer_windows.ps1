param(
    [string]$Profile = "",
    [string]$Version = "0.0.0-dev"
)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Join-Path $ScriptRoot ".." | Resolve-Path

if (-not $Profile) {
    $Profile = Join-Path $RepoRoot "deploy/packaging/profiles/windows.toml"
}

if (-not (Test-Path -LiteralPath $Profile)) {
    Write-Error "Brak profilu: $Profile"
    exit 1
}

$argsList = @(
    "--profile", (Resolve-Path $Profile).Path,
    "--version", $Version,
    "--platform", "windows"
)

if ($args.Count -gt 0) {
    $argsList += $args
}

python (Join-Path $RepoRoot "scripts/build_installer_from_profile.py") @argsList
