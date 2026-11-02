param(
  [Parameter(Mandatory)][ValidateSet("prepare-reboot", "qualify")][string]$Mode,
  [Parameter(Mandatory)][string]$State,
  [string]$QualifiedManifest,
  [string]$CleanInstallReceipt,
  [int]$SoakHours = 24,
  [string]$Output
)
$ErrorActionPreference = "Stop"

# Zgodne parametry zachowuja reczny workflow, ale nie zezwalaja na restart ani soak.
# Przywrocenie wykonania wymaga niezaleznego autorytetu dla kazdego dowodu,
# istniejacej kwalifikacji Production Trust i osobnej autoryzacji fizycznej proby.
# Readiness JSON i zapisany raport spojnosci artefaktu nie sa takim autorytetem.
if ($Mode -ceq "qualify" -and [string]::IsNullOrWhiteSpace($Output)) {
  throw "qualify wymaga Output"
}
$reportPath = if ($Mode -ceq "prepare-reboot") { $State } else { $Output }
python -m deployment.windows_stage10_proof_validation --output $reportPath
if ($LASTEXITCODE -ne 0) {
  throw "WINDOWS_STAGE10_NATIVE_PROOF_AUTHORITY_UNAVAILABLE"
}
# Takze nieoczekiwany sukces procesu pomocniczego nie moze odblokowac rebootu.
throw "WINDOWS_STAGE10_NATIVE_PROOF_AUTHORITY_UNAVAILABLE"
