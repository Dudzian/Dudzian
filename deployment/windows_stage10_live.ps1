param(
  [Parameter(Mandatory)][ValidateSet("prepare-reboot", "qualify")][string]$Mode,
  [Parameter(Mandatory)][string]$State,
  [string]$QualifiedManifest,
  [string]$CleanInstallReceipt,
  [int]$SoakHours = 24,
  [string]$Output,
  [string]$ApprovedDiagnosticDirectory,
  [string]$DiagnosticOutput
)
$ErrorActionPreference = "Stop"

# Zgodne parametry zachowuja reczny workflow, ale nie zezwalaja na restart ani soak.
# Przywrocenie wykonania wymaga niezaleznego autorytetu dla kazdego dowodu,
# istniejacej kwalifikacji Production Trust i osobnej autoryzacji fizycznej proby.
# Readiness JSON i zapisany raport spojnosci artefaktu nie sa takim autorytetem.
# Legacy State i Output nie sa sciezkami diagnostyki i pozostaja nietkniete.
# Ich nazwy oraz manifest i receipt moga byc przekazane tylko jako chronione wejscia.
# Domyslnie drukujemy BLOCKED na stdout; zapis wymaga osobnego zatwierdzonego
# katalogu. Helper kwalifikuje jego wlasciciela i uprawnienia przed publikacja.
if (-not [string]::IsNullOrWhiteSpace($DiagnosticOutput) -and
    [string]::IsNullOrWhiteSpace($ApprovedDiagnosticDirectory)) {
  throw "DiagnosticOutput wymaga ApprovedDiagnosticDirectory"
}
$diagnosticArguments = @("-m", "deployment.windows_stage10_proof_validation")
if (-not [string]::IsNullOrWhiteSpace($ApprovedDiagnosticDirectory)) {
  $diagnosticArguments += @("--approved-output-directory", $ApprovedDiagnosticDirectory)
  foreach ($protectedInput in @($State, $Output, $QualifiedManifest, $CleanInstallReceipt)) {
    if (-not [string]::IsNullOrWhiteSpace($protectedInput)) {
      $diagnosticArguments += @("--protected-input", $protectedInput)
    }
  }
  if (-not [string]::IsNullOrWhiteSpace($DiagnosticOutput)) {
    $diagnosticArguments += @("--output", $DiagnosticOutput)
  }
}
python @diagnosticArguments
if ($LASTEXITCODE -ne 0) {
  throw "WINDOWS_STAGE10_NATIVE_PROOF_AUTHORITY_UNAVAILABLE"
}
# Takze nieoczekiwany sukces procesu pomocniczego nie moze odblokowac rebootu.
throw "WINDOWS_STAGE10_NATIVE_PROOF_AUTHORITY_UNAVAILABLE"
