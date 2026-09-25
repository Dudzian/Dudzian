param(
  [Parameter(Mandatory = $true)]
  [string]$PythonExecutable
)

$ErrorActionPreference = "Stop"
$probe = Join-Path $PSScriptRoot "windows_scm_probe.ps1"
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
  $probe, [ref]$tokens, [ref]$parseErrors
)
if ($parseErrors.Count -ne 0) {
  throw "windows_scm_probe.ps1 has parser errors"
}
$helper = @(
  $ast.FindAll(
    {
      param($node)
      $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -ceq "Invoke-CapturedPython"
    },
    $true
  )
)
if ($helper.Count -ne 1) { throw "Invoke-CapturedPython definition is not unique" }
. ([scriptblock]::Create($helper[0].Extent.Text))

$stderrRun = Invoke-CapturedPython -Arguments @(
  "-c",
  "import sys; print('stderr-line-1', file=sys.stderr); print('stderr-line-2', file=sys.stderr); print('stderr-line-3', file=sys.stderr); raise SystemExit(7)"
)
if ($stderrRun.ExitCode -ne 7) { throw "native non-zero exit code was not preserved" }
foreach ($line in @("stderr-line-1", "stderr-line-2", "stderr-line-3")) {
  if ($stderrRun.Output -cnotcontains $line) { throw "native stderr line was not preserved: $line" }
}
if ($ErrorActionPreference -cne "Stop") { throw "ErrorActionPreference was not restored" }

$successRun = Invoke-CapturedPython -Arguments @(
  "-c",
  "import json; print(json.dumps({'captured_native_transport': 'PASS'}))"
)
if ($successRun.ExitCode -ne 0) { throw "native success exit code was not preserved" }
$payload = $successRun.Output[-1] | ConvertFrom-Json
if ($payload.captured_native_transport -cne "PASS") { throw "native stdout JSON was damaged" }
if ($ErrorActionPreference -cne "Stop") { throw "ErrorActionPreference was not restored" }

Write-Output "WINDOWS_STAGE4_CAPTURED_NATIVE_TRANSPORT=PASS"
