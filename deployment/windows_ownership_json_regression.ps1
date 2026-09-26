$ErrorActionPreference = "Stop"
$probe = Join-Path $PSScriptRoot "windows_scm_probe.ps1"
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
  $probe, [ref]$tokens, [ref]$parseErrors
)
if ($parseErrors.Count -ne 0) { throw "windows_scm_probe.ps1 has parser errors" }
$saveHelpers = @(
  $ast.FindAll(
    {
      param($node)
      $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -ceq "Save-OwnershipRecord"
    },
    $true
  )
)
if ($saveHelpers.Count -ne 1) { throw "Save-OwnershipRecord definition is not unique" }
$saveSource = $saveHelpers[0].Extent.Text
$depthMatch = [regex]::Match($saveSource, 'ConvertTo-Json\s+-Depth\s+(\d+)')
if (-not $depthMatch.Success -or [int]$depthMatch.Groups[1].Value -lt 10) {
  throw "Save-OwnershipRecord does not preserve the frozen ownership schema depth"
}

$configPolicy = [System.Collections.ArrayList]::new()
[void]$configPolicy.Add(@("sid-a", 1, 3))
[void]$configPolicy.Add(@("sid-b", 2, 3))
[void]$configPolicy.Add(@("sid-c", 3, 3))
$statePolicy = [System.Collections.ArrayList]::new()
[void]$statePolicy.Add(@("sid-a", 4, 3))
[void]$statePolicy.Add(@("sid-b", 5, 3))
[void]$statePolicy.Add(@("sid-c", 6, 3))
$ownership = [ordered]@{
  ownership_phase = "SERVICE_PROVEN"
  service_sid = "S-1-5-80-123"
  grants = @("C:\ProgramData\CryptoHunter", "C:\ProgramData\CryptoHunter\scm-health.txt")
  path_security_plan = [ordered]@{
    targets = @(
      [ordered]@{ role = "CONFIG"; path = "C:\ProgramData\CryptoHunter\Config" }
      [ordered]@{ role = "STATE"; path = "C:\ProgramData\CryptoHunter\State" }
      [ordered]@{ role = "RUNTIME"; path = "C:\ProgramData\CryptoHunter\Runtime" }
    )
    config_policy = $configPolicy
    state_policy = $statePolicy
  }
}
$roundTripped = $ownership | ConvertTo-Json -Depth 10 -Compress | ConvertFrom-Json
if ($roundTripped.ownership_phase -cne "SERVICE_PROVEN") { throw "ownership_phase changed" }
if ($roundTripped.service_sid -cne "S-1-5-80-123") { throw "service_sid changed" }
if (@($roundTripped.grants).Count -ne 2) { throw "grants changed" }
if ($roundTripped.grants[0] -cne "C:\ProgramData\CryptoHunter" -or
    $roundTripped.grants[1] -cne "C:\ProgramData\CryptoHunter\scm-health.txt") {
  throw "grant identities changed"
}
$expectedRoles = @("CONFIG", "STATE", "RUNTIME")
$expectedPaths = @(
  "C:\ProgramData\CryptoHunter\Config",
  "C:\ProgramData\CryptoHunter\State",
  "C:\ProgramData\CryptoHunter\Runtime"
)
$targets = @($roundTripped.path_security_plan.targets)
if ($targets.Count -ne 3) { throw "target count changed" }
for ($i = 0; $i -lt 3; $i++) {
  if ($targets[$i].role -cne $expectedRoles[$i] -or
      $targets[$i].path -cne $expectedPaths[$i]) { throw "target identity changed" }
}
foreach ($policyName in @("config_policy", "state_policy")) {
  $actual = @($roundTripped.path_security_plan.$policyName)
  if ($actual.Count -ne 3) { throw "$policyName outer array changed" }
  for ($i = 0; $i -lt 3; $i++) {
    if (@($actual[$i]).Count -ne 3) { throw "$policyName nested array changed" }
    $expectedSid = "sid-$([char](97 + $i))"
    $expectedMask = if ($policyName -ceq "config_policy") { $i + 1 } else { $i + 4 }
    if ($actual[$i][0] -cne $expectedSid -or
        [int]$actual[$i][1] -ne $expectedMask -or
        [int]$actual[$i][2] -ne 3) {
      throw "$policyName value changed"
    }
  }
}

Write-Output "WINDOWS_STAGE4_PATH_SECURITY_PLAN_ROUNDTRIP=PASS"
Write-Output "WINDOWS_STAGE4_OWNERSHIP_JSON_DEPTH=PASS"
