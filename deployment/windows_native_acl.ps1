# Load the inbox Windows PowerShell ACL implementation by an exact path.  The
# process may have inherited PSModulePath from a pwsh parent, so module discovery
# is deliberately not used here.
function Test-WindowsPathIdentity([string]$Actual, [string]$Expected) {
  $actualRoot = [System.IO.Path]::GetFullPath($Actual).TrimEnd('\')
  $expectedRoot = [System.IO.Path]::GetFullPath($Expected).TrimEnd('\')
  return [StringComparer]::OrdinalIgnoreCase.Equals($actualRoot, $expectedRoot)
}

function New-NativeSecurityPathFailure([string]$ActualRoot) {
  return @(
    "EXPECTED_NATIVE_SECURITY_ROOT=$nativeSecurityRoot"
    "ACTUAL_SECURITY_MODULE_BASE=$ActualRoot"
    "CHILD_PS_EDITION=$($PSVersionTable.PSEdition)"
    "CHILD_PS_HOME=$PSHOME"
  ) -join [Environment]::NewLine
}

$nativeSecurityManifest = Join-Path $PSHOME "Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1"
if (-not (Test-Path -LiteralPath $nativeSecurityManifest -PathType Leaf)) {
  throw "native Windows PowerShell Security module is missing: $nativeSecurityManifest"
}
$nativeSecurityModule = Import-Module -Name $nativeSecurityManifest -Force -PassThru -ErrorAction Stop
$nativeSecurityRoot = [System.IO.Path]::GetFullPath((Join-Path $PSHOME "Modules\Microsoft.PowerShell.Security"))
$loadedSecurityRoot = [System.IO.Path]::GetFullPath($nativeSecurityModule.ModuleBase)
if (-not (Test-WindowsPathIdentity $loadedSecurityRoot $nativeSecurityRoot)) {
  throw (New-NativeSecurityPathFailure $loadedSecurityRoot)
}
$nativeGetAcl = Get-Command -Name Get-Acl -Module Microsoft.PowerShell.Security -ErrorAction Stop |
  Where-Object { Test-WindowsPathIdentity $_.Module.ModuleBase $nativeSecurityRoot } |
  Select-Object -First 1
if ($null -eq $nativeGetAcl) {
  $getAclCommand = Get-Command -Name Get-Acl -Module Microsoft.PowerShell.Security -ErrorAction SilentlyContinue |
    Select-Object -First 1
  $getAclModuleBase = if ($null -eq $getAclCommand) { "<UNAVAILABLE>" } else { $getAclCommand.Module.ModuleBase }
  throw (New-NativeSecurityPathFailure $getAclModuleBase)
}

function Test-NativePrincipalGrant([string]$Target, [string]$Principal, [string]$PrincipalSid) {
  if (-not $Target -or -not (Test-Path -LiteralPath $Target)) { return $false }
  $acl = & $nativeGetAcl -LiteralPath $Target
  return [bool]($acl.Access | Where-Object {
    $_.IdentityReference.Value -ieq $Principal -or
    ($PrincipalSid -and $_.IdentityReference.Value -ceq $PrincipalSid)
  })
}
