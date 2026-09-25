# Load the inbox Windows PowerShell ACL implementation by an exact path.  The
# process may have inherited PSModulePath from a pwsh parent, so module discovery
# is deliberately not used here.
$nativeSecurityManifest = Join-Path $PSHOME "Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1"
if (-not (Test-Path -LiteralPath $nativeSecurityManifest -PathType Leaf)) {
  throw "native Windows PowerShell Security module is missing: $nativeSecurityManifest"
}
$nativeSecurityModule = Import-Module -Name $nativeSecurityManifest -Force -PassThru -ErrorAction Stop
$nativeSecurityRoot = [System.IO.Path]::GetFullPath((Join-Path $PSHOME "Modules\Microsoft.PowerShell.Security"))
$loadedSecurityRoot = [System.IO.Path]::GetFullPath($nativeSecurityModule.ModuleBase)
if ($loadedSecurityRoot.TrimEnd('\') -cne $nativeSecurityRoot.TrimEnd('\')) {
  throw "native Windows PowerShell Security module path mismatch"
}
$nativeGetAcl = Get-Command -Name Get-Acl -Module Microsoft.PowerShell.Security -ErrorAction Stop |
  Where-Object { [System.IO.Path]::GetFullPath($_.Module.ModuleBase).TrimEnd('\') -ceq $nativeSecurityRoot.TrimEnd('\') } |
  Select-Object -First 1
if ($null -eq $nativeGetAcl) {
  throw "native Windows PowerShell Get-Acl command is unavailable"
}

function Test-NativePrincipalGrant([string]$Target, [string]$Principal, [string]$PrincipalSid) {
  if (-not $Target -or -not (Test-Path -LiteralPath $Target)) { return $false }
  $acl = & $nativeGetAcl -LiteralPath $Target
  return [bool]($acl.Access | Where-Object {
    $_.IdentityReference.Value -ieq $Principal -or
    ($PrincipalSid -and $_.IdentityReference.Value -ceq $PrincipalSid)
  })
}
