# Load the inbox Windows PowerShell ACL implementation by an exact path.  The
# process may have inherited PSModulePath from a pwsh parent, so module discovery
# is deliberately not used here.
function Test-WindowsPathIdentity([string]$Actual, [string]$Expected) {
  $actualRoot = [System.IO.Path]::GetFullPath($Actual).TrimEnd('\')
  $expectedRoot = [System.IO.Path]::GetFullPath($Expected).TrimEnd('\')
  return [StringComparer]::OrdinalIgnoreCase.Equals($actualRoot, $expectedRoot)
}

function New-NativeSecurityAuthorityFailure([string]$Reason) {
  return @(
    "NATIVE_SECURITY_AUTHORITY_FAILURE=$Reason"
    "EXPECTED_NATIVE_SECURITY_MANIFEST=$nativeSecurityManifest"
    "EXPECTED_NATIVE_SECURITY_ASSEMBLY=$nativeSecurityAssembly"
    "CHILD_PS_EDITION=$($PSVersionTable.PSEdition)"
    "CHILD_PS_HOME=$PSHOME"
  ) -join [Environment]::NewLine
}

$nativeSecurityManifest = Join-Path $PSHOME "Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1"
$nativeSecurityAssembly = Join-Path $PSHOME "Microsoft.PowerShell.Security.dll"
if (-not (Test-Path -LiteralPath $nativeSecurityManifest -PathType Leaf)) {
  throw "native Windows PowerShell Security module is missing: $nativeSecurityManifest"
}
if (-not (Test-Path -LiteralPath $nativeSecurityAssembly -PathType Leaf)) {
  throw "native Windows PowerShell Security assembly is missing: $nativeSecurityAssembly"
}
$nativeSecurityModule = Import-Module -Name $nativeSecurityManifest -Force -PassThru -ErrorAction Stop
$nativeGetAcl = Get-Command -Name Get-Acl -Module Microsoft.PowerShell.Security -ErrorAction Stop |
  Select-Object -First 1
if ($null -eq $nativeGetAcl) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl is unavailable after exact manifest import")
}
if ($nativeGetAcl.CommandType -ne [System.Management.Automation.CommandTypes]::Cmdlet) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl is not a native cmdlet")
}
if ($nativeGetAcl.ModuleName -cne "Microsoft.PowerShell.Security") {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl has an unexpected module name")
}
$nativeGetAclAssembly = $nativeGetAcl.ImplementingType.Assembly.Location
if (-not $nativeGetAclAssembly) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl implementing assembly is unavailable")
}
if (-not (Test-WindowsPathIdentity $nativeGetAclAssembly $nativeSecurityAssembly)) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl implementing assembly is not the native Security assembly")
}

function Test-NativePrincipalGrant([string]$Target, [string]$Principal, [string]$PrincipalSid) {
  if (-not $Target -or -not (Test-Path -LiteralPath $Target)) { return $false }
  $acl = & $nativeGetAcl -LiteralPath $Target
  return [bool]($acl.Access | Where-Object {
    $_.IdentityReference.Value -ieq $Principal -or
    ($PrincipalSid -and $_.IdentityReference.Value -ceq $PrincipalSid)
  })
}
