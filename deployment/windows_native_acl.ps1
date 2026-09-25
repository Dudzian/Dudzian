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
    "CHILD_PS_EDITION=$($PSVersionTable.PSEdition)"
    "CHILD_PS_HOME=$PSHOME"
  ) -join [Environment]::NewLine
}

if ($PSVersionTable.PSEdition -cne "Desktop") {
  throw "native ACL authority must run in Windows PowerShell Desktop"
}

$nativeSecurityManifest = Join-Path $PSHOME "Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1"
if (-not (Test-Path -LiteralPath $nativeSecurityManifest -PathType Leaf)) {
  throw "native Windows PowerShell Security module is missing: $nativeSecurityManifest"
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
if ($null -eq $nativeGetAcl.ImplementingType) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl implementing type is unavailable")
}
if ($null -eq $nativeGetAcl.ImplementingType.Assembly) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl implementing assembly is unavailable")
}
$nativeGetAclAssembly = $nativeGetAcl.ImplementingType.Assembly.Location
if (-not $nativeGetAclAssembly) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl implementing assembly location is unavailable")
}
$nativeGetAclAssemblyFileExists = Test-Path -LiteralPath $nativeGetAclAssembly -PathType Leaf
Write-Output "SECURITY_IMPORT_NAME=$($nativeSecurityModule.Name)"
Write-Output "SECURITY_IMPORT_MODULE_BASE=$($nativeSecurityModule.ModuleBase)"
Write-Output "SECURITY_IMPORT_PATH=$($nativeSecurityModule.Path)"
Write-Output "GET_ACL_COMMAND_TYPE=$($nativeGetAcl.CommandType)"
Write-Output "GET_ACL_MODULE_NAME=$($nativeGetAcl.ModuleName)"
Write-Output "GET_ACL_MODULE_BASE=$($nativeGetAcl.Module.ModuleBase)"
Write-Output "GET_ACL_MODULE_PATH=$($nativeGetAcl.Module.Path)"
Write-Output "GET_ACL_IMPLEMENTING_TYPE=$($nativeGetAcl.ImplementingType.FullName)"
Write-Output "GET_ACL_ASSEMBLY_FULL_NAME=$($nativeGetAcl.ImplementingType.Assembly.FullName)"
Write-Output "GET_ACL_ASSEMBLY_LOCATION=$nativeGetAclAssembly"
Write-Output "GET_ACL_ASSEMBLY_FILE_EXISTS=$nativeGetAclAssemblyFileExists"
Write-Output "PS_EDITION=$($PSVersionTable.PSEdition)"
Write-Output "PS_HOME=$PSHOME"
if (-not $nativeGetAclAssemblyFileExists) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl implementing assembly file is unavailable")
}

# The exact manifest proves discovery provenance, but it does not expose a stable
# executable-file authority against which Assembly.Location can yet be compared.
# Keep the production boundary closed until a live runner supplies that layout.
throw (New-NativeSecurityAuthorityFailure "NATIVE_SECURITY_EXECUTABLE_AUTHORITY_LAYOUT_UNQUALIFIED")

function Test-NativePrincipalGrant([string]$Target, [string]$Principal, [string]$PrincipalSid) {
  if (-not $Target -or -not (Test-Path -LiteralPath $Target)) { return $false }
  $acl = & $nativeGetAcl -LiteralPath $Target
  return [bool]($acl.Access | Where-Object {
    $_.IdentityReference.Value -ieq $Principal -or
    ($PrincipalSid -and $_.IdentityReference.Value -ceq $PrincipalSid)
  })
}
