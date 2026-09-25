# Load the inbox Windows PowerShell ACL implementation by an exact path.  The
# process may have inherited PSModulePath from a pwsh parent, so module discovery
# is deliberately not used here.
function Test-WindowsPathIdentity([string]$Actual, [string]$Expected) {
  $actualRoot = [System.IO.Path]::GetFullPath($Actual).TrimEnd('\')
  $expectedRoot = [System.IO.Path]::GetFullPath($Expected).TrimEnd('\')
  return [StringComparer]::OrdinalIgnoreCase.Equals($actualRoot, $expectedRoot)
}

function Test-WindowsPathWithinRoot([string]$Path, [string]$Root) {
  $fullPath = [System.IO.Path]::GetFullPath($Path)
  $fullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd([char[]]@('\', '/'))
  $rootWithSeparator = $fullRoot + [System.IO.Path]::DirectorySeparatorChar
  return $fullPath.StartsWith($rootWithSeparator, [StringComparison]::OrdinalIgnoreCase)
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
if ($nativeSecurityModule.Name -cne "Microsoft.PowerShell.Security") {
  throw (New-NativeSecurityAuthorityFailure "imported Security module has an unexpected name")
}
if (-not (Test-WindowsPathIdentity $nativeSecurityModule.Path $nativeSecurityManifest)) {
  throw (New-NativeSecurityAuthorityFailure "imported Security module does not match the exact manifest")
}
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
if ($null -eq $nativeGetAcl.Module -or
    -not (Test-WindowsPathIdentity $nativeGetAcl.Module.Path $nativeSecurityManifest)) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl does not belong to the exact imported manifest")
}
if ($null -eq $nativeGetAcl.ImplementingType) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl implementing type is unavailable")
}
if ($nativeGetAcl.ImplementingType.FullName -cne "Microsoft.PowerShell.Commands.GetAclCommand") {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl has an unexpected implementing type")
}
if ($null -eq $nativeGetAcl.ImplementingType.Assembly) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl implementing assembly is unavailable")
}
$nativeGetAclAssemblyIdentity = "Microsoft.PowerShell.Security, Version=3.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35"
$nativeGetAclAssemblyObject = $nativeGetAcl.ImplementingType.Assembly
if ($nativeGetAclAssemblyObject.FullName -cne $nativeGetAclAssemblyIdentity) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl has an unexpected strong assembly identity")
}
$globalAssemblyCacheProperty = $nativeGetAclAssemblyObject.PSObject.Properties["GlobalAssemblyCache"]
if ($null -eq $globalAssemblyCacheProperty -or $nativeGetAclAssemblyObject.GlobalAssemblyCache -ne $true) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl assembly is not in the Global Assembly Cache")
}
$nativeGetAclAssembly = $nativeGetAclAssemblyObject.Location
if (-not $nativeGetAclAssembly) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl implementing assembly location is unavailable")
}
$nativeGetAclAssemblyFileExists = Test-Path -LiteralPath $nativeGetAclAssembly -PathType Leaf
if (-not $nativeGetAclAssemblyFileExists) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl implementing assembly file is unavailable")
}
if ([System.IO.Path]::GetFileName($nativeGetAclAssembly) -cne "Microsoft.PowerShell.Security.dll") {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl implementing assembly has an unexpected filename")
}
$nativeWindowsRoot = $PSHOME
for ($parentIndex = 0; $parentIndex -lt 3; $parentIndex++) {
  $nativeWindowsRoot = [System.IO.Directory]::GetParent($nativeWindowsRoot)
  if ($null -eq $nativeWindowsRoot) {
    throw (New-NativeSecurityAuthorityFailure "native Windows root cannot be derived from PSHOME")
  }
  $nativeWindowsRoot = $nativeWindowsRoot.FullName
}
$nativeGacSecurityRoot = Join-Path $nativeWindowsRoot "Microsoft.Net\assembly\GAC_MSIL\Microsoft.PowerShell.Security"
if (-not (Test-WindowsPathWithinRoot $nativeGetAclAssembly $nativeGacSecurityRoot)) {
  throw (New-NativeSecurityAuthorityFailure "Get-Acl assembly is outside the trusted native GAC root")
}
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
Write-Output "GET_ACL_GLOBAL_ASSEMBLY_CACHE=$($nativeGetAclAssemblyObject.GlobalAssemblyCache)"
Write-Output "NATIVE_GAC_SECURITY_ROOT=$nativeGacSecurityRoot"
Write-Output "SECURITY_EXECUTABLE_AUTHORITY_QUALIFIED=True"
Write-Output "PS_EDITION=$($PSVersionTable.PSEdition)"
Write-Output "PS_HOME=$PSHOME"
function Test-NativePrincipalGrant([string]$Target, [string]$Principal, [string]$PrincipalSid) {
  if (-not $Target -or -not (Test-Path -LiteralPath $Target)) { return $false }
  $acl = & $nativeGetAcl -LiteralPath $Target
  return [bool]($acl.Access | Where-Object {
    $_.IdentityReference.Value -ieq $Principal -or
    ($PrincipalSid -and $_.IdentityReference.Value -ceq $PrincipalSid)
  })
}
