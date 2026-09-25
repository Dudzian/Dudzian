$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_native_acl.ps1")

if ($PSVersionTable.PSEdition -cne "Desktop") {
  throw "ACL child is not native Windows PowerShell"
}
Write-Output "SECURITY_IMPORT_NAME=$($nativeSecurityModule.Name)"
Write-Output "SECURITY_IMPORT_MODULE_BASE=$($nativeSecurityModule.ModuleBase)"
Write-Output "SECURITY_IMPORT_PATH=$($nativeSecurityModule.Path)"
Write-Output "GET_ACL_COMMAND_TYPE=$($nativeGetAcl.CommandType)"
Write-Output "GET_ACL_MODULE_NAME=$($nativeGetAcl.ModuleName)"
Write-Output "GET_ACL_MODULE_BASE=$($nativeGetAcl.Module.ModuleBase)"
Write-Output "GET_ACL_MODULE_PATH=$($nativeGetAcl.Module.Path)"
Write-Output "GET_ACL_ASSEMBLY_LOCATION=$nativeGetAclAssembly"
Write-Output "PS_EDITION=$($PSVersionTable.PSEdition)"
Write-Output "PS_HOME=$PSHOME"
$principal = "NT AUTHORITY\LOCAL SERVICE"
$principalSid = ([System.Security.Principal.NTAccount]::new($principal)).Translate(
  [System.Security.Principal.SecurityIdentifier]
).Value
$temporary = Join-Path ([System.IO.Path]::GetTempPath()) ("cryptohunter-acl-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
  $before = Test-NativePrincipalGrant $temporary $principal $principalSid
  if ($before) { throw "temporary ACL unexpectedly contains the regression principal" }
  & icacls.exe $temporary /grant "${principal}:RX" | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "controlled ACL grant failed" }
  $afterGrant = Test-NativePrincipalGrant $temporary $principal $principalSid
  & icacls.exe $temporary /remove:g $principal | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "controlled ACL removal failed" }
  $afterRemove = Test-NativePrincipalGrant $temporary $principal $principalSid
  [ordered]@{
    parent_edition = $env:WINDOWS_ACL_REGRESSION_PARENT_EDITION
    child_edition = $PSVersionTable.PSEdition
    child_pshome = $PSHOME
    security_module_path = $nativeSecurityModule.Path
    get_acl_assembly_location = $nativeGetAclAssembly
    expected_security_assembly = $nativeSecurityAssembly
    explicit_native_import = $true
    before = $before
    after_grant = $afterGrant
    after_remove = $afterRemove
  } | ConvertTo-Json -Compress | Write-Output
} finally {
  if (Test-Path -LiteralPath $temporary) {
    & icacls.exe $temporary /remove:g $principal 2>$null | Out-Null
    Remove-Item -LiteralPath $temporary -Recurse -Force
  }
}
