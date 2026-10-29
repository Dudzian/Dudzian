function Assert-Stage10SoakIdentity {
  param(
    [Parameter(Mandatory)][hashtable]$Expected,
    [Parameter(Mandatory)][hashtable]$Observed
  )
  if ($Expected.pid -isnot [int] -or $Expected.pid -le 0 -or
      $Observed.pid -isnot [int] -or $Observed.pid -le 0 -or
      $Expected.started -isnot [string] -or [string]::IsNullOrWhiteSpace($Expected.started) -or
      $Observed.started -isnot [string] -or [string]::IsNullOrWhiteSpace($Observed.started) -or
      $Observed.pid -ne $Expected.pid -or $Observed.started -cne $Expected.started) {
    throw "service instance changed or is missing during the 24-hour lifecycle qualification"
  }
}
