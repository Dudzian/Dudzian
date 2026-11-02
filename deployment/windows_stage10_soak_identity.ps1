# Porownanie dwoch punktowych obserwacji nie dowodzi ciaglosci procesu ani 24h pracy.
function Assert-Stage10SoakIdentity {
  param(
    [Parameter(Mandatory)][hashtable]$Expected,
    [Parameter(Mandatory)][hashtable]$Observed
  )
  foreach ($snapshot in @($Expected, $Observed)) {
    $parsedStart = [System.DateTimeOffset]::MinValue
    $validStart = $snapshot.started -is [string] -and
      [System.DateTimeOffset]::TryParseExact(
        $snapshot.started, "yyyy-MM-dd'T'HH:mm:ss.fffffff'Z'",
        [System.Globalization.CultureInfo]::InvariantCulture,
        [System.Globalization.DateTimeStyles]::AssumeUniversal,
        [ref]$parsedStart)
    if ($snapshot.pid -isnot [int] -or $snapshot.pid -le 0 -or -not $validStart -or
        $snapshot.backend_path -isnot [string] -or
        $snapshot.backend_path -cnotmatch '\A[A-Za-z]:\\(?:[^\\\r\n"<>|:?*]+\\)*CryptoHunterBackend\.exe\z' -or
        $snapshot.backend_path -match '(?:^|\\)\.\.?\\' -or
        $snapshot.backend_sha256 -isnot [string] -or
        $snapshot.backend_sha256 -cnotmatch '\A[0-9a-f]{64}\z') {
      throw "niepelna lub nieprawidlowa punktowa tozsamosc uslugi Stage 10"
    }
  }
  if ($Observed.pid -ne $Expected.pid -or $Observed.started -cne $Expected.started -or
      $Observed.backend_path -cne $Expected.backend_path -or
      $Observed.backend_sha256 -cne $Expected.backend_sha256) {
    throw "zmienila sie punktowa tozsamosc procesu lub programu uslugi Stage 10"
  }
}
