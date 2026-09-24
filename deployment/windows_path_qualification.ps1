function Test-FullyQualifiedWindowsPath([AllowNull()][string]$Path) {
  # Do not normalize: callers must supply an already-qualified native Windows path.
  if ([string]::IsNullOrWhiteSpace($Path) -or $Path.Contains('/')) { return $false }

  $segments = $null
  if ($Path -match '^[A-Za-z]:\\') {
    $segments = @($Path.Substring(3).Split('\'))
  } elseif ($Path.StartsWith('\\')) {
    # Exactly two leading separators, followed by non-empty server and share names.
    if ($Path.StartsWith('\\\')) { return $false }
    $segments = @($Path.Substring(2).Split('\'))
    if ($segments.Count -lt 2 -or -not $segments[0] -or -not $segments[1]) { return $false }
  } else {
    return $false
  }

  foreach ($segment in $segments) {
    if (-not $segment -or $segment -eq '.' -or $segment -eq '..') { return $false }
    if ($segment.IndexOfAny([char[]]'<>:"|?*') -ge 0) { return $false }
    if ($segment.EndsWith(' ') -or $segment.EndsWith('.')) { return $false }
  }
  return $true
}
