# SBOM w CI

Pipeline CI generuje SBOM w formacie CycloneDX JSON jako osobny job `sbom`
w workflow `.github/workflows/ci.yml`.

## Gdzie powstaje SBOM

- Job: `Generate SBOM`
- Narzędzie: `anchore/sbom-action`
- Plik wynikowy: `sbom.cdx.json`
- Artefakt GitHub Actions: `sbom-cyclonedx`

## Jak pobrać

1. Otwórz run workflow `CI` w GitHub Actions.
2. Wejdź w sekcję artefaktów.
3. Pobierz artefakt `sbom-cyclonedx` zawierający `sbom.cdx.json`.

## Skan CVE zależności (quick win)

W tym samym workflow działa też job `dependency-vulnerability-scan`, który:

- skanuje zależności Pythona z `requirements.txt`,
- używa narzędzia `pip-audit` (advisories/PyPI),
- zapisuje raport JSON `pip-audit-report.json`.

Zakres tego quick wina obejmuje tylko to, co jest rozwiązywane przez
`requirements.txt` (w tym wpisy referencjonowane przez ten plik), bez
osobnego skanowania pełnego `pyproject.toml`/wszystkich extras.

### Gdzie widać wynik

- status joba `Dependency vulnerability scan` w runie GitHub Actions,
- artefakt `pip-audit-report` do pobrania z runa.

### Out of scope tej iteracji

- skan obrazów kontenerów i system packages,
- pełna polityka wyjątków/akceptacji ryzyka,
- centralna platforma security i korelacja wielu skanerów.
